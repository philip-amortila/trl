import random as _random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models.utils import unwrap_model_for_generation
from ..utils import empty_cache
from .gkd_trainer import GKDTrainer


class CorrectionNetwork(nn.Module):
    """
    Learnable correction ζ: maps student hidden states to per-token Q-corrections.

    Architecture: a single linear projection (hidden_size → vocab_size), initialised
    with zero weights so training starts with no correction applied.
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, hidden_size)
        Returns:
            zeta: (B, T, vocab_size)
        """
        return self.linear(hidden_states)


class OPDTrainer(GKDTrainer):
    """
    Trainer implementing three variants of the OPD (On-Policy Distillation) loss,
    selected via the ``mode`` argument.

    All three modes share the same outer loop inherited from GKDTrainer: at each
    training step the student generates a full trajectory (sequence of tokens)
    on-policy, and *then* the loss is computed over that trajectory.  What differs
    is **how** the per-token distillation signal is formed.

    ─────────────────────────────────────────────────────────────────
    Mode 1 – "expectation"  (default, mode="expectation")
    ─────────────────────────────────────────────────────────────────
    Loss per token:

        l_i = log π_e(a_{e,i} | x_i)  -  Σ_a π_s(a | x_i) · log π_e(a | x_i)

    where  a_{e,i} ~ π_e(· | x_i)  is a fresh sample from the *teacher*.

    • The second term is the **exact expectation** E_{a~π_s}[log π_e(a | x_i)],
      computed analytically by summing over the full vocabulary.  Its gradient
      w.r.t. student parameters flows analytically through π_s (no REINFORCE
      needed).
    • The first term  log π_e(a_{e,i})  is a **constant** w.r.t. student params
      (it depends only on the teacher) and acts as a control variate / baseline
      that centres the loss.
    • Net effect: the gradient pushes π_s to put more mass on tokens that the
      teacher scores highly.

    When ``trust_region=True``, the analytical update is replaced by a
    REINFORCE estimator (using the teacher sample as baseline and the student's
    on-policy token as the action) wrapped in the PPO clipped surrogate, making
    the trust-region mechanism consistent across all three REINFORCE modes.

    ─────────────────────────────────────────────────────────────────
    Mode 2 – "stochastic"  (mode="stochastic")
    ─────────────────────────────────────────────────────────────────
    Loss per token:

        l_i = log π_e(a_{e,i} | x_i)  -  log π_e(a_{s,i} | x_i)

    where
        a_{e,i} ~ π_e  is sampled from the *teacher*  (Monte Carlo baseline)
        a_{s,i}         is the *student's* on-policy token (from inputs["labels"])

    Because a_{s,i} is a discrete integer (no gradient), backpropagation uses
    the REINFORCE / score-function estimator:

        ∇ l_i  ≈  (log π_e(a_{e,i}) - log π_e(a_{s,i})).detach()
                   · ∇ log π_s(a_{s,i} | x_i)

    i.e. the advantage  (log π_e(a_{e,i}) - log π_e(a_{s,i}))  is treated as a
    fixed scalar reward and the gradient carrier is  log π_s(a_{s,i}).

    Compared to mode="expectation" this is a *higher-variance* but
    *gradient-unbiased* estimator.

    ─────────────────────────────────────────────────────────────────
    Mode 3 – "entropy_baseline"  (mode="entropy_baseline")
    ─────────────────────────────────────────────────────────────────
    Loss per token:

        l_i = -H(π_e | x_i)  -  log π_e(a_{s,i} | x_i)

    where  -H(π_e) = Σ_a π_e(a | x_i) · log π_e(a | x_i)  is the **exact**
    expected log-probability under the teacher (i.e. the negative teacher entropy),
    and  a_{s,i}  is again the student's on-policy token.

    Backpropagation still uses REINFORCE:

        ∇ l_i  ≈  (-H(π_e | x_i) - log π_e(a_{s,i} | x_i)).detach()
                   · ∇ log π_s(a_{s,i} | x_i)

    The key difference from mode="stochastic":  instead of a *Monte Carlo*
    estimate  log π_e(a_{e,i})  for the baseline we use the **exact expectation**
    -H(π_e) = E_{a~π_e}[log π_e(a)].  This removes the variance from the
    baseline sample, giving a lower-variance gradient estimate while remaining
    unbiased.

    Pseudocode equivalent:

        teacher_probs       = softmax(teacher_logits)
        expected_logprob    = (teacher_probs * log_softmax(teacher_logits)).sum(-1)
                            # = -H(π_e),  shape (B, T)
        advantage           = expected_logprob - log_pi_e(a_{s,i})
        loss                = advantage.detach() * log_pi_s(a_{s,i})

    ─────────────────────────────────────────────────────────────────
    Mode 4 – "softmax"  (mode="softmax")
    ─────────────────────────────────────────────────────────────────
    Implements the Q-function maximization step of Algorithm 4 (softmax
    mode) from the Q^{π_E}-realizability paper.

    The student logits are treated directly as the Q-function Q_θ.
    The objective maximizes the advantage of the expert action over the
    expected Q under the **current** (fixed) student policy:

        max_θ  Q_θ(x, a_{e,i}) − E_{a∼π_θ_fixed}[Q_θ(x, a)]

    Equivalently, we minimize:

        l_i = E_{a∼π_θ_fixed}[Q_θ(x, a)] − Q_θ(x, a_{e,i})

    where
        a_{e,i} ~ π_e  is sampled from the *teacher*
        Q_θ(x, a)      = student_logits[..., a]  (the student IS Q_θ)
        π_θ_fixed       = softmax(Q_θ).detach()  (current policy, frozen)

    Crucially, the gradient flows through the student *logits* (Q-values),
    while the policy probability weights are **detached**.  This is the
    opposite of mode="expectation", where the gradient flows through the
    student *probabilities* against a fixed teacher Q.

    The policy at the next step is implicitly π_{k+1}(·|x) = softmax(Q_θ),
    i.e. the greedy update is folded into the model's own forward pass.

    Pseudocode equivalent:

        expert_probs        = softmax(teacher_logits)           # π_E(a|x)
        student_probs_fixed = softmax(student_logits).detach()  # π_θ, no grad
        loss = -((expert_probs - student_probs_fixed) * student_logits).sum(-1)
        #      maximises  E_{π_E}[Q_θ] − E_{π_θ}[Q_θ]  analytically,
        #      using the full teacher distribution instead of a single a_E sample.

    ─────────────────────────────────────────────────────────────────
    Trust-region option  (trust_region=True)
    ─────────────────────────────────────────────────────────────────
    Applies to modes "expectation", "stochastic", and "entropy_baseline".

    Replaces the plain REINFORCE gradient carrier  advantage · ∇ log π_s(a_{s,i})
    with the PPO clipped surrogate:

        r_i = π_s_new(a_{s,i}) / π_s_old(a_{s,i})
            = exp( log π_s_new(a_{s,i}) − log π_s_old(a_{s,i}) )

        loss_i = max( r_i · adv_i,  clip(r_i, 1−ε, 1+ε) · adv_i )

    Note on sign convention: the code minimises  advantage · log π_s  where
    advantage > 0 means "student token is *worse* than the baseline → decrease
    its probability".  The PPO pessimistic bound in this convention is
    torch.max (not torch.min) of the clipped and unclipped objectives.

    The behaviour-policy log-probs  log π_s_old(a_{s,i})  are recorded with a
    no-grad forward pass immediately after trajectory collection (before any
    inner gradient steps) and stored in the replay buffer so they remain fixed
    across all ``num_inner_steps`` inner updates.

    ``trust_region`` is ignored for mode="softmax" (which does not use
    REINFORCE).
    """

    def __init__(
        self,
        *args,
        mode: str = "expectation",
        num_inner_steps: int = 1,
        replay_buffer_size: int = 1,
        trust_region: bool = False,
        ppo_clip_eps: float = 0.2,
        use_correction: bool = False,
        correction_alpha: float = 0.2,
        correction_lr: float = 1e-3,
        **kwargs,
    ):
        """
        Args:
            mode: One of "expectation" | "stochastic" | "entropy_baseline" | "softmax".
                  See class docstring for a description of each mode.
            num_inner_steps: Number of gradient steps on Q per outer training step
                  (the ``L`` inner loop in Algorithm 4).  Default ``1`` recovers
                  the original single-step behaviour.
            replay_buffer_size: Maximum number of past batches kept in the replay
                  buffer (rounds ``1 … k`` in Algorithm 4).  Default ``1`` uses
                  only the current batch (no replay from older policies).
            trust_region: If ``True``, replaces the plain REINFORCE update with a
                  PPO-style clipped surrogate objective for modes "expectation",
                  "stochastic", and "entropy_baseline".  An extra no-grad forward
                  pass is performed at data-collection time to record the
                  behaviour-policy log-probabilities.  Ignored for mode="softmax".
            ppo_clip_eps: Clipping radius ε for the PPO importance-ratio.  Only
                  used when ``trust_region=True``.  Default ``0.2``.
            use_correction: If ``True``, learns a correction network ζ (Algorithm 8)
                  that refines the teacher Q-function before each PPO step.  A
                  separate Adam optimizer is maintained for ζ.
            correction_alpha: Mixing coefficient α in Q̃E = (1−α)·QE + α·ζ.
                  Default ``0.2``.
            correction_lr: Learning rate for the ζ Adam optimizer.  Default ``1e-3``.
        """
        valid_modes = {"expectation", "stochastic", "entropy_baseline", "softmax"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.num_inner_steps = num_inner_steps
        self.trust_region = trust_region
        self.ppo_clip_eps = ppo_clip_eps
        # CPU-side ring buffer of past batches (prompt + student trajectory).
        self._replay_buffer: deque[dict] = deque(maxlen=max(replay_buffer_size, 1))

        # ── Algorithm 8: learnable correction ζ ──────────────────────────────
        self.use_correction = use_correction
        self.correction_alpha = correction_alpha
        if use_correction:
            hidden_size = self.model.config.hidden_size
            vocab_size = self.model.config.vocab_size
            device = next(self.model.parameters()).device
            self.correction_network = CorrectionNetwork(hidden_size, vocab_size).to(device)
            self.correction_optimizer = torch.optim.Adam(
                self.correction_network.parameters(), lr=correction_lr
            )

    # ── Algorithm 8: ζ correction update ────────────────────────────────────

    def _update_correction(self, model, inputs: dict) -> None:
        """
        Perform one gradient step on ζ maximizing (Algorithm 8):

            max_ζ  Σ_{X∈D} Σ_{a∈A} ζ(X, a) · (softmax(QE(X, a)) − π_θ(a|X))

        Both θ (student) and teacher parameters are frozen during this step;
        only the correction network ζ is updated.
        """
        prompt_lengths = inputs["prompts"].shape[1]
        shifted_labels = inputs["labels"][:, prompt_lengths:]  # (B, T')
        mask = shifted_labels != -100
        if not mask.any():
            return

        # Forward student (no grad for θ) – need hidden states for ζ
        model.eval()
        with torch.no_grad():
            student_out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            teacher_out = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        model.train()

        # Slice to the response tokens (mirroring compute_loss alignment)
        hidden = student_out.hidden_states[-1][:, prompt_lengths - 1 : -1, :]  # (B, T', H)
        student_logits = student_out.logits[:, prompt_lengths - 1 : -1, :] / self.temperature
        teacher_logits = teacher_out.logits[:, prompt_lengths - 1 : -1, :] / self.temperature

        min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
        student_logits = student_logits[..., :min_vocab]
        teacher_logits = teacher_logits[..., :min_vocab]

        # Probabilities are detached: ζ gradient must not flow back into θ or teacher
        teacher_probs = F.softmax(teacher_logits, dim=-1).detach()  # (B, T', V)
        student_probs = F.softmax(student_logits, dim=-1).detach()  # (B, T', V)
        diff = teacher_probs - student_probs                         # (B, T', V)

        # ζ forward (grad flows through correction_network only)
        zeta = self.correction_network(hidden.detach())  # (B, T', vocab_size)
        zeta = zeta[..., :min_vocab]

        # Objective: maximize Σ_{X,a} ζ(X,a)·diff(X,a)  →  minimise negative mean
        obj = (zeta * diff).sum(-1)   # (B, T')
        loss_zeta = -obj[mask].mean()

        self.correction_optimizer.zero_grad()
        loss_zeta.backward()
        self.correction_optimizer.step()

    # ── Replay buffer helpers ─────────────────────────────────────────────────

    def _push_to_replay_buffer(self, inputs: dict) -> None:
        """Offload a batch to CPU and push it into the replay buffer."""
        self._replay_buffer.append(
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        )

    def _sample_from_replay_buffer(self) -> dict:
        """Return a uniformly random batch from the buffer, moved to the training device."""
        return self._prepare_inputs(dict(_random.choice(self._replay_buffer)))

    # ── Training step with L inner updates ───────────────────────────────────

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Outer step of Algorithm 4:

          1. (Optionally) generate a fresh on-policy student trajectory.
          2. Push the batch into the replay buffer.
          3. Run ``num_inner_steps`` (L) gradient updates on batches sampled
             uniformly from the replay buffer, accumulating data from all past
             rounds as in the pseudocode's sum over s = 1 … k.

        When ``num_inner_steps=1`` and ``replay_buffer_size=1`` the behaviour is
        identical to the original GKDTrainer single-step update.

        After the L inner steps the gradients are zeroed, so the outer
        Trainer's ``optimizer.step()`` call becomes a no-op and the
        lr_scheduler advances exactly once per outer step (as expected).
        """
        # ── 1. On-policy generation (mirrors GKDTrainer.training_step) ───────
        if self.seq_kd:
            with unwrap_model_for_generation(
                self.teacher_model,
                self.accelerator,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_teacher:
                new_ids, new_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_teacher,
                    inputs,
                    self.generation_config,
                    self.processing_class.pad_token_id,
                )
            inputs["input_ids"] = new_ids
            inputs["attention_mask"] = new_mask
            inputs["labels"] = new_labels

        if _random.random() <= self.lmbda:
            with unwrap_model_for_generation(
                model,
                self.accelerator,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model:
                new_ids, new_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model,
                    inputs,
                    self.generation_config,
                    self.processing_class.pad_token_id,
                )
            inputs["input_ids"] = new_ids
            inputs["attention_mask"] = new_mask
            inputs["labels"] = new_labels

        # ── 2a. Record behaviour-policy log-probs for PPO trust region ──────
        if self.trust_region and self.mode != "softmax":
            with torch.no_grad():
                old_out = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            prompt_lengths = inputs["prompts"].shape[1]
            # Align with the shifted logits used in compute_loss / opd_loss
            old_logits = old_out.logits[:, prompt_lengths - 1 : -1, :] / self.temperature
            old_lp = F.log_softmax(old_logits, dim=-1)  # (B, T', V)

            shifted_labels = inputs["labels"][:, prompt_lengths:]  # (B, T')
            sa = shifted_labels.clone()
            sa[sa == -100] = 0
            old_log_probs = old_lp.gather(-1, sa.unsqueeze(-1)).squeeze(-1)  # (B, T')
            # Zero padded positions so they never affect the ratio
            old_log_probs[shifted_labels == -100] = 0.0
            inputs["old_log_probs"] = old_log_probs

        # ── 2b. Update ζ correction network (Algorithm 8) ────────────────────
        if self.use_correction:
            self._update_correction(model, inputs)

        # ── 2c. Store freshly collected batch (round k data) ──────────────────
        self._push_to_replay_buffer(inputs)

        # ── 3. L inner gradient steps over the accumulated replay buffer ──────
        model.train()
        total_loss = torch.tensor(0.0, device=self.args.device)

        for _ in range(self.num_inner_steps):
            # Sample uniformly from rounds 1 … k (current + all past batches).
            batch = self._sample_from_replay_buffer()

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, batch, num_items_in_batch=num_items_in_batch)

            self.accelerator.backward(loss)

            if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.detach()

        return total_loss / self.num_inner_steps

    @staticmethod
    def opd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        temperature=1.0,
        mode="expectation",
        reduction="batchmean",
        trust_region=False,
        ppo_clip_eps=0.2,
        old_log_probs=None,
    ):
        """
        Compute the OPD loss for a single batch.

        Args:
            student_logits: (B, T, V)  raw logits from the student model
            teacher_logits: (B, T, V)  raw logits from the teacher model
            labels:         (B, T)     token ids of the student's on-policy
                                       trajectory; positions to ignore carry -100
            temperature:    softmax temperature applied to both models
            mode:           "expectation" | "stochastic" | "entropy_baseline"
                            (see class docstring for details)
            reduction:      "batchmean" | "sum" | "mean" | "none"
            trust_region:   If ``True``, replace the REINFORCE gradient with a
                            PPO clipped surrogate for modes "expectation",
                            "stochastic", and "entropy_baseline".  Ignored for
                            mode="softmax".
            ppo_clip_eps:   Clipping radius ε for the PPO importance-ratio
                            (only used when ``trust_region=True``).
            old_log_probs:  (B, T) per-token log-probs of the *behaviour* policy
                            (i.e. the student at data-collection time), aligned
                            with ``labels``.  Required when ``trust_region=True``;
                            falls back to plain REINFORCE when ``None``.

        Returns:
            Scalar loss (or per-token tensor when reduction="none").
        """
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Truncate to the smaller vocabulary (handles embedding-padding differences
        # across model families; shared real tokens are always in the same positions).
        min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
        student_logits = student_logits[..., :min_vocab]
        teacher_logits = teacher_logits[..., :min_vocab]

        student_log_probs = F.log_softmax(student_logits, dim=-1)  # log π_s(a | x)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)  # log π_e(a | x)

        if mode == "expectation":
            # ── Mode 1: analytical gradient through π_s ──────────────────────
            # Sample a_{e,i} ~ π_e for a per-position baseline (constant w.r.t. student).
            B, T, V = teacher_log_probs.shape
            expert_actions = torch.multinomial(
                teacher_log_probs.exp().view(B * T, V), num_samples=1
            ).view(B, T)  # (B, T)

            # log π_e(a_{e,i} | x_i)  — teacher constant, acts as baseline
            log_pi_e_expert = teacher_log_probs.gather(-1, expert_actions.unsqueeze(-1)).squeeze(-1)

            if trust_region and old_log_probs is not None:
                # PPO path: switch to REINFORCE on student tokens so we can form
                # a proper importance ratio, using the teacher sample as baseline.
                student_actions = labels.clone()
                student_actions[student_actions == -100] = 0

                log_pi_e_student = teacher_log_probs.gather(
                    -1, student_actions.unsqueeze(-1)
                ).squeeze(-1)  # (B, T)

                log_pi_s_student = student_log_probs.gather(
                    -1, student_actions.unsqueeze(-1)
                ).squeeze(-1)  # (B, T)

                advantage = (log_pi_e_expert - log_pi_e_student).detach()
                ratio = (log_pi_s_student - old_log_probs).exp()
                clipped_ratio = ratio.clamp(1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps)
                # Sign convention: advantage > 0 → decrease prob → minimise max(...)
                loss = torch.max(ratio * advantage, clipped_ratio * advantage)
            else:
                # Exact expectation  E_{a~π_s}[log π_e(a | x_i)]  — gradient flows analytically
                expected_log_pi_e = (student_log_probs.exp() * teacher_log_probs).sum(-1)  # (B, T)
                loss = log_pi_e_expert - expected_log_pi_e

        elif mode == "stochastic":
            # ── Mode 2: REINFORCE with Monte Carlo teacher baseline ───────────
            # Sample a_{e,i} ~ π_e for the baseline.
            B, T, V = teacher_log_probs.shape
            expert_actions = torch.multinomial(
                teacher_log_probs.exp().view(B * T, V), num_samples=1
            ).view(B, T)  # (B, T)

            # log π_e(a_{e,i} | x_i)  — Monte Carlo baseline (part of advantage)
            log_pi_e_expert = teacher_log_probs.gather(-1, expert_actions.unsqueeze(-1)).squeeze(-1)

            # a_{s,i}: student's on-policy tokens stored in labels.
            # Replace -100 padding sentinel with 0 for safe gather; masked out below.
            student_actions = labels.clone()
            student_actions[student_actions == -100] = 0

            # log π_e(a_{s,i} | x_i)  — part of the detached advantage
            log_pi_e_student = teacher_log_probs.gather(
                -1, student_actions.unsqueeze(-1)
            ).squeeze(-1)  # (B, T)

            # log π_s(a_{s,i} | x_i)  — gradient carrier (has grad w.r.t. student params)
            log_pi_s_student = student_log_probs.gather(
                -1, student_actions.unsqueeze(-1)
            ).squeeze(-1)  # (B, T)

            # Advantage is the same regardless of trust_region
            advantage = (log_pi_e_expert - log_pi_e_student).detach()

            if trust_region and old_log_probs is not None:
                ratio = (log_pi_s_student - old_log_probs).exp()
                clipped_ratio = ratio.clamp(1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps)
                loss = torch.max(ratio * advantage, clipped_ratio * advantage)
            else:
                # Plain REINFORCE: treat advantage as a fixed scalar reward
                loss = advantage * log_pi_s_student

        elif mode == "entropy_baseline":
            # ── Mode 3: REINFORCE with exact teacher-entropy baseline ─────────
            # Instead of sampling a_{e,i} ~ π_e for the baseline, compute the
            # *exact* expected teacher log-prob  -H(π_e) = Σ_a π_e(a) log π_e(a).
            # This removes sampling noise from the baseline → lower variance.

            # -H(π_e | x_i) = E_{a~π_e}[log π_e(a | x_i)], shape (B, T)
            neg_teacher_entropy = (teacher_log_probs.exp() * teacher_log_probs).sum(-1)

            # a_{s,i}: student's on-policy tokens stored in labels.
            student_actions = labels.clone()
            student_actions[student_actions == -100] = 0

            # log π_e(a_{s,i} | x_i)  — per-position reward signal
            log_pi_e_student = teacher_log_probs.gather(
                -1, student_actions.unsqueeze(-1)
            ).squeeze(-1)  # (B, T)

            # log π_s(a_{s,i} | x_i)  — gradient carrier
            log_pi_s_student = student_log_probs.gather(
                -1, student_actions.unsqueeze(-1)
            ).squeeze(-1)  # (B, T)

            # Advantage = (exact baseline) - (per-token reward) = -H(π_e) - log π_e(a_{s,i})
            advantage = (neg_teacher_entropy - log_pi_e_student).detach()

            if trust_region and old_log_probs is not None:
                ratio = (log_pi_s_student - old_log_probs).exp()
                clipped_ratio = ratio.clamp(1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps)
                loss = torch.max(ratio * advantage, clipped_ratio * advantage)
            else:
                # Plain REINFORCE
                loss = advantage * log_pi_s_student

        elif mode == "softmax":
            # ── Mode 4: Algorithm 4 softmax – Q-function maximization ────────
            # Treats student logits as Q_θ and maximises the analytical objective:
            #
            #   E_{a∼π_E}[Q_θ(x,a)] − E_{a∼π_θ_fixed}[Q_θ(x,a)]
            #   = Σ_a (π_E(a) − π_θ_fixed(a)) · Q_θ(x,a)
            #   = ((expert_probs − student_probs_fixed) * student_logits).sum(-1)
            #
            # π_θ_fixed is DETACHED so gradient only flows through student_logits
            # (Q-values), not through the policy distribution weights.
            # Using the full teacher distribution avoids the sampling noise of a
            # single a_E ~ π_E sample.

            expert_probs = teacher_log_probs.exp()          # π_E(a|x),   no grad
            student_probs_fixed = student_log_probs.exp().detach()  # π_θ, no grad

            # Minimise −(E_{π_E}[Q_θ] − E_{π_θ_fixed}[Q_θ])
            loss = -((expert_probs - student_probs_fixed) * student_logits).sum(-1)

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        # ── Masking and reduction ─────────────────────────────────────────────
        if labels is not None:
            mask = labels != -100
            loss = loss[mask]

        if reduction == "batchmean":
            return loss.sum() / mask.sum() if labels is not None else loss.sum() / loss.size(0)
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "mean":
            return loss.mean()
        else:
            return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=self.use_correction,
        )

        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
        shifted_teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]

        # ── Algorithm 8: apply ζ correction to teacher Q-function ────────────
        if self.use_correction:
            # Hidden state of the last transformer layer, aligned with response tokens.
            hidden = student_outputs.hidden_states[-1][:, prompt_lengths - 1 : -1, :]
            # ζ is frozen here (PPO step should not update ζ parameters).
            with torch.no_grad():
                zeta = self.correction_network(hidden)  # (B, T', vocab_size)

            min_vocab = min(shifted_student_logits.size(-1), shifted_teacher_logits.size(-1))
            zeta = zeta[..., :min_vocab]
            # Q̃E = (1 − α)·QE + α·ζ
            shifted_teacher_logits = (
                (1 - self.correction_alpha) * shifted_teacher_logits[..., :min_vocab]
                + self.correction_alpha * zeta
            )
            shifted_student_logits = shifted_student_logits[..., :min_vocab]

        # Retrieve behaviour-policy log-probs stored at collection time (PPO).
        old_log_probs = inputs.get("old_log_probs", None)

        loss = self.opd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            temperature=self.temperature,
            mode=self.mode,
            trust_region=self.trust_region,
            ppo_clip_eps=self.ppo_clip_eps,
            old_log_probs=old_log_probs,
        )

        empty_cache()

        return (loss, student_outputs) if return_outputs else loss
