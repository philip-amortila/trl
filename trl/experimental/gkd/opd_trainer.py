import random as _random
from collections import deque

import torch
import torch.nn.functional as F

from ...models.utils import unwrap_model_for_generation
from ..utils import empty_cache
from .gkd_trainer import GKDTrainer


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
    """

    def __init__(
        self,
        *args,
        mode: str = "expectation",
        num_inner_steps: int = 1,
        replay_buffer_size: int = 1,
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
        """
        valid_modes = {"expectation", "stochastic", "entropy_baseline", "softmax"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.num_inner_steps = num_inner_steps
        # CPU-side ring buffer of past batches (prompt + student trajectory).
        self._replay_buffer: deque[dict] = deque(maxlen=max(replay_buffer_size, 1))

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

        # ── 2. Store freshly collected batch (round k data) ───────────────────
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
        student_logits, teacher_logits, labels=None, temperature=1.0, mode="expectation", reduction="batchmean"
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

        Returns:
            Scalar loss (or per-token tensor when reduction="none").
        """
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

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

            # REINFORCE: treat advantage as a fixed scalar reward
            advantage = (log_pi_e_expert - log_pi_e_student).detach()
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

            # REINFORCE: advantage = (exact baseline) - (per-token reward)
            # = -H(π_e) - log π_e(a_{s,i})
            advantage = (neg_teacher_entropy - log_pi_e_student).detach()
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

        loss = self.opd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            temperature=self.temperature,
            mode=self.mode,
        )

        empty_cache()

        return (loss, student_outputs) if return_outputs else loss
