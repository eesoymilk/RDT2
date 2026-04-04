from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rdt_runner import RDTRunner
from models.ucvla.ucvla_rdt import UCVLARDTModel


class UCVLARDTRunner(nn.Module):
    """
    UCVLA Stage 1: thin training/inference wrapper around UCVLARDTModel.

    Runs a single ODE through the user-conditioned RDT — no residual, no
    base+bias composition. The per-user bias is injected into t_embedder
    inside UCVLARDTModel, which modulates every adaLN block and shifts the
    trajectory toward user preferences in one pass.

    Training
    --------
    Direct flow matching on action_gt:
        loss = MSE(ucvla_model(noisy_action, t, user_id, obs), action_gt - noise)

    No y_cached, no precomputed base predictions.

    Trainable parameters
    --------------------
    Only UCVLARDTModel.user_bias and UCVLARDTModel.bias_proj. Call freeze_base()
    after construction to lock everything else.

    Args:
        base:    A loaded RDTRunner (e.g. from RDTRunner.from_pretrained(...)).
        n_users: Number of distinct users.
        d_bias:  Per-user bias dimension (default 64).
    """

    def __init__(self, base: RDTRunner, n_users: int, d_bias: int = 64):
        super().__init__()
        self.base = base
        # ucvla_model.rdt is the SAME object as base.model — no weight copy.
        self.ucvla_model = UCVLARDTModel(base.model, n_users, d_bias)

    # =========  Parameter management  ===========

    def freeze_base(self):
        """Freeze all base RDTRunner weights. Only user_bias + bias_proj remain trainable."""
        for p in self.base.parameters():
            p.requires_grad_(False)

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Returns only the UCVLA-specific trainable parameters."""
        return (
            list(self.ucvla_model.user_bias.parameters())
            + list(self.ucvla_model.bias_proj.parameters())
        )

    # =========  Inference  ===========

    @torch.no_grad()
    def predict_action(
        self,
        user_id: torch.Tensor,
        state_tokens: torch.Tensor,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_kv_cache: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        img_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the user-conditioned ODE to produce a personalized action trajectory.

        Args:
            user_id:      (B,) LongTensor of user indices.
            state_tokens: (B, 1, state_dim) proprioceptive state.

        Returns:
            (B, horizon, action_dim) personalized action trajectory.
        """
        lang_cond, img_cond, _, state_cond = self.base.adapt_conditions(
            lang_tokens, img_tokens, None, state_tokens
        )

        batch_size = state_cond.shape[0]
        device = state_cond.device
        dtype = state_cond.dtype

        noisy_action = torch.randn(
            (batch_size, self.base.pred_horizon, self.base.action_dim),
            dtype=dtype, device=device,
        )

        condition_inputs = self.base._prepare_condition_inputs(
            lang_cond=lang_cond,
            lang_kv_cache=lang_kv_cache,
            lang_attn_mask=lang_attn_mask,
            img_cond=img_cond,
            state_cond=state_cond,
        )

        timestep = torch.tensor([0.0], dtype=dtype, device=device)
        step_size = 1.0 / self.base.num_inference_timesteps

        for _ in range(self.base.num_inference_timesteps):
            action_traj = self.base.act_adaptor(noisy_action)
            model_output = self.ucvla_model(
                x=action_traj,
                t=timestep,
                user_id=user_id,
                **condition_inputs,
            )
            noisy_action = model_output * step_size + noisy_action
            timestep += step_size

        return noisy_action

    # =========  Training  ===========

    def compute_loss(
        self,
        action_gt: torch.Tensor,
        user_id: torch.Tensor,
        state_tokens: torch.Tensor,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_kv_cache: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        img_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Stage 1 training loss (direct flow matching on action_gt).

        Args:
            action_gt:    (B, horizon, action_dim) ground-truth personalized trajectories.
            user_id:      (B,) LongTensor of user indices.
            state_tokens: (B, 1, state_dim) proprioceptive state.

        Returns:
            Scalar loss tensor.
        """
        dtype = action_gt.dtype
        device = action_gt.device
        batch_size = action_gt.shape[0]

        noise = torch.randn(action_gt.shape, dtype=dtype, device=device)
        timesteps = self.base.sample_timesteps(batch_size, device)
        t = timesteps.view(-1, 1, 1).to(dtype=dtype)
        noisy_action = action_gt * t + noise * (1 - t)

        lang_cond, img_cond, action_traj, state_cond = self.base.adapt_conditions(
            lang_tokens, img_tokens, noisy_action, state_tokens
        )
        condition_inputs = self.base._prepare_condition_inputs(
            lang_cond=lang_cond,
            lang_kv_cache=lang_kv_cache,
            lang_attn_mask=lang_attn_mask,
            img_cond=img_cond,
            state_cond=state_cond,
        )

        pred = self.ucvla_model(
            x=action_traj,
            t=timesteps,
            user_id=user_id,
            **condition_inputs,
        )

        target = action_gt - noise
        return F.mse_loss(pred, target)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)
