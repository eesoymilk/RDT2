from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rdt_runner import RDTRunner
from models.ucvla.ucvla_rdt import UCVLARDTModel


class UCVLARDTRunner(nn.Module):
    """
    UCVLA Stage 1b: residual personalization wrapper around a base RDTRunner.

    Architecture
    ------------
    The base RDTRunner (frozen) provides the "what to do" prediction.
    UCVLARDTRunner trains a tiny residual model (same RDT DiT, with a per-user
    bias injected into the time embedding) to predict "how this user wants it."

    At inference:
        final_action = base_VLA(obs) + bias_model(obs, user_id)

    Training
    --------
    Requires y_cached: the base VLA's denoised action for each episode,
    precomputed offline with a fixed noise seed. The bias model is trained
    with flow matching on the residual:
        residual_gt = action_gt - y_cached
        loss = flow_matching_loss(bias_model(noisy_residual, t, user_id), residual_gt)

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
        noisy_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the bias model ODE to produce a residual action trajectory.

        This is NOT the final action — add it to base.predict_action() output,
        or use predict_personalized_action() which does that automatically.

        Args:
            user_id: (B,) LongTensor of user indices.
            state_tokens: (B, 1, state_dim) proprioceptive state.
            noisy_action: optional fixed initial noise for deterministic caching.

        Returns:
            (B, horizon, action_dim) residual action trajectory.
        """
        lang_cond, img_cond, _, state_cond = self.base.adapt_conditions(
            lang_tokens, img_tokens, None, state_tokens
        )

        batch_size = state_cond.shape[0]
        device = state_cond.device
        dtype = state_cond.dtype

        if noisy_action is None:
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

    @torch.no_grad()
    def predict_personalized_action(
        self,
        user_id: torch.Tensor,
        state_tokens: torch.Tensor,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_kv_cache: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        img_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full personalized inference: base VLA output + bias model residual.

        Returns:
            (B, horizon, action_dim) personalized action trajectory.
        """
        y_base = self.base.predict_action(
            lang_tokens=lang_tokens,
            lang_kv_cache=lang_kv_cache,
            lang_attn_mask=lang_attn_mask,
            img_tokens=img_tokens,
            state_tokens=state_tokens,
        )
        y_residual = self.predict_action(
            user_id=user_id,
            state_tokens=state_tokens,
            lang_tokens=lang_tokens,
            lang_kv_cache=lang_kv_cache,
            lang_attn_mask=lang_attn_mask,
            img_tokens=img_tokens,
        )
        return y_base + y_residual

    # =========  Training  ===========

    def compute_residual_loss(
        self,
        y_cached: torch.Tensor,
        action_gt: torch.Tensor,
        user_id: torch.Tensor,
        state_tokens: torch.Tensor,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_kv_cache: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        img_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Stage 1b training loss (flow matching on the residual action).

        The bias model is trained to predict the residual:
            residual_gt = action_gt - y_cached

        using the same flow matching objective as the base VLA.

        Args:
            y_cached:   (B, horizon, action_dim) base VLA predictions, precomputed
                        offline with a fixed noise seed (frozen, no grad needed).
            action_gt:  (B, horizon, action_dim) ground-truth personalized trajectories.
            user_id:    (B,) LongTensor of user indices.
            state_tokens: (B, 1, state_dim) proprioceptive state.

        Returns:
            Scalar loss tensor.
        """
        dtype = action_gt.dtype
        device = action_gt.device
        batch_size = action_gt.shape[0]

        # What the bias model needs to add on top of the base VLA
        residual_gt = action_gt - y_cached.to(dtype=dtype, device=device)

        # Flow matching: interpolate between noise and residual_gt
        noise = torch.randn(residual_gt.shape, dtype=dtype, device=device)
        timesteps = self.base.sample_timesteps(batch_size, device)
        t = timesteps.view(-1, 1, 1)
        noisy_residual = residual_gt * t + noise * (1 - t)

        lang_cond, img_cond, residual_traj, state_cond = self.base.adapt_conditions(
            lang_tokens, img_tokens, noisy_residual, state_tokens
        )
        condition_inputs = self.base._prepare_condition_inputs(
            lang_cond=lang_cond,
            lang_kv_cache=lang_kv_cache,
            lang_attn_mask=lang_attn_mask,
            img_cond=img_cond,
            state_cond=state_cond,
        )

        pred = self.ucvla_model(
            x=residual_traj,
            t=timesteps,
            user_id=user_id,
            **condition_inputs,
        )

        # Flow matching velocity target: direction from noise toward residual_gt
        target = residual_gt - noise
        return F.mse_loss(pred, target)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_residual_loss(*args, **kwargs)
