from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from models.rdt.model import RDT


class UCVLARDTModel(nn.Module):
    """
    Wraps a vanilla RDT and injects a per-user bias into the time embedding.

    The bias is added immediately after t_embedder(t) — before the timestep is
    concatenated with the proprioception state and before it reaches any adaLN
    block. This single vector addition fans out into all 9 adaLN parameters
    (shift/scale/gate for self-attn, cross-attn, FFN) in every RDTBlock.

    The wrapped RDT is a shared reference — no weights are copied.
    Only user_bias and bias_proj are trainable; everything in rdt stays frozen
    when freeze_base() is called on UCVLARDTRunner.

    Args:
        rdt:     A vanilla RDT instance (e.g. from a loaded RDTRunner.model).
        n_users: Number of distinct users (size of the embedding table).
        d_bias:  Dimension of the per-user bias vector. Keep small (32–128).
    """

    def __init__(self, rdt: RDT, n_users: int, d_bias: int = 64):
        super().__init__()
        self.rdt = rdt
        hidden_size = rdt.hidden_size
        dtype = rdt.dtype

        self.user_bias = nn.Embedding(n_users, d_bias)
        self.bias_proj = nn.Linear(d_bias, hidden_size)

        # Cast to match the model dtype (bfloat16 by default)
        self.user_bias = self.user_bias.to(dtype=dtype)
        self.bias_proj = self.bias_proj.to(dtype=dtype)

        # Zero-init: bias starts as identity so the wrapped model behaves
        # exactly like the base VLA until training shifts user_bias.
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.bias_proj.weight)
        nn.init.zeros_(self.bias_proj.bias)

    @property
    def hidden_size(self) -> int:
        return self.rdt.hidden_size

    @property
    def dtype(self):
        return self.rdt.dtype

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, T, D) noisy action trajectory (already adapted).
            t:       (B,) or (1,) diffusion timesteps.
            user_id: (B,) LongTensor of user indices, or None for no bias.
            **kwargs: passed through to rdt.forward (lang_c, img_c, state_c, masks…).

        Returns:
            (B, T, action_dim) predicted velocity (same shape as rdt.forward output).
        """
        if user_id is None:
            return self.rdt(x, t, **kwargs)

        # Compute bias on the correct device
        device = next(self.bias_proj.parameters()).device
        bias = self.bias_proj(self.user_bias(user_id.to(device)))  # (B, hidden_size)

        # One-shot hook: adds bias to t_embedder output, then removes itself.
        # Registered inside the call so it never persists across forward passes.
        def _hook(module, inp, out):
            return out + bias

        handle = self.rdt.t_embedder.register_forward_hook(_hook)
        try:
            result = self.rdt(x, t, **kwargs)
        finally:
            handle.remove()

        return result
