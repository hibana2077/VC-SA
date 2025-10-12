# -*- coding: utf-8 -*-

"""
SQuaRe-Fuse: Sliced-Quantile & Quadratic-trend Robust Fusion

Drop-in temporal fusion layer for ViT token sequences without using Attention/SSM/Graphs.

I/O contract
  - forward(x:[B,T,N,D], valid_mask:[B,T,N]|None) -> h:[B,T,N,D]

Notes
  - Keeps projection directions approximately orthonormal by QR each forward (optional)
  - Uses differentiable soft-quantiles on sorted time series per projected direction
  - Adds low-order Legendre trend coefficients along the time axis
  - Maps concatenated stats back to D dims and fuses via residual with a learnable gate beta
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _legendre_basis(T: int, order: int, device, dtype):
    """Build discrete Legendre basis rows P_0..P_order over t in [-1,1] with L2 row-norm 1.

    Returns: Tensor [order+1, T]
    """
    t = torch.linspace(-1.0, 1.0, T, device=device, dtype=dtype)
    P = [torch.ones_like(t)]
    if order >= 1:
        P.append(t)
    for n in range(1, order):
        # (n+1) P_{n+1} = (2n+1) t P_n - n P_{n-1}
        P.append(((2 * n + 1) * t * P[n] - n * P[n - 1]) / (n + 1))
    B = torch.stack(P[: order + 1], dim=0)  # [ord+1, T]
    # L2 normalize rows for numerical stability
    B = B / (B.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-6).sqrt())
    return B


def _soft_quantiles(sorted_vals: torch.Tensor, levels: torch.Tensor, sigma: float = 0.5):
    """Differentiable quantiles via Gaussian weights on ranks.

    Args:
      sorted_vals: [..., T] ascending along last dim
      levels: [Q] in (0,1)
      sigma: rank-space std for Gaussian weighting

    Returns:
      quantiles: [..., Q]
    """
    *lead, T = sorted_vals.shape
    ranks = torch.arange(T, device=sorted_vals.device, dtype=sorted_vals.dtype)
    targets = levels * (T - 1)
    diff = ranks[None, :] - targets[:, None]  # [Q, T]
    w = torch.exp(-0.5 * (diff / max(sigma, 1e-6)) ** 2)
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    s_flat = sorted_vals.reshape(-1, T)  # [L, T]
    q = (w @ s_flat.T).T  # [L, Q]
    return q.reshape(*lead, -1)


class SQuaReFuse(nn.Module):
    """Sliced-Quantile & Quadratic-trend Robust Fusion.

    Input : x:[B,T,N,D], valid_mask:[B,T,N] (optional)
    Output: h:[B,T,N,D]
    """

    def __init__(
        self,
        d_model: int,
        num_dirs: int = 8,  # K: number of projection directions
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        poly_order: int = 2,  # Legendre order (0..2 typical)
        beta_init: float = 0.5,
        ortho_every_forward: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.K = int(num_dirs)
        # store as buffer for device/dtype alignment later
        self.register_buffer("_q_levels", torch.tensor(quantiles, dtype=torch.float32), persistent=False)
        self.P = int(poly_order)
        self.ortho_every_forward = bool(ortho_every_forward)

        # D -> K learned projection (orthonormalized per forward if enabled)
        self.proj = nn.Linear(self.d_model, self.K, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=5 ** 0.5)

        # Map concatenated stats back to D
        feat_in = self.K * (len(quantiles) + (self.P + 1))
        self.head = nn.Sequential(
            nn.LayerNorm(feat_in),
            nn.Linear(feat_in, 4 * self.d_model),
            nn.GELU(),
            nn.Linear(4 * self.d_model, self.d_model),
        )

        # Residual gate (scalar)
        self.beta = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))

    @torch.no_grad()
    def _orthonormalize(self):
        W = self.proj.weight.data  # [K, D]
        Q, _ = torch.linalg.qr(W.t(), mode="reduced")  # [D, K]
        self.proj.weight.data.copy_(Q.t())

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        """Fuse along the time dimension using sliced quantiles and Legendre trends.

        Args:
          x: [B, T, N, D]
          valid_mask: [B, T, N] or None (1=valid)

        Returns:
          h: [B, T, N, D]
        """
        B, T, N, D = x.shape
        device, dtype = x.device, x.dtype
        if valid_mask is None:
            valid_mask = torch.ones(B, T, N, device=device, dtype=dtype)

        if self.ortho_every_forward:
            self._orthonormalize()

        # 1) Project to K directions: v:[B,T,N,K]
        U = self.proj.weight.t()  # [D, K]
        v = torch.einsum('btnd,dk->btnk', x, U)
        v = v * valid_mask[..., None]

        # 2) Soft quantiles along time per (B,N,K)
        s, _ = torch.sort(v.permute(0, 2, 3, 1).contiguous(), dim=-1)  # [B,N,K,T]
        levels = self._q_levels.to(device=device, dtype=dtype)
        qv = _soft_quantiles(s, levels=levels, sigma=max(0.5, T / 16))  # [B,N,K*Q]
        qv = qv.view(B, N, self.K, -1)  # [B,N,K,Q]

        # 3) Legendre trend coefficients per projected dir
        Bmat = _legendre_basis(T, self.P, device, dtype)  # [P+1, T]
        v_bnkt = v.permute(0, 2, 3, 1).contiguous()  # [B,N,K,T]
        coeff = torch.einsum('bnkt,pt->bnkp', v_bnkt, Bmat)  # [B,N,K,P+1]

        # Concatenate stats across K and map back to D
        feats = torch.cat([qv, coeff], dim=-1).reshape(B, N, -1)  # [B,N,K*(Q+P+1)]
        y = self.head(feats)  # [B,N,D]

        # Broadcast over time and residual fuse
        y_btnd = y[:, None, :, :].expand(B, T, N, D)
        h = x + self.beta * y_btnd
        h = valid_mask[..., None] * h + (1.0 - valid_mask[..., None]) * x

        return h


__all__ = ["SQuaReFuse"]
