# -*- coding: utf-8 -*-
# FRIDA: Fréchet‑Inspired Difference Aggregator (drop‑in fusion layer for [B,T,N,D])
# Author: ChatGPT (PyTorch) — MIT License
#
# I/O contract (same as BDRFuse):
#   forward(x:[B,T,N,D], valid_mask:[B,T,N] (optional)) -> h:[B,T,N,D], aux:dict
#
# Core idea:
#   1) Linearly project token features onto K orthogonal directions: v ∈ [B,T,N,K]
#   2) For multiple temporal scales r ∈ scales, compute discrete Fréchet / finite-difference features:
#        meanΔ_r = mean signed difference, magΔ_r = L1 or RMS magnitude
#      This corresponds to JVP-style finite-difference approximations of Gateaux/Fréchet derivatives
#      defined by zero-sum templates like {1,-1}, {1,0,-1}, ...
#   3) Use an interpretable non-negative linear head (NonnegLinear) to weight features and fuse back
#      to the original features via a residual gate.
#
# Dependencies: torch
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class NonnegLinear(nn.Module):
    """Single-layer non-negative linear head: y = x @ W (W >= 0).
    Returns (y, W_pos).
    """
    def __init__(self, fin: int, fout: int):
        super().__init__()
        self.W_raw = nn.Parameter(torch.empty(fin, fout))
        nn.init.xavier_uniform_(self.W_raw)

    def forward(self, x: torch.Tensor):
        W = F.softplus(self.W_raw)  # enforce non-negativity while remaining differentiable
        return x @ W, W

def _orthonormalize_cols(W: torch.Tensor) -> torch.Tensor:
    """Obtain column-orthonormal basis via QR (works for Linear.weight shapes [K,D] or [D,K])."""
    Q, _ = torch.linalg.qr(W.t(), mode="reduced")
    return Q.t()

class FRIDA(nn.Module):
    """FRIDA: Fréchet‑Inspired Difference Aggregator
    Parameters:
      d_model: channel dimension D
      num_dirs: number of temporal projection directions K
      scales: list/tuple of difference spans (e.g., (1,2,4))
      use_rms: if True use RMS magnitude, else use L1 magnitude
      ortho_every_forward: whether to orthonormalize projection every forward
      bound_scale: tanh bounding factor to stabilize gradients and outliers
      beta_init: initial value for the residual gate (applied via sigmoid)
    Inputs:
      x: [B,T,N,D]
      valid_mask: [B,T,N] (1=valid; 0=padding), optional
    Outputs:
      h: [B,T,N,D]
      aux: dict(U, W, scales, feats_sample)
    """
    def __init__(
        self,
        d_model: int,
        num_dirs: int = 8,
        scales: Tuple[int, ...] = (1, 2, 4),
        use_rms: bool = True,
        ortho_every_forward: bool = True,
        bound_scale: float = 2.5,
        beta_init: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.K = int(num_dirs)
        assert self.K > 0
        self.scales: Tuple[int, ...] = tuple(int(s) for s in scales if int(s) > 0)
        assert len(self.scales) > 0, "scales must be positive integers"
        self.use_rms = bool(use_rms)
        self.ortho_every_forward = bool(ortho_every_forward)
        self.bound_scale = float(bound_scale)

        # Linear projection to K directions (columns are directions), compatible with BDRF
        self.proj = nn.Linear(d_model, self.K, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=5 ** 0.5)

        # feature dim = K * (2 * len(scales)) ; concatenation of [meanΔ_r, magΔ_r]
        feat_in = self.K * (2 * len(self.scales))
        self.head = NonnegLinear(fin=feat_in, fout=d_model)

        # residual gating (one parameter per channel)
        self.beta = nn.Parameter(torch.full((d_model,), float(beta_init)))

    @torch.no_grad()
    def _orthonormalize(self):
        self.proj.weight.data.copy_(_orthonormalize_cols(self.proj.weight.data))

    def _diff_feats(self, v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Compute multiscale difference features from v:[B,T,N,K], returning [B,N, K*2*|scales|]."""
        B, T, N, K = v.shape
        feats = []
        for r in self.scales:
            if T <= r:
                # If sequence is too short, append zero features
                feats.append(torch.zeros(B, N, K, device=v.device, dtype=v.dtype))
                feats.append(torch.zeros(B, N, K, device=v.device, dtype=v.dtype))
                continue
            v1 = v[:, r:, :, :]    # [B,T-r,N,K]
            v0 = v[:, :-r, :, :]   # [B,T-r,N,K]
            m1 = valid[:, r:, :, None]  # [B,T-r,N,1]
            m0 = valid[:, :-r, :, None]
            m  = m1 * m0
            delta = (v1 - v0) * m

            denom = m.sum(dim=1).clamp_min(1e-6)  # [B,N,1]
            mean_delta = delta.sum(dim=1) / denom  # mean signed change [B,N,K]

            if self.use_rms:
                mag = torch.sqrt((delta.pow(2).sum(dim=1) / denom).clamp_min(1e-6))  # RMS magnitude
            else:
                mag = (delta.abs().sum(dim=1) / denom)  # L1 magnitude

            feats.append(mean_delta)  # [B,N,K]
            feats.append(mag)         # [B,N,K]

        return torch.cat(feats, dim=-1)  # [B,N,K*2*|scales|]

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        assert x.dim() == 4, "x must be [B,T,N,D]"
        B, T, N, D = x.shape
        device, dtype = x.device, x.dtype

        if valid_mask is None:
            valid_mask = torch.ones(B, T, N, device=device, dtype=dtype)

        if self.ortho_every_forward:
            self._orthonormalize()

        # 1) Project to K directions -> v:[B,T,N,K]
        U = self.proj.weight.t()                      # [D,K]
        v = torch.einsum('btnd,dk->btnk', x, U)       # [B,T,N,K]
        v = v * valid_mask[..., None]

        # 2) Bounding (RMS normalization per token+dir) to avoid outliers
        rms = torch.sqrt(v.pow(2).mean(dim=1, keepdim=True) + 1e-6)  # [B,1,N,K]
        v_bounded = self.bound_scale * torch.tanh(v / (rms + 1e-6))

        # 3) Multiscale difference features (finite-difference approximation of Fréchet/Gâteaux derivatives)
        feats = self._diff_feats(v_bounded, valid_mask)  # [B,N,F]

        # 4) Interpretable non-negative linear head
        y, W = self.head(feats)  # [B,N,D], [F,D]

        # 5) Residual fusion (per-channel gating)
        beta = torch.sigmoid(self.beta)[None, None, None, :]  # [1,1,1,D]
        y_btnd = y[:, None, :, :].expand(B, T, N, D)
        h = x + beta * y_btnd

        # Preserve padding regions
        h = valid_mask[..., None] * h + (1.0 - valid_mask[..., None]) * x

        aux = {'W': W, 'U': U, 'scales': self.scales, 'feats_sample': feats[:1].detach()}
        return h, aux

__all__ = ['FRIDA']
