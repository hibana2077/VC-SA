# -*- coding: utf-8 -*-
# FrierenFuse: Fréchet-inspired Robust, Interpretable ENvelope & derivative Fuse
# Drop-in fusion layer for [B,T,N,D], no Attention/SSM/Graph/Low-rank/ToMe/VTM
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonnegLinear(nn.Module):
    """y = x @ W, with W >= 0 for parts-based interpretability."""

    def __init__(self, fin: int, fout: int):
        super().__init__()
        self.W_raw = nn.Parameter(torch.empty(fin, fout))
        nn.init.xavier_uniform_(self.W_raw)

    def forward(self, x: torch.Tensor):
        W = F.softplus(self.W_raw)  # non-negative
        return x @ W, W


def _orthonormalize_cols_(W: torch.Tensor):
    # Column-orthonormalize in-place via QR
    Q, _ = torch.linalg.qr(W.t(), mode="reduced")
    W.copy_(Q.t())


def _legendre_basis(T: int, order: int, device, dtype):
    t = torch.linspace(-1.0, 1.0, T, device=device, dtype=dtype)
    P = [torch.ones_like(t), t]
    for n in range(1, order):
        P.append(((2 * n + 1) * t * P[n] - n * P[n - 1]) / (n + 1))
    B = torch.stack(P[: order + 1], dim=0)  # [order+1, T]
    B = B / (B.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-6).sqrt())
    return B  # [P+1, T]


class FrierenFuse(nn.Module):
    """
    Input : x:[B,T,N,D], valid_mask:[B,T,N] (optional, 1=valid)
    Output: h:[B,T,N,D], aux:dict
    """

    def __init__(
        self,
        d_model: int,
        num_dirs: int = 8,
        scales: Tuple[int, ...] = (1, 2, 4),
        include_second: bool = True,
        include_posneg: bool = True,
        poly_order: int = 2,  # Legendre 0..P
        bound_scale: float = 2.5,  # tanh bounding for robustness
        beta_init: float = 0.5,
        ortho_every_forward: bool = True,
    ):
        super().__init__()
        self.D = int(d_model)
        self.K = int(num_dirs)
        self.scales = tuple(int(s) for s in scales if int(s) > 0)
        assert self.K > 0 and len(self.scales) > 0
        self.include_second = bool(include_second)
        self.include_posneg = bool(include_posneg)
        self.P = int(poly_order)
        self.bound_scale = float(bound_scale)
        self.ortho_every_forward = bool(ortho_every_forward)

        # Learn K directions; orthonormalized per-forward for stability
        self.proj = nn.Linear(self.D, self.K, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=5 ** 0.5)

        # Feature layout per direction:
        #   for each r: [meanΔ, mean|Δ|, (optional) mean|Δ2|, (optional) posVar, negVar]
        per_r = 2 + (1 if self.include_second else 0) + (2 if self.include_posneg else 0)
        # plus trend terms (Legendre 0..P) and range (max-min)
        base_extra = (self.P + 1) + 1
        fin = self.K * (len(self.scales) * per_r + base_extra)

        self.head = NonnegLinear(fin, self.D)  # interpretable fusion
        self.beta = nn.Parameter(torch.full((self.D,), float(beta_init)))  # residual gate

    @torch.no_grad()
    def _orthonormalize(self):
        _orthonormalize_cols_(self.proj.weight.data)

    def _first_second_diffs(self, v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        v:[B,T,N,K], valid:[B,T,N] -> concatenated features [B,N,K * per_r*|scales|]
        """
        B, T, N, K = v.shape
        feats = []
        for r in self.scales:
            if T <= r:
                Z = torch.zeros(B, N, K, device=v.device, dtype=v.dtype)
                feats.extend([Z, Z])  # meanΔ, mean|Δ|
                if self.include_second:
                    feats.append(Z)  # mean|Δ2|
                if self.include_posneg:
                    feats.extend([Z, Z])  # posVar, negVar
                continue
            v1, v0 = v[:, r:, :, :], v[:, :-r, :, :]
            m1, m0 = valid[:, r:, :, None], valid[:, :-r, :, None]
            m = m1 * m0  # [B,T-r,N,1]
            d1 = (v1 - v0) * m
            denom1 = m.sum(dim=1).clamp_min(1e-6)  # [B,N,1]
            mean_delta = d1.sum(dim=1) / denom1  # signed velocity
            mean_speed = d1.abs().sum(dim=1) / denom1  # TV rate

            feats.extend([mean_delta, mean_speed])

            if self.include_second:
                if T <= 2 * r:
                    feats.append(torch.zeros(B, N, K, device=v.device, dtype=v.dtype))
                else:
                    v2 = v[:, 2 * r :, :, :]
                    m2 = valid[:, 2 * r :, :, None]
                    msec = m0[:, r:, :, :] * m1[:, r:, :, :] * m2  # all three valid
                    d2 = (v2 - 2 * v1[:, r:, :, :] + v0[:, :-r, :, :]) * msec
                    denom2 = msec.sum(dim=1).clamp_min(1e-6)
                    mean_acc_mag = d2.abs().sum(dim=1) / denom2  # acceleration magnitude
                    feats.append(mean_acc_mag)

            if self.include_posneg:
                posVar = F.relu(d1).sum(dim=1) / denom1  # upward variation
                negVar = F.relu(-d1).sum(dim=1) / denom1  # downward variation
                feats.extend([posVar, negVar])

        return torch.cat(feats, dim=-1)  # [B,N,K * per_r*|scales|]

    def _trend_and_range(self, v: torch.Tensor, valid: torch.Tensor, P: int):
        # v:[B,T,N,K] -> trend coeffs [B,N,K,(P+1)] and range [B,N,K,1]
        B, T, N, K = v.shape
        device, dtype = v.device, v.dtype
        Bmat = _legendre_basis(T, P, device, dtype)  # [P+1, T]
        mask = valid[..., None]  # [B,T,N,1]
        denom = mask.sum(dim=1).clamp_min(1e-6)  # [B,N,1,1]
        v_bnkt = v.permute(0, 2, 3, 1)  # [B,N,K,T]
        trend = torch.einsum("bnkt,pt->bnkp", v_bnkt, Bmat) / denom  # normalized coeffs
        v_masked = v * mask
        vmin = v_masked.min(dim=1, keepdim=False).values  # [B,N,K]
        vmax = v_masked.max(dim=1, keepdim=False).values  # [B,N,K]
        vrange = (vmax - vmin)[..., None]  # [B,N,K,1]
        return trend, vrange

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        assert x.dim() == 4, "x must be [B,T,N,D]"
        B, T, N, D = x.shape
        device, dtype = x.device, x.dtype
        if valid_mask is None:
            valid_mask = torch.ones(B, T, N, device=device, dtype=dtype)

        if self.ortho_every_forward:
            self._orthonormalize()

        # Project to K directions
        U = self.proj.weight.t()  # [D,K]
        v = torch.einsum("btnd,dk->btnk", x, U)  # [B,T,N,K]
        v = v * valid_mask[..., None]  # mask padding

        # Robust bounding (per token+dir RMS) to avoid outliers
        rms = torch.sqrt(v.pow(2).mean(dim=1, keepdim=True) + 1e-6)
        v = self.bound_scale * torch.tanh(v / (rms + 1e-6))

        # Multiscale first/second differences & pos/neg variations
        feats_dyn = self._first_second_diffs(v, valid_mask)  # [B,N,F1]

        # Trend (Legendre 0..P) + range
        trend, vrange = self._trend_and_range(v, valid_mask, self.P)  # [B,N,K,P+1], [B,N,K,1]
        feats_stat = torch.cat([trend.reshape(B, N, -1), vrange.reshape(B, N, -1)], dim=-1)

        feats = torch.cat([feats_dyn, feats_stat], dim=-1)  # [B,N,F_total]
        y, W = self.head(feats)  # [B,N,D], [F_total,D]

        # Residual broadcast back to [B,T,N,D]
        beta = torch.sigmoid(self.beta)[None, None, None, :]  # [1,1,1,D]
        y_btnd = y[:, None, :, :].expand(B, T, N, D)
        h = valid_mask[..., None] * (x + beta * y_btnd) + (1.0 - valid_mask[..., None]) * x

        aux: Dict[str, torch.Tensor] = {
            "U": U,
            "W": W,
            "scales": torch.tensor(self.scales, device=U.device, dtype=U.dtype),
            "feature_dim": torch.tensor([feats.shape[-1]], device=U.device, dtype=torch.long),
        }
        return h, aux


__all__ = ["FrierenFuse"]
