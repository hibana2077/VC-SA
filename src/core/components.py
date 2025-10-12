# -*- coding: utf-8 -*-

from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
# SoftSort no longer required for BDRF; keep module import list minimal


def _topk_straight_through(logits: torch.Tensor, k: int, dim: int = -1, tau: float = 1.0):
    """
    連續近似的 Top-k + Straight-Through：
      1) 計算 soft = softmax(logits / tau)
      2) 取 hard = one-hot(topk)
      3) 返回 hard + (soft - hard).detach() 以保留 soft gradient
    回傳：
      mask ∈ {≈0..1} 同 logits 形狀，沿 dim 方向有 k 個近似 1
      idx  ∈ long，硬選的 indices（不回傳梯度）
    """
    soft = F.softmax(logits / max(tau, 1e-6), dim=dim)
    topk = torch.topk(soft, k=min(k, soft.size(dim)), dim=dim)
    hard = torch.zeros_like(soft)
    hard.scatter_(dim, topk.indices, 1.0)
    mask = hard + (soft - hard).detach()
    return mask, topk.indices


class FrameTokenCoSelector(nn.Module):
    """
    (A) Learnable 影格／token 共選器
    - 影格層級 policy：對每格做全域池化 + MLP，估計效用，Top-k 選影格
    - 影格內 token 層級：對每個 token 做 gating（小型 MLP），Top-k 選 token
    - 以 ST 近似做可微分訓練；推論可用硬選
    參數：
      d_model: token 特徵維度 D
      frame_topk: 每段 clip 要保留的影格數 Kf
      token_topk: 每格要保留的 token 數 Kt
      use_cls: 若輸入含 CLS，是否在 frame 池化時使用 CLS（否則用 mean）
      tau_frame/token: 溫度參數
    forward 輸入：
      x: [B, T, N, D]
      mask: [B, T, N]，1=有效token，0=pad（可選）
    forward 輸出：
      z: [B, Kf, Kt, D] 已壓縮特徵
      frame_idx: [B, Kf]
      token_idx: [B, Kf, Kt]
      frame_mask: [B, T]（近似 one-hot Kf 個 1）
      token_mask: [B, T, N]（每個被選影格內約 Kt 個 1；未被選影格其 token_mask 接近 0）
    """
    def __init__(
        self,
        d_model: int,
        frame_topk: int,
        token_topk: int,
        use_cls: bool = False,
        tau_frame: float = 1.0,
        tau_token: float = 1.0,
        hidden: int = 4,
    ):
        super().__init__()
        self.frame_topk = frame_topk
        self.token_topk = token_topk
        self.tau_frame = tau_frame
        self.tau_token = tau_token
        self.use_cls = use_cls

        # 影格層級 policy（全域池化 -> MLP -> 標量）
        self.frame_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden * d_model),
            nn.GELU(),
            nn.Linear(hidden * d_model, 1),
        )

        # token 層級 policy（逐 token gating）
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, N, D = x.shape
        if mask is None:
            mask = torch.ones(B, T, N, device=x.device, dtype=x.dtype)

        # ---- Frame-level scoring ----
        # frame_repr: [B, T, D]
        if self.use_cls and N >= 1:
            frame_repr = x[:, :, 0, :]  # 使用 CLS
        else:
            # 對有效 token 做加權平均（忽略 padding）
            denom = mask.sum(dim=2, keepdim=False).clamp_min(1e-6)[..., None]  # [B, T, 1]
            frame_repr = (x * mask[..., None]).sum(dim=2) / denom  # [B, T, D]

        frame_logit = self.frame_mlp(frame_repr).squeeze(-1)  # [B, T]
        frame_mask, frame_idx = _topk_straight_through(frame_logit, self.frame_topk, dim=1, tau=self.tau_frame)  # [B,T]

        # ---- Token-level scoring (per frame) ----
        token_logit = self.token_mlp(x).squeeze(-1)  # [B, T, N]
        # 無效 token 的分數設為非常小，避免被選
        token_logit = token_logit + torch.log(mask.clamp_min(1e-9))

        token_mask_all, token_idx_all = _topk_straight_through(
            token_logit, self.token_topk, dim=2, tau=self.tau_token
        )  # [B, T, N], [B, T, Kt]

        # 僅保留被選影格的 token，未選影格的 token_mask ≈ 0
        token_mask = token_mask_all * frame_mask.unsqueeze(-1)  # [B, T, N]
        token_mask = token_mask_all * frame_mask.unsqueeze(-1)  # [B, T, N]


# -----------------------------------------
# RPFuse: Ramanujan Periodic Fusion (drop-in)
# -----------------------------------------
from typing import Optional, Dict, Tuple, List
import math


def _coprimes(q: int) -> List[int]:
    return [a for a in range(1, q + 1) if math.gcd(a, q) == 1]


def _ramanujan_sum_row(q: int, T: int, device, dtype) -> torch.Tensor:
    # Real Ramanujan sum: sum_{(a,q)=1} cos(2π a n / q), n=0..T-1
    n = torch.arange(T, device=device, dtype=dtype)
    a_list = _coprimes(q)
    if len(a_list) == 0:
        return torch.zeros(T, device=device, dtype=dtype)
    angles = 2 * math.pi * torch.outer(n, torch.tensor(a_list, device=device, dtype=dtype)) / q  # [T, phi(q)]
    row = torch.cos(angles).sum(dim=1)  # [T]
    # ℓ2 normalize to unit energy for numerical stability
    row = row / (row.square().sum().sqrt() + 1e-6)
    return row


def _build_ramanujan_basis(T: int, q_max: int, device, dtype) -> Tuple[torch.Tensor, List[int]]:
    qs = [q for q in range(1, min(q_max, T) + 1)]
    rows = [_ramanujan_sum_row(q, T, device, dtype) for q in qs]
    B = torch.stack(rows, dim=0)  # [L, T], L = len(qs)
    # Orthonormalize rows via QR on B^T to improve conditioning
    Q, _ = torch.linalg.qr(B.t(), mode="reduced")  # [T, L]
    R = Q.t()  # [L, T], row-orthonormal
    return R, qs


class _NonnegLinear(nn.Module):
    def __init__(self, fin: int, fout: int):
        super().__init__()
        self.W_raw = nn.Parameter(torch.empty(fin, fout))
        nn.init.xavier_uniform_(self.W_raw)

    def forward(self, x: torch.Tensor):
        W = F.softplus(self.W_raw)  # non-negative for interpretability
        return x @ W, W


class RPFuse(nn.Module):
    """
    Ramanujan Periodic Fusion (RPFuse)
    Input : x:[B,T,N,D], valid_mask:[B,T,N] (optional, 1=valid)
    Output: h:[B,T,N,D], info: {'qs': Tensor[List[int]], 'energy_q': Tensor, 'W_dir2ch': Tensor, 'cales': Tensor}
    禁用: Attention / SSM / Tensor decomposition / Graph / 低秩核 / ToMe/VTM
    """

    def __init__(
        self,
        d_model: int,
        num_dirs: int = 8,     # Few directions to compress D -> K
        q_max: int = 16,       # Max integer period considered (≤ T)
        ortho_dirs: bool = True,
        beta_init: float = 0.5,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.K = int(num_dirs)
        self.q_max = int(q_max)
        self.ortho_dirs = bool(ortho_dirs)

        # Directional projection D->K
        self.proj = nn.Linear(self.d_model, self.K, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=5 ** 0.5)

        # Non-negative mapping K->D for interpretability
        self.head = _NonnegLinear(self.K, self.d_model)

        # Residual gate
        self.beta = nn.Parameter(torch.full((self.d_model,), float(beta_init)))

        # Basis cache
        self._cache_T = None
        self._cache = None

    @torch.no_grad()
    def _orthonormalize_dirs(self):
        W = self.proj.weight.data  # [K, D]
        Q, _ = torch.linalg.qr(W.t(), mode="reduced")
        self.proj.weight.data.copy_(Q.t())

    def _get_basis(self, T: int, device, dtype):
        if self._cache_T == (T, device, dtype):
            return self._cache
        R, qs = _build_ramanujan_basis(T, self.q_max, device, dtype)  # [L, T], List[int]
        self._cache_T = (T, device, dtype)
        self._cache = (R, qs)
        return self._cache

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        B, T, N, D = x.shape
        device, dtype = x.device, x.dtype
        if valid_mask is None:
            valid_mask = torch.ones(B, T, N, device=device, dtype=dtype)

        if self.ortho_dirs:
            self._orthonormalize_dirs()

        # 1) Channel-direction compression: v:[B,T,N,K]
        v = torch.einsum('btnd,dk->btnk', x, self.proj.weight.t())  # [B,T,N,K]
        v = v * valid_mask[..., None]

        # 2) Ramanujan basis (row-orthonormal): R:[L,T], periods:qs
        R, qs = self._get_basis(T, device, dtype)  # [L, T]
        # Coefficients along time alpha:[B,N,K,L]
        v_bnkt = v.permute(0, 2, 3, 1).contiguous()  # [B,N,K,T]
        alpha = torch.einsum('bnkt,lt->bnkl', v_bnkt, R)  # [B,N,K,L]

        # 3) Energy spectrum per q
        energy_q = alpha.square().mean(dim=(0, 1, 2))  # [L]

        # 4) Reconstruct K-dim time signal and map back to D: y:[B,T,N,D]
        v_hat_bnkt = torch.einsum('bnkl,lt->bnkt', alpha, R)  # [B,N,K,T]
        v_hat = v_hat_bnkt.permute(0, 3, 1, 2).contiguous()   # [B,T,N,K]
        # Non-negative mapping K->D (constant over time/space)
        W_dir2ch_param = F.softplus(self.head.W_raw)
        y_btnd = torch.einsum('btnk,kd->btnd', v_hat, W_dir2ch_param)

        # 5) Residual fusion + gating
        beta = torch.sigmoid(self.beta)[None, None, None, :]
        h = x + beta * y_btnd
        h = valid_mask[..., None] * h + (1.0 - valid_mask[..., None]) * x

        # Prepare logging info; "cales" provided as the tensor of periods for compatibility with requested logs
        qs_tensor = torch.as_tensor(qs, device=device, dtype=dtype)
        info: Dict[str, torch.Tensor | List[int]] = {
            'qs': qs,                              # list of integer periods
            'energy_q': energy_q,                  # [L]
            'W_dir2ch': W_dir2ch_param.detach(),   # [K,D]
            'cales': qs_tensor,                    # alias for requested logging field
        }
        return h, info

__all__ = [
    'FrameTokenCoSelector',
    'RPFuse',
]