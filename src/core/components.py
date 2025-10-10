# -*- coding: utf-8 -*-
# ViT-GraphSampler: (A) Learnable 影格／token 共選器 & (B) Graph-Based MemBank（輕量時序 GNN）
# 只含兩個可直接 import 的 PyTorch 模組，無外部相依（除 torch）。
# 形狀慣例：
#   - 視覺主幹（如 ViT）輸出：x ∈ [B, T, N, D]，B 批次、T 影格數、N token 數（含或不含 CLS 皆可，建議不含）、D 通道維度
#   - 共選器輸出：選到的影格索引 / mask、每格選到的 token 索引 / mask、以及壓縮後特徵
#   - 記憶庫輸入：已共選完之特徵 z ∈ [B, T, M, D]（每格 M 個 token；若每格 M 不同，可用 padding + mask）
#
# 注意：此為最小可行原型（MVP）。訓練時建議：
#   - frame/token 的 Top-k 以 straight-through (ST) 估計子梯度
#   - tau（溫度）可從較高值線性退火
#   - 共選損失 = 分類損 + 稀疏正則（約束平均選取比例）+ 增強一致性（同一 clip 兩種增強下的選取相似）
#
# Author: ChatGPT (PyTorch)
# License: MIT

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# -----------------------------
# SQuaRe-Fuse (Sliced-Quantile & Quadratic-trend Robust Fusion)
# -----------------------------

def _legendre_basis(T: int, order: int, device, dtype):
    t = torch.linspace(-1.0, 1.0, T, device=device, dtype=dtype)
    P = [torch.ones_like(t)]
    if order >= 1:
        P.append(t)
    for n in range(1, order):
        P.append(((2 * n + 1) * t * P[n] - n * P[n - 1]) / (n + 1))
    B = torch.stack(P[: order + 1], dim=0)
    B = B / (B.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-6).sqrt())
    return B  # [order+1, T]


def _soft_quantiles(sorted_vals: torch.Tensor, levels: torch.Tensor, sigma: float = 0.5):
    """
    Differentiable quantiles using Gaussian weights over ranks.
    sorted_vals: [..., T] (ascending)
    levels: [Q] in (0,1)
    returns: [..., Q]
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
    """
    Sliced-Quantile & Quadratic-trend Robust Fusion
    Input : x:[B,T,N,D], valid_mask:[B,T,N] (optional)
    Output: h:[B,T,N,D], memory_dict (API parity with RamaFuse)
    K: num_dirs: int, 投影方向數
    Q: quantiles: Tuple[float,...], 量化切片
    P: poly_order: int, 多項式階數
    """

    def __init__(
        self,
        d_model: int,
        num_dirs: int = 8,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        poly_order: int = 2,
        beta_init: float = 0.5,
        ortho_every_forward: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.K = int(num_dirs)
        self.register_buffer("Q_levels", torch.tensor(quantiles, dtype=torch.float32), persistent=False)
        self.P = int(poly_order)
        self.ortho_every_forward = ortho_every_forward

        # Learn projection matrix (D -> K); we orthonormalize rows each forward
        self.proj = nn.Linear(d_model, self.K, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=5 ** 0.5)

        feat_in = self.K * (len(quantiles) + (self.P + 1))
        self.head = nn.Sequential(
            # nn.LayerNorm(feat_in),
            nn.RMSNorm(feat_in),
            nn.Linear(feat_in, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        # Per-channel residual scale (vector gate). Shape: [D]
        # Initialized to beta_init for all channels.
        self.beta = nn.Parameter(torch.full((d_model,), float(beta_init), dtype=torch.float32))
        self._mem_state: Dict[str, torch.Tensor] = {}

    def _orthonormalize(self):
        with torch.no_grad():
            W = self.proj.weight.data  # [K, D]
            Q, _ = torch.linalg.qr(W.t(), mode="reduced")  # [D, K]
            self.proj.weight.data.copy_(Q.t())

    def forward(
        self,
        x: torch.Tensor,  # [B,T,N,D]
        valid_mask: Optional[torch.Tensor] = None
    ):
        B, T, N, D = x.shape
        device, dtype = x.device, x.dtype
        if valid_mask is None:
            valid_mask = torch.ones(B, T, N, device=device, dtype=dtype)

        if self.ortho_every_forward:
            self._orthonormalize()

        # 1) Project to K directions: v = x @ U (U:[D,K]) => [B,T,N,K]
        U = self.proj.weight.t()  # [D,K]
        v = torch.einsum("btnd,dk->btnk", x, U)  # [B,T,N,K]
        v = v * valid_mask[..., None]

        # 2) Differentiable quantiles along time
        s, _ = torch.sort(v.permute(0, 2, 3, 1).contiguous(), dim=-1)  # [B,N,K,T]
        levels = self.Q_levels.to(device=device, dtype=dtype)
        qv = _soft_quantiles(s, levels=levels, sigma=max(0.5, T / 16))  # [B,N,K*Q]
        qv = qv.view(B, N, self.K, -1)  # [B,N,K,Q]

        # 3) Legendre trend coefficients along time
        Bmat = _legendre_basis(T, self.P, device, dtype)  # [P+1, T]
        v_bnkt = v.permute(0, 2, 3, 1).contiguous()  # [B,N,K,T]
        coeff = torch.einsum("bnkt,pt->bnkp", v_bnkt, Bmat)  # [B,N,K,P+1]

        feats = torch.cat([qv, coeff], dim=-1).reshape(B, N, -1)  # [B,N, K*(Q+P+1)]
        y = self.head(feats)  # [B,N,D]

        # Broadcast over time and residual fuse
        y_btnd = y[:, None, :, :].expand(B, T, N, D)
        # Vector gate beta: broadcast to [B,T,N,D]
        h = x + self.beta[None, None, None, :] * y_btnd
        h = valid_mask[..., None] * h + (1.0 - valid_mask[..., None]) * x

        # Compute residual impact ratio r for logging (detach to avoid autograd overhead)
        with torch.no_grad():
            num = ((self.beta[None, None, None, :] * y_btnd).pow(2).sum(dim=-1).sqrt()).mean()
            den = (x.pow(2).sum(dim=-1).sqrt()).mean().clamp_min(1e-9)
            r = (num / den).detach()
        return h, {'r': r}


__all__ = [
    'SQuaReFuse',
]
