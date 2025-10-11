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


# -----------------------------
# BDRF (Bounded-DCT Residual Fusion)
# -----------------------------

def _dct2_basis(T: int, P: int, device, dtype):
    # DCT-II basis (approximate; normalized to avoid scale issues)
    t = torch.arange(T, device=device, dtype=dtype) + 0.5  # [T]
    B = [torch.ones(T, device=device, dtype=dtype)]  # p=0 constant term
    for p in range(1, P + 1):
        B.append(torch.cos(torch.pi * p * t / T))
    B = torch.stack(B, dim=0)  # [P+1, T]
    B = B / (B.square().sum(dim=1, keepdim=True).sqrt() + 1e-6)
    return B


class NonnegLinear(nn.Module):
    """Single-layer non-negative linear head for interpretability."""
    def __init__(self, fin: int, fout: int):
        super().__init__()
        self.W_raw = nn.Parameter(torch.empty(fin, fout))
        nn.init.xavier_uniform_(self.W_raw)

    def forward(self, x: torch.Tensor):
        W = F.softplus(self.W_raw)
        return x @ W, W


class BDRFuse(nn.Module):
    """
    BDRF: Bounded-DCT Residual Fusion
    Input : x:[B,T,N,D], valid_mask:[B,T,N] (optional)
    Output: h:[B,T,N,D], {'W': weight_matrix}
    """
    def __init__(
        self,
        d_model: int,
        num_dirs: int = 8,
        P: int = 2,
        beta_init: float = 0.5,
        ortho_every_forward: bool = True,
        bound_scale: float = 2.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.K = int(num_dirs)
        self.P = int(P)
        self.ortho_every_forward = bool(ortho_every_forward)
        self.bound_scale = float(bound_scale)

        self.proj = nn.Linear(d_model, self.K, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=5 ** 0.5)

        # features = K * [ DCT coeffs (P+1) + 2 bounded moments (mean, rms) ]
        feat_in = self.K * ((self.P + 1) + 2)
        self.head = NonnegLinear(fin=feat_in, fout=d_model)

        self.beta = nn.Parameter(torch.full((d_model,), float(beta_init)))

    def _orthonormalize(self):
        with torch.no_grad():
            W = self.proj.weight.data  # [K, D]
            Q, _ = torch.linalg.qr(W.t(), mode="reduced")
            self.proj.weight.data.copy_(Q.t())

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        B, T, N, D = x.shape
        device, dtype = x.device, x.dtype
        if valid_mask is None:
            valid_mask = torch.ones(B, T, N, device=device, dtype=dtype)

        if self.ortho_every_forward:
            self._orthonormalize()

        # 1) Projection to K directions
        U = self.proj.weight.t()                      # [D,K]
        v = torch.einsum('btnd,dk->btnk', x, U)       # [B,T,N,K]
        v = v * valid_mask[..., None]
        v_bnkt = v.permute(0, 2, 3, 1).contiguous()   # [B,N,K,T]

        # 2) Bounded via tanh after RMS normalization (robust to outliers)
        rms = torch.sqrt(v_bnkt.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        v_bounded = self.bound_scale * torch.tanh(v_bnkt / (rms + 1e-6))  # [B,N,K,T]

        # 3) Low-order DCT coefficients as trend features
        Bmat = _dct2_basis(T, self.P, device, dtype)                       # [P+1, T]
        coeff = torch.einsum('bnkt,pt->bnkp', v_bounded, Bmat)             # [B,N,K,P+1]

        # 4) Two bounded moments: mean and RMS
        mean_feat = v_bounded.mean(dim=-1, keepdim=True)                   # [B,N,K,1]
        rms_feat  = torch.sqrt(v_bounded.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

        feats = torch.cat([coeff, mean_feat, rms_feat], dim=-1).reshape(B, N, -1)  # [B,N,F]
        y, W = self.head(feats)                                            # [B,N,D], [F,D]

        # 5) Residual fusion with per-channel gate
        beta = torch.sigmoid(self.beta)[None, None, None, :]               # [1,1,1,D]
        y_btnd = y[:, None, :, :].expand(B, T, N, D)
        h = x + beta * y_btnd
        h = valid_mask[..., None] * h + (1.0 - valid_mask[..., None]) * x
        return h, {'W': W}


__all__ = [
    'BDRFuse',
]
