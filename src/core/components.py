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
"""StatLite Components
=======================
精簡版的兩個統計式/貪婪式模組，取代原本的可學共選器 + 圖記憶庫：

1. SimpleFrameTokenSelector
   - 影格：facility-location（覆蓋） + 可學標量分數（貪婪）
   - token：k-Center Greedy（最大最小距離覆蓋）
2. StatMem
   - 時序統計記憶：Approximate Rank Pooling (ARP) + EMA 平滑

兩者均只依賴 torch，可直接嵌入現有管線；與舊版 I/O 介面保持一致：
  Selector forward 輸入：x:[B,T,N,D] (mask 可選) → 輸出 (z, frame_idx, token_idx, frame_mask, token_mask)
  Memory forward 輸入：z:[B,T,M,D] (valid_mask 可選) → 輸出 (h, memory_dict)
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Helpers
# -----------------------------

def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.transpose(-1, -2)


def _facility_location_greedy(frame_repr: torch.Tensor, k: int, score: Optional[torch.Tensor] = None,
                              lambda_cov: float = 1.0, lambda_score: float = 0.0,
                              valid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, D = frame_repr.shape
    device = frame_repr.device
    sim_bt = []
    for b in range(B):
        fb = frame_repr[b]
        sim_bt.append(_safe_cosine(fb, fb))
    sim = torch.stack(sim_bt, dim=0)  # [B,T,T]

    if score is None:
        score = torch.zeros(B, T, device=device)
    if valid is None:
        valid = torch.ones(B, T, device=device)

    frame_idx = torch.zeros(B, k, dtype=torch.long, device=device)
    frame_mask = torch.zeros(B, T, dtype=frame_repr.dtype, device=device)
    best_cover = torch.zeros(B, T, device=device)
    chosen = torch.zeros(B, T, dtype=torch.bool, device=device)

    for step in range(k):
        best = best_cover.unsqueeze(-1)  # [B,T,1]
        new_best = torch.maximum(best, sim)
        cov_gain_all = (new_best - best).sum(dim=1)  # [B,T]
        total_gain = lambda_cov * cov_gain_all + lambda_score * score
        total_gain = total_gain.masked_fill(~valid.bool(), float('-inf'))
        total_gain = total_gain.masked_fill(chosen, float('-inf'))
        j = torch.argmax(total_gain, dim=1)
        frame_idx[:, step] = j
        batch_arange = torch.arange(B, device=device)
        frame_mask[batch_arange, j] = 1.0
        bj = sim[batch_arange, :, j]
        best_cover = torch.maximum(best_cover, bj)
        chosen[batch_arange, j] = True

    return frame_idx, frame_mask


def _kcenter_greedy(X: torch.Tensor, k: int, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    N, D = X.shape
    device = X.device
    if valid is None:
        valid = torch.ones(N, device=device, dtype=torch.bool)
    else:
        valid = valid.bool()
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return X.new_zeros(k, dtype=torch.long)
    Xv = X[valid_idx]
    dists = (Xv ** 2).sum(dim=1)
    first = torch.argmax(dists)
    selected = [int(valid_idx[first].item())]
    min_dist = torch.cdist(X, X[selected].detach(), p=2).squeeze(-1)
    min_dist[~valid] = -1.0
    for _ in range(1, k):
        cand = torch.argmax(min_dist)
        selected.append(int(cand.item()))
        new_d = torch.cdist(X, X[cand].unsqueeze(0).detach(), p=2).squeeze(-1)
        min_dist = torch.minimum(min_dist, new_d)
        min_dist[~valid] = -1.0
    return torch.tensor(selected, dtype=torch.long, device=device)


class SimpleFrameTokenSelector(nn.Module):
    def __init__(self,
                 d_model: int,
                 frame_topk: int,
                 token_topk: int,
                 lambda_cov: float = 1.0,
                 lambda_score: float = 0.0,
                 use_cls: bool = False,
                 hidden: int = 4):
        super().__init__()
        self.frame_topk = frame_topk
        self.token_topk = token_topk
        self.lambda_cov = lambda_cov
        self.lambda_score = lambda_score
        self.use_cls = use_cls
        self.frame_score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden * d_model),
            nn.GELU(),
            nn.Linear(hidden * d_model, 1),
        )

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, N, D = x.shape
        device = x.device
        if mask is None:
            mask = torch.ones(B, T, N, device=device, dtype=x.dtype)
        if self.use_cls and N >= 1:
            frame_repr = x[:, :, 0, :]
            valid_frame = torch.ones(B, T, device=device, dtype=torch.bool)
        else:
            denom = mask.sum(dim=2, keepdim=False).clamp_min(1e-6)[..., None]
            frame_repr = (x * mask[..., None]).sum(dim=2) / denom
            valid_frame = (denom.squeeze(-1) > 0)
        frame_score = self.frame_score_head(frame_repr).squeeze(-1)
        frame_idx, frame_mask = _facility_location_greedy(
            frame_repr, k=self.frame_topk, score=frame_score,
            lambda_cov=self.lambda_cov, lambda_score=self.lambda_score,
            valid=valid_frame.to(frame_repr.dtype)
        )
        token_idx = torch.zeros(B, self.frame_topk, self.token_topk, dtype=torch.long, device=device)
        token_mask = torch.zeros(B, T, N, dtype=x.dtype, device=device)
        batch_arange = torch.arange(B, device=device)
        for b in range(B):
            for kf in range(self.frame_topk):
                t_sel = int(frame_idx[b, kf].item())
                X = x[b, t_sel]
                m = mask[b, t_sel] > 0.5
                idxs = _kcenter_greedy(X, self.token_topk, valid=m)
                token_idx[b, kf] = idxs
                token_mask[b, t_sel, idxs] = 1.0
        x_sel_frames = x[batch_arange[:, None], frame_idx]
        z = x_sel_frames[batch_arange[:, None, None], torch.arange(self.frame_topk, device=device)[None, :, None], token_idx]
        return z, frame_idx, token_idx, frame_mask, token_mask


class StatMem(nn.Module):
    def __init__(self, d_model: int, use_arp: bool = True, window: int = 8, alpha: float = 0.5):
        super().__init__()
        self.use_arp = use_arp
        self.window = max(1, int(window))
        self.alpha = float(alpha)
        self._mem_state: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _arp_window(z_seq: torch.Tensor, t: int, W: int, valid_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = z_seq.device
        start = max(0, t - W + 1)
        L = t - start + 1
        w = torch.arange(1, L + 1, device=device, dtype=z_seq.dtype)
        w = 2 * w - (L + 1)
        w = w / (w.abs().sum().clamp_min(1e-6))
        window = z_seq[start:t + 1]
        if valid_seq is not None:
            v = valid_seq[start:t + 1].unsqueeze(-1).to(z_seq.dtype)
            window = window * v
            denom = (v * w.view(L, 1, 1)).abs().sum(dim=0).clamp_min(1e-6)
            return (window * w.view(L, 1, 1)).sum(dim=0) / denom
        else:
            return (window * w.view(L, 1, 1)).sum(dim=0)

    def forward(self,
                z: torch.Tensor,
                pos: Optional[torch.Tensor] = None,
                valid_mask: Optional[torch.Tensor] = None,
                memory_id: Optional[str] = None,
                reset_memory: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, M, D = z.shape
        device = z.device
        if valid_mask is None:
            valid_mask = torch.ones(B, T, M, device=device, dtype=z.dtype)
        key = memory_id or "default"
        if reset_memory or (key not in self._mem_state):
            self._mem_state[key] = torch.zeros(B, M, D, device=device, dtype=z.dtype)
        mem = self._mem_state[key]
        h_list = []
        for t in range(T):
            z_t = z[:, t]
            v_t = valid_mask[:, t].unsqueeze(-1)
            if self.use_arp:
                arp_t = torch.zeros_like(z_t)
                for b in range(B):
                    arp_t[b] = self._arp_window(z[b], t, self.window, valid_seq=valid_mask[b])
                base = arp_t
            else:
                base = z_t
            h_t = (1.0 - self.alpha) * mem + self.alpha * base
            h_t = v_t * h_t + (1.0 - v_t) * mem
            h_list.append(h_t)
            mem = h_t
        h = torch.stack(h_list, dim=1)
        self._mem_state[key] = mem
        return h, {key: mem}


__all__ = [
    'SimpleFrameTokenSelector',
    'StatMem'
]
