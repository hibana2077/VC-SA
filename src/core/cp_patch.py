# -*- coding: utf-8 -*-
"""
StatLite Modules (Drop-in)
==========================
兩個完全不含 Attention／SSM／GNN／Graph 的替代層，維持與現有原型相同 I/O：

(1) SimpleFrameTokenSelector
    - in : x ∈ [B, T, N, D], (mask 可選)  
    - out: z ∈ [B, Kf, Kt, D], frame_idx ∈ [B, Kf], token_idx ∈ [B, Kf, Kt], frame_mask ∈ [B, T], token_mask ∈ [B, T, N]
    - 影格：facility-location（覆蓋）+ 可學分數，加權貪婪近似，無注意力；
      token：k-Center Greedy（最大最小覆蓋），無圖。

(2) StatMem（Fréchet/EMA + ARP）
    - in : z ∈ [B, T, M, D], (valid_mask 可選)
    - out: h ∈ [B, T, M, D] 與持久化記憶 dict
    - 純統計：Approximate Rank Pooling (ARP) 作趨勢摘要 + 指數移動平均 (EMA) 作平滑；
      以「同一局部 token id」為 memory slot，無需建圖或注意力。

備註：僅相依 torch；可直接取代原本 FrameTokenCoSelector / GraphBasedMemBank。
"""
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Helpers
# -----------------------------

def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity with tiny eps for stability.
    a: [..., D], b: [..., D] -> sim: [..., a_len, b_len]
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.transpose(-1, -2)


def _facility_location_greedy(frame_repr: torch.Tensor, k: int, score: Optional[torch.Tensor] = None,
                              lambda_cov: float = 1.0, lambda_score: float = 0.0,
                              valid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch-wise facility-location 貪婪：
    frame_repr: [B, T, D]
    score:      [B, T] 或 None（若 None 則僅覆蓋）
    valid:      [B, T] 1/0，可選
    回傳：frame_idx [B, k], frame_mask [B, T]
    複雜度：O(B * T * k * T)，T≤~64 時通常足夠；必要時可做分塊/近似。
    """
    B, T, D = frame_repr.shape
    device = frame_repr.device
    sim_bt = []  # 先算每個 batch 的 T×T 相似度（cosine）
    for b in range(B):
        fb = frame_repr[b]  # [T, D]
        sim_bt.append(_safe_cosine(fb, fb))  # [T, T]
    sim = torch.stack(sim_bt, dim=0)  # [B, T, T]

    if score is None:
        score = torch.zeros(B, T, device=device)
    if valid is None:
        valid = torch.ones(B, T, device=device)

    frame_idx = torch.zeros(B, k, dtype=torch.long, device=device)
    frame_mask = torch.zeros(B, T, dtype=frame_repr.dtype, device=device)
    # best cover per "ground" element（被覆蓋者是所有 T 個影格表示）
    best_cover = torch.zeros(B, T, device=device)
    chosen = torch.zeros(B, T, dtype=torch.bool, device=device)

    for step in range(k):
        # 針對每個候選 j，計算加入後的覆蓋增益 sum_i max(best_cover_i, sim[i,j]) - best_cover_i
        # sim: [B, T, T]，取第二維作候選 j
        cov_gain = torch.relu(sim.max(dim=1).values * 0.0)  # 占位，形狀 [B, T]
        # 向量化計算增益：對每個 j，Δ = (max(best, sim[:, j]) - best).sum(dim=1)
        # 先展開：best_cover: [B, T] -> [B, T, 1] 方便與 sim[..., j] 對齊
        best = best_cover.unsqueeze(-1)  # [B, T, 1]
        # 對所有 j 一次算：
        # new_best = torch.maximum(best, sim) -> [B, T, T]
        new_best = torch.maximum(best, sim)
        cov_gain_all = (new_best - best).sum(dim=1)  # [B, T]（每個候選 j 的覆蓋增益）

        total_gain = lambda_cov * cov_gain_all + lambda_score * score
        # 不能選無效/已選
        total_gain = total_gain.masked_fill(~valid.bool(), float('-inf'))
        total_gain = total_gain.masked_fill(chosen, float('-inf'))

        j = torch.argmax(total_gain, dim=1)  # [B]
        frame_idx[:, step] = j
        # 更新狀態
        batch_arange = torch.arange(B, device=device)
        frame_mask[batch_arange, j] = 1.0
        # 更新 best_cover：best = max(best, sim[:, j])
        bj = sim[batch_arange, :, j]  # [B, T]
        best_cover = torch.maximum(best_cover, bj)
        chosen[batch_arange, j] = True

    return frame_idx, frame_mask


def _kcenter_greedy(X: torch.Tensor, k: int, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """k-Center Greedy（farthest-first）在單一集合上挑 k 個索引。
    X: [N, D]；valid: [N] 1/0，可選。
    回傳 idx: [k]
    """
    N, D = X.shape
    device = X.device
    if valid is None:
        valid = torch.ones(N, device=device, dtype=torch.bool)
    else:
        valid = valid.bool()

    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        # 全無效則退回 [0]*k
        return X.new_zeros(k, dtype=torch.long)

    Xv = X[valid_idx]  # [Nv, D]
    # 使用歐氏距離；先選離原點（或均值）最遠者作起點，這裡用對原點 L2 最大者。
    dists = (Xv ** 2).sum(dim=1)
    first = torch.argmax(dists)
    selected = [int(valid_idx[first].item())]

    # 維護每點到「已選集合」的最小距離
    min_dist = torch.cdist(X, X[selected].detach(), p=2).squeeze(-1)  # [N]
    min_dist[~valid] = -1.0  # 無效者不會被選中

    for _ in range(1, k):
        # 選擇目前距已選集合最遠者
        cand = torch.argmax(min_dist)
        selected.append(int(cand.item()))
        # 更新每點的最短距離
        new_d = torch.cdist(X, X[cand].unsqueeze(0).detach(), p=2).squeeze(-1)
        min_dist = torch.minimum(min_dist, new_d)
        min_dist[~valid] = -1.0

    return torch.tensor(selected, dtype=torch.long, device=device)


# -----------------------------
# (1) SimpleFrameTokenSelector
# -----------------------------
class SimpleFrameTokenSelector(nn.Module):
    """
    極簡 Frame/Token 共選器（無注意力、無圖）
    - 影格以 facility-location（覆蓋）+ 可學標量分數（MLP）做貪婪選 Kf。
    - 每格 token 以 k-Center Greedy 選 Kt（最大最小覆蓋）。

    參數：
      d_model: 特徵維度 D
      frame_topk: Kf；token_topk: Kt
      lambda_cov: 影格覆蓋權重
      lambda_score: 影格可學分數權重（=0 則只看覆蓋）
      use_cls: frame 池化是否用 CLS（否則用 mean）
    輸入：
      x: [B, T, N, D], mask: [B, T, N]（可選）
    輸出：
      z: [B, Kf, Kt, D], frame_idx: [B, Kf], token_idx: [B, Kf, Kt],
      frame_mask: [B, T], token_mask: [B, T, N]
    """

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

        # 可學的影格分數頭（標量）；lambda_score=0 時等於不啟用
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

        # ----- Frame-level pooling -----
        if self.use_cls and N >= 1:
            frame_repr = x[:, :, 0, :]  # [B, T, D]
            valid_frame = torch.ones(B, T, device=device, dtype=torch.bool)
        else:
            denom = mask.sum(dim=2, keepdim=False).clamp_min(1e-6)[..., None]  # [B, T, 1]
            frame_repr = (x * mask[..., None]).sum(dim=2) / denom  # [B, T, D]
            valid_frame = (denom.squeeze(-1) > 0)

        # 可學分數（標量）
        frame_score = self.frame_score_head(frame_repr).squeeze(-1)  # [B, T]

        # ----- Facility-location greedy for frames -----
        frame_idx, frame_mask = _facility_location_greedy(
            frame_repr, k=self.frame_topk, score=frame_score,
            lambda_cov=self.lambda_cov, lambda_score=self.lambda_score,
            valid=valid_frame.to(frame_repr.dtype)
        )  # [B, Kf], [B, T]

        # ----- Token-level selection per selected frame（k-Center） -----
        token_idx = torch.zeros(B, self.frame_topk, self.token_topk, dtype=torch.long, device=device)
        token_mask = torch.zeros(B, T, N, dtype=x.dtype, device=device)

        batch_arange = torch.arange(B, device=device)
        for b in range(B):
            for kf in range(self.frame_topk):
                t_sel = int(frame_idx[b, kf].item())
                X = x[b, t_sel]  # [N, D]
                m = mask[b, t_sel] > 0.5  # [N]
                idxs = _kcenter_greedy(X, self.token_topk, valid=m)
                token_idx[b, kf] = idxs
                token_mask[b, t_sel, idxs] = 1.0

        # ----- Gather 到壓縮輸出 [B, Kf, Kt, D] -----
        x_sel_frames = x[batch_arange[:, None], frame_idx]  # [B, Kf, N, D]
        z = x_sel_frames[batch_arange[:, None, None], torch.arange(self.frame_topk, device=device)[None, :, None], token_idx]
        return z, frame_idx, token_idx, frame_mask, token_mask


# -----------------------------
# (2) StatMem: ARP + EMA（純統計）
# -----------------------------
class StatMem(nn.Module):
    """
    統計式記憶層（無圖、無注意力）
      - ARP（Approximate Rank Pooling）捕捉趨勢/方向性
      - EMA（指數移動平均／Fréchet 均值的序列版）作平滑

    參數：
      d_model: D
      use_arp: 是否啟用 ARP（否則僅 EMA）
      window:  ARP/EMA 滑窗長度 W（=0 或 1 表示僅用當前步）
      alpha:   EMA 平滑係數 α ∈ (0,1]（=1 等於無 EMA）

    I/O：
      in : z [B, T, M, D], valid_mask [B, T, M]（可選）
      out: h [B, T, M, D]，以及持久化記憶 dict（跨 forward 延續）
    """
    def __init__(self, d_model: int, use_arp: bool = True, window: int = 8, alpha: float = 0.5):
        super().__init__()
        self.use_arp = use_arp
        self.window = max(1, int(window))
        self.alpha = float(alpha)
        self._mem_state: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _arp_window(z_seq: torch.Tensor, t: int, W: int, valid_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """對時間 t 作 ARP 加權視窗匯聚。
        z_seq: [T, M, D]（單一 batch）; valid_seq: [T, M] 或 None
        回傳: [M, D]
        權重 w_Δ = 2Δ - (L-1)，Δ=0..L-1（近似 rank pooling 封閉解權重；t 為最新）
        """
        device = z_seq.device
        start = max(0, t - W + 1)
        L = t - start + 1
        w = torch.arange(1, L + 1, device=device, dtype=z_seq.dtype)
        w = 2 * w - (L + 1)
        w = w / (w.abs().sum().clamp_min(1e-6))  # 正規化以免數值爆炸
        window = z_seq[start:t + 1]  # [L, M, D]
        if valid_seq is not None:
            v = valid_seq[start:t + 1].unsqueeze(-1).to(z_seq.dtype)  # [L, M, 1]
            window = window * v
            denom = (v * w.view(L, 1, 1)).abs().sum(dim=0).clamp_min(1e-6)
            return (window * w.view(L, 1, 1)).sum(dim=0) / denom
        else:
            return (window * w.view(L, 1, 1)).sum(dim=0)

    def forward(self,
                z: torch.Tensor,                 # [B, T, M, D]
                pos: Optional[torch.Tensor] = None,  # 佔位，為與舊接口一致（未使用）
                valid_mask: Optional[torch.Tensor] = None,  # [B, T, M]
                memory_id: Optional[str] = None,
                reset_memory: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, M, D = z.shape
        device = z.device
        if valid_mask is None:
            valid_mask = torch.ones(B, T, M, device=device, dtype=z.dtype)

        key = memory_id or "default"
        if reset_memory or (key not in self._mem_state):
            self._mem_state[key] = torch.zeros(B, M, D, device=device, dtype=z.dtype)
        mem = self._mem_state[key]  # [B, M, D]

        h_list = []
        for t in range(T):
            z_t = z[:, t]  # [B, M, D]
            v_t = valid_mask[:, t].unsqueeze(-1)  # [B, M, 1]

            if self.use_arp:
                # 逐 batch 做 ARP
                arp_t = torch.zeros_like(z_t)
                for b in range(B):
                    arp_t[b] = self._arp_window(z[b], t, self.window, valid_seq=valid_mask[b])
                base = arp_t
            else:
                base = z_t

            # EMA：h_t = (1-α)·h_{t-1} + α·base
            h_t = (1.0 - self.alpha) * mem + self.alpha * base
            # 若 token 無效，保持前一狀態（或置零），這裡採保留前一狀態
            h_t = v_t * h_t + (1.0 - v_t) * mem

            h_list.append(h_t)
            mem = h_t  # 更新持久化狀態（以局部 token id 作 slot）

        h = torch.stack(h_list, dim=1)  # [B, T, M, D]
        self._mem_state[key] = mem
        return h, {key: mem}


# -----------------------------
# Quick self-test (optional)
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, N, D = 2, 8, 64, 128
    Kf, Kt = 3, 6
    x = torch.randn(B, T, N, D)
    mask = (torch.rand(B, T, N) > 0.1).float()

    selector = SimpleFrameTokenSelector(d_model=D, frame_topk=Kf, token_topk=Kt, lambda_cov=1.0, lambda_score=0.1)
    z, fidx, tidx, fmask, tmask = selector(x, mask)
    print("z", z.shape, "fidx", fidx.shape, "tidx", tidx.shape)

    mem = StatMem(d_model=D, use_arp=True, window=4, alpha=0.5)
    h, md = mem(z, valid_mask=torch.ones(B, Kf, Kt))
    print("h", h.shape, "mem keys", list(md.keys()))
