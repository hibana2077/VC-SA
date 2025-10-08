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

        # ---- Gather 壓縮輸出到 [B, Kf, Kt, D] ----
        # 先 gather 影格
        batch_arange = torch.arange(B, device=x.device)[:, None]
        x_sel_frames = x[batch_arange, frame_idx]  # [B, Kf, N, D]
        token_idx = token_idx_all[batch_arange, frame_idx]  # [B, Kf, Kt]

        # 再 gather 每格 token
        batch_arange2 = torch.arange(B, device=x.device)[:, None, None]
        frame_arange2 = torch.arange(self.frame_topk, device=x.device)[None, :, None]
        z = x_sel_frames[batch_arange2, frame_arange2, token_idx]  # [B, Kf, Kt, D]

        return z, frame_idx, token_idx, frame_mask, token_mask


class GraphBasedMemBank(nn.Module):
    """
    (B) Graph-Based MemBank（輕量時序 GNN）
    - 節點：共選後 token 特徵（z ∈ [B, T, M, D]）
    - 邊：時間近鄰（|Δt| ≤ temporal_window） + 語義 kNN（cosine top-k）
    - 消息傳遞：簡化版 GAT / 加權平均
    - 記憶：對每個「空間 token id」維護一個 memory（此 MVP 以位置 idx 作為 id；若無位置，可用每格的局部 idx）
    參數：
      d_model: 特徵維度
      knn_k: 語義鄰居數
      temporal_window: 與前幾格連邊的範圍（>=1）
      num_layers: 消息傳遞層數
      use_gat: True=GAT 注意力；False=平均聚合
      mem_dim: 記憶維度（GRUCell 隱層）
    forward 輸入：
      z: [B, T, M, D]（共選後）
      pos: [B, T, M, 2]（可選，若提供可加到語義相似度的 bias）
      valid_mask: [B, T, M] 1/0（可選）
    forward 輸出：
      h: [B, T, M, D]  消息傳遞 + 記憶後的更新特徵
      mem: Dict[str, torch.Tensor]  最終記憶字典（可跨 clip 繼續喂）
    """
    def __init__(
        self,
        d_model: int,
        knn_k: int = 8,
        temporal_window: int = 2,
        num_layers: int = 2,
        use_gat: bool = True,
        mem_dim: Optional[int] = None,
    ):
        super().__init__()
        self.knn_k = knn_k
        self.temporal_window = max(1, temporal_window)
        self.num_layers = num_layers
        self.use_gat = use_gat
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        mem_dim = mem_dim or d_model
        self.mem_cell = nn.GRUCell(input_size=d_model, hidden_size=mem_dim)
        self.mem_to_feat = nn.Linear(mem_dim, d_model)

        # GAT-style 注意力參數
        if self.use_gat:
            self.att_q = nn.Linear(d_model, d_model, bias=False)
            self.att_k = nn.Linear(d_model, d_model, bias=False)
            self.att_vec = nn.Linear(d_model, 1, bias=False)

        # 內部「跨批次」記憶，實務上建議由上層管理重置/持久化
        self._mem_state: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _cosine_topk(q: torch.Tensor, k: torch.Tensor, topk: int):
        # q: [Lq, D], k: [Lk, D]
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        sim = q_norm @ k_norm.t()  # [Lq, Lk]
        vals, idx = torch.topk(sim, k=min(topk, k.shape[0]), dim=-1)
        return idx, vals

    def _edge_index_and_weights(
        self,
        z_seq: torch.Tensor,  # [T, M, D]
        pos_seq: Optional[torch.Tensor] = None,  # [T, M, 2]
        valid_seq: Optional[torch.Tensor] = None,  # [T, M]
    ):
        """
        建圖（僅在單一 batch 內）：對每個時間 t 的每個節點，從 t-Δ..t 的節點取 kNN 鄰居。
        回傳列表 edges[(t, i)] = (nbr_idx, nbr_w)；其中 i 是該格的第 i 個 token。
        """
        T, M, D = z_seq.shape
        if valid_seq is None:
            valid_seq = torch.ones(T, M, device=z_seq.device, dtype=z_seq.dtype)

        edges = []
        for t in range(T):
            # 查詢集合 Q：第 t 格有效節點
            q = z_seq[t]  # [M, D]
            q_mask = valid_seq[t]  # [M]
            q_idx = torch.nonzero(q_mask > 0.5, as_tuple=False).squeeze(-1)
            if q_idx.numel() == 0:
                edges.append(([], []))
                continue

            # 鍵集合 K：從 t-Δ..t 之間所有有效節點（含當前格，允許自鄰）
            k_list, id_map = [], []
            for dt in range(self.temporal_window + 1):
                tt = t - dt
                if tt < 0:
                    break
                k_mask = valid_seq[tt]
                k_idx = torch.nonzero(k_mask > 0.5, as_tuple=False).squeeze(-1)
                if k_idx.numel() == 0:
                    continue
                k_list.append(z_seq[tt][k_idx])  # [Mk, D]
                # 儲存 (tt, local_id) 以便之後反查
                id_map += [(tt, int(ii)) for ii in k_idx.tolist()]

            if len(k_list) == 0:
                edges.append(([], []))
                continue

            K = torch.cat(k_list, dim=0)  # [Lk, D]
            # 逐查詢點做 top-k
            q_feat = q[q_idx]  # [Lq, D]
            nbr_idx, nbr_w = self._cosine_topk(q_feat, K, self.knn_k)  # [Lq, k]
            # 轉回全域 (t', i') 索引
            nbr_global = []
            for row in nbr_idx.tolist():
                nbr_global.append([id_map[j] for j in row])  # List[(t', i')]
            edges.append(([(t, int(i.item())) for i in q_idx], (nbr_global, nbr_w)))
        return edges  # List per t：([targets], (neighbors(list of list), weights[tensor]))

    def _gat_message(self, q, K):
        # 單頭 GAT 風格打分
        qh = self.att_q(q)  # [Lq, D]
        kh = self.att_k(K)  # [Lk, D]
        # 注意：這裡做簡化，對每個 q 與其候選 K 計分時可重複利用 kh
        # 直接使用 cosine 已有 weights，此函數僅在需要額外學習注意力時使用
        return qh, kh

    def forward(
        self,
        z: torch.Tensor,                  # [B, T, M, D]
        pos: Optional[torch.Tensor] = None,   # [B, T, M, 2]
        valid_mask: Optional[torch.Tensor] = None,  # [B, T, M]
        memory_id: Optional[str] = None,  # 若提供，將跨 forward 持久化記憶
        reset_memory: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, M, D = z.shape
        if valid_mask is None:
            valid_mask = torch.ones(B, T, M, device=z.device, dtype=z.dtype)

        h_list = []
        mem_key = memory_id or "default"
        if reset_memory or (mem_key not in self._mem_state):
            self._mem_state[mem_key] = torch.zeros(B, M, self.mem_cell.hidden_size, device=z.device)

        mem = self._mem_state[mem_key]  # [B, M, H]

        for t in range(T):
            h_t = z[:, t]  # [B, M, D]
            vmask_t = valid_mask[:, t]  # [B, M]

            # （語義/時間）鄰居選擇與消息傳遞（分批做）
            # 這裡以單批次循環實作，易讀優先；實務可合併張量化以提速
            updated = torch.zeros_like(h_t)
            for b in range(B):
                h_seq = z[b]  # [T, M, D]
                v_seq = valid_mask[b]  # [T, M]
                edges = self._edge_index_and_weights(h_seq, pos_seq=None, valid_seq=v_seq)

                # 取出時間 t 的 target 與其鄰居
                targets, (nbrs, weights) = edges[t]  # targets: List[(t,i)]（其實 t 相同）
                if len(targets) == 0:
                    continue
                tgt_local_idx = [i for (_, i) in targets]
                q = h_seq[t][tgt_local_idx]  # [Lq, D]

                # 聚合訊息
                msgs = []
                for row, w in zip(nbrs, weights):  # row: List[(tt, ii)], w: [k]
                    K = torch.stack([h_seq[tt][ii] for (tt, ii) in row], dim=0)  # [k, D]
                    if self.use_gat:
                        qh, kh = self._gat_message(q[q.new_zeros(()).long()], K)  # broadcast-safe trick
                        # 用 cosine 權重 w 作為基權，GAT 再細調（簡化：點積）
                        att = torch.softmax((qh[0:1] * kh).sum(-1), dim=-1)  # [k]
                        alpha = 0.5 * w + 0.5 * att  # 融合
                    else:
                        alpha = w  # 僅用語義相似權重
                    msgs.append((alpha.unsqueeze(-1) * self.v_proj(K)).sum(dim=0))  # [D]

                msg = torch.stack(msgs, dim=0)  # [Lq, D]
                # 將訊息寫回對應的 target
                updated[b, tgt_local_idx] = msg

            # 記憶更新：以每格的「相同局部 token id」作為 memory slot（M 個）
            # msg 與當前 h_t 融合，再用 GRUCell 更新
            inp = self.out(self.norm(updated + self.q_proj(h_t)))  # [B, M, D]
            mem = self.mem_cell(inp.reshape(-1, D), mem.reshape(-1, self.mem_cell.hidden_size)).reshape(B, M, -1)
            h_t_out = h_t + self.mem_to_feat(mem)  # 殘差注入記憶
            h_list.append(h_t_out)

        h = torch.stack(h_list, dim=1)  # [B, T, M, D]
        self._mem_state[mem_key] = mem  # 持久化
        return h, {mem_key: mem}