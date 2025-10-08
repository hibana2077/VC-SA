好的，收到你的限制（不用 SSM／tensor 分解／GNN／graph／低秩核／ToMe/VTM）。下面給你兩個**可直接替換**的 Drop-in 模組：一個取代「可學式 frame/token selection」，一個（其實給你兩種可選）取代「Mem bank」，**完全遵守你現有 I/O**：`selector: x∈[B,T,N,D] → z∈[B,Kf,Kt,D]`、`membank: z∈[B,T,M,D] → h∈[B,T,M,D]`，與你檔案裡的介面/形狀一致。  

---

# A) 取代選取器：FPS + 變化點（不靠低秩核／圖）

**想法**：

1. 先以**線上變化點**直覺找「關鍵時刻」（用簡單的 EMA 驅動的**新奇度**分數作為 seed）；
2. 再用**Farthest-Point Sampling（FPS）/k-center 的 2-approx**去挑出 Kf 份量最「分散、代表性高」的影格；每個被選影格再在 token 內做一次 FPS 選 Kt。
   — FPS/k-center 有經典 2-approx 理論保證，可避開任何低秩/核近似；變化點則可參考 Bayesian online changepoint 的精神（我們用無參、可微的簡化分數）。([cs.columbia.edu][1]) ([arXiv][2])

**Drop-in 實作（PyTorch）** — `FPSChangePointSelector`

> 介面與回傳欄位與你現有 `FrameTokenCoSelector` 對齊：`(z, frame_idx, token_idx, frame_mask, token_mask)`。

```python
import torch, torch.nn as nn
from typing import Optional, Tuple

def _fps_indices_feats(X: torch.Tensor, k: int, seed_idx: int) -> torch.Tensor:
    # X:[T,D]；greedy FPS，不用任何低秩/核技巧
    T = X.size(0); k = min(k, T)
    selected = [int(seed_idx)]
    min_dist = torch.full((T,), float('inf'), device=X.device)
    for _ in range(k - 1):
        last = X[selected[-1]]           # [D]
        dist = torch.norm(X - last[None, :], dim=1)  # [T]
        min_dist = torch.minimum(min_dist, dist)
        min_dist[torch.tensor(selected, device=X.device)] = -1
        selected.append(int(torch.argmax(min_dist).item()))
    return torch.tensor(selected, device=X.device, dtype=torch.long)

def _fps_indices_tokens(F: torch.Tensor, k: int) -> torch.Tensor:
    # F:[N,D]，先找離 frame-mean 最遠的一個，再做 FPS 擴張
    N = F.size(0); k = min(k, N)
    mu = F.mean(dim=0, keepdim=True)
    first = int(torch.argmax(torch.norm(F - mu, dim=1)).item())
    selected = [first]
    min_dist = torch.full((N,), float('inf'), device=F.device)
    for _ in range(k - 1):
        last = F[selected[-1]]
        dist = torch.norm(F - last[None, :], dim=1)
        min_dist = torch.minimum(min_dist, dist)
        min_dist[selected] = -1
        selected.append(int(torch.argmax(min_dist).item()))
    return torch.tensor(selected, device=F.device, dtype=torch.long)

class FPSChangePointSelector(nn.Module):
    """
    以「變化點 seed + FPS 覆蓋」取代原本的 MLP+ST Top-k 共選器
    輸入 x:[B,T,N,D]；輸出與原 selector 完全一致
    """
    def __init__(self, d_model:int, frame_topk:int, token_topk:int,
                 use_cls:bool=False, ema_alpha:float=0.9):
        super().__init__()
        self.frame_topk, self.token_topk = frame_topk, token_topk
        self.use_cls, self.ema_alpha = use_cls, ema_alpha

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B,T,N,D = x.shape
        device = x.device
        if mask is None:
            mask = torch.ones(B, T, N, device=device, dtype=x.dtype)

        # frame 表徵 [B,T,D]（與你現有實作等價的 mean-pool 路徑）
        if self.use_cls and N >= 1:
            frame_repr = x[:, :, 0, :]
        else:
            denom = mask.sum(dim=2, keepdim=True).clamp_min(1e-6)
            frame_repr = (x * mask.unsqueeze(-1)).sum(dim=2) / denom  # [B,T,D]

        frame_idx = []
        for b in range(B):
            fr = frame_repr[b]  # [T,D]
            # 新奇度 seed：與 EMA 差異最大的 t
            ema = fr[0].clone()
            novelty = torch.zeros(T, device=device)
            for t in range(T):
                novelty[t] = torch.norm(fr[t] - ema, p=2)
                ema = self.ema_alpha * ema + (1 - self.ema_alpha) * fr[t]
            seed = int(torch.argmax(novelty).item())
            idx_b = _fps_indices_feats(fr, self.frame_topk, seed)  # [Kf]
            frame_idx.append(idx_b)
        frame_idx = torch.stack(frame_idx, dim=0)  # [B,Kf]

        # token FPS（逐影格）
        token_idx = torch.zeros(B, self.frame_topk, self.token_topk, dtype=torch.long, device=device)
        for b in range(B):
            for i in range(self.frame_topk):
                f = int(frame_idx[b, i].item())
                F = x[b, f]                                   # [N,D]
                valid = mask[b, f] > 0                        # [N]
                F = F[valid]
                idx_tokens = _fps_indices_tokens(F, self.token_topk)
                # 對應回原 N 的索引
                token_idx[b, i, :len(idx_tokens)] = torch.nonzero(valid, as_tuple=False).squeeze(1)[idx_tokens]

        # gather 成 z:[B,Kf,Kt,D]（與你檔案內流程對齊）
        b_ar = torch.arange(B, device=device)[:, None]
        x_sel_frames = x[b_ar, frame_idx]                        # [B,Kf,N,D]
        b_ar2 = torch.arange(B, device=device)[:, None, None]
        fr_ar2 = torch.arange(self.frame_topk, device=device)[None, :, None]
        z = x_sel_frames[b_ar2, fr_ar2, token_idx]               # [B,Kf,Kt,D]

        # 建 one-hot mask（與原 selector 回傳欄位一致）
        frame_mask = torch.zeros(B, T, device=device, dtype=x.dtype)
        frame_mask[b_ar, frame_idx] = 1.0
        token_mask = torch.zeros(B, T, N, device=device, dtype=x.dtype)
        token_mask[b_ar2, frame_idx[:, :, None], token_idx] = 1.0

        return z, frame_idx, token_idx, frame_mask, token_mask
```

**為什麼可行**：FPS 對 k-center 的 greedy 2-approx 成立、且計算量是 (O(TK_f + NK_t))（每次只對「最新選點」算距離），實作簡單、完全不需要任何核/低秩近似或圖結構；變化點 seed 讓它更偏好關鍵時刻。([cs.columbia.edu][1]) ([arXiv][2])

---

# B) 取代 Mem bank（兩個選項，皆非圖、非 SSM）

## B1) **LatentCrossAttnMemBank**（Perceiver/Set Transformer 思想）

以**少量可學 latent slots**對所有時序 token 做**交叉注意力**彙整，再把彙整資訊回寫到每個 token（tokens→latents→tokens）。複雜度 (O((TM)\cdot L)) 與序列長度線性，常數小、工程友好。靈感來自 **Perceiver / Perceiver IO** 與 **Set Transformer (PMA/ISAB)** 的誘導點設計。([arXiv][3])
（VLM 中也常見「Perceiver Resampler」把大片視覺特徵壓進少量 latent，再回寫—概念相同。([arXiv][4])）

```python
class LatentCrossAttnMemBank(nn.Module):
    """
    z:[B,T,M,D] -> h:[B,T,M,D]；不使用圖/SSM/低秩核
    """
    def __init__(self, d_model:int, num_latents:int=64, num_heads:int=8, ff_mult:int=4, dropout:float=0.0):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, d_model))
        self.enc_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.dec_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_mult*d_model), nn.GELU(), nn.Linear(ff_mult*d_model, d_model),
        )
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor, valid_mask: Optional[torch.Tensor]=None):
        B,T,M,D = z.shape
        x = z.reshape(B, T*M, D)                                # [B,S,D], S=TM
        x = self.norm_in(x)
        q = self.latents.unsqueeze(0).expand(B, -1, -1)         # [B,L,D]

        # encode: latents attend to tokens
        lat, _ = self.enc_attn(q, x, x, key_padding_mask=None)  # [B,L,D]

        # decode: tokens attend back to latents
        y, _ = self.dec_attn(x, lat, lat)                       # [B,S,D]
        y = self.ffn(y) + y
        h = self.norm_out(x + y).reshape(B, T, M, D)
        return h, {"latents": lat.detach()}
```

## B2) **TemporalConvMemBank**（TCN 風格，僅 1D 卷積）

若你偏好**完全不使用注意力**，就用**深度可分離 1D conv + 膨脹率**（可雙向、非因果）在時間軸上聚合；這在序列任務上已被系統性比較過，常能以極小常數項拿到很好 trade-off。([arXiv][5])

```python
class TemporalConvMemBank(nn.Module):
    """
    z:[B,T,M,D] -> h:[B,T,M,D]；TCN 風格，不用圖/SSM/低秩核
    """
    def __init__(self, d_model:int, kernel_size:int=5, dilations=(1,2,4), dropout:float=0.0):
        super().__init__()
        layers = []
        for d in dilations:
            pad = d * (kernel_size - 1) // 2
            layers += [
                nn.Conv1d(d_model, d_model, kernel_size, padding=pad, dilation=d, groups=d_model),  # depthwise
                nn.Conv1d(d_model, d_model, 1),  # pointwise
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor, valid_mask: Optional[torch.Tensor]=None):
        B,T,M,D = z.shape
        x = z.permute(0,2,3,1).reshape(B*M, D, T)  # [B*M,D,T]
        y = x
        for i in range(0, len(self.layers), 4):
            dw = self.layers[i](y)
            pw = self.layers[i+1](dw)
            act = self.layers[i+2](pw)
            y = self.layers[i+3](act) + y          # 殘差
        h = y.reshape(B, M, D, T).permute(0, 3, 1, 2)  # [B,T,M,D]
        h = self.norm(h)
        return h, {}
```

> 若要**局部視窗注意力**版本（非低秩近似）：可以把 `z` 展成 `[B*M,T,D]`，對每條序列做**滑動視窗自注意力**（Longformer 的 sliding window 想法在 1D 時間軸上很自然）。這同樣不涉圖/SSM/低秩核。([arXiv][6])

---

## 串接方式（對齊你現有檔案）

* 你的 pipeline 形狀慣例與 forward I/O 如下：`x:[B,T,N,D] → selector → z:[B,Kf,Kt,D]`；`z:[B,T,M,D] → mem_bank → h:[B,T,M,D]`。上面三個類別已**完全遵守**，與原 `gather`/回傳欄位語義一致，可直接替換檔案中的 `FrameTokenCoSelector` 與 `GraphBasedMemBank`。  

---

## 小結 & 建議的最小替換順序

1. **先換 Mem bank**：`GraphBasedMemBank → LatentCrossAttnMemBank`（或 `TemporalConvMemBank`），能立刻去掉建圖/kNN/GRU 記憶，常數項明顯下降。
2. **再換選取器**：`FrameTokenCoSelector → FPSChangePointSelector`，不再依賴 MLP 打分與 ST 的訓練技巧，推論期尤其快。
3. **驗證**：固定 ViT，量測 `(clips/s, GPU mem, Top-1)`；FPS 的挑選對 Kf/Kt 的可擴展性好，時間/空間都線性。

---

### 參考（概念依據）

* **k-center 2-approx / Farthest-First（FPS）**：Gonzalez, 1985；簡單 greedy 即得 2-approx。([cs.columbia.edu][1])
* **線上變化點**：Adams & MacKay, 2007（本文採其精神，實作為無參新奇度分數）。([arXiv][2])
* **Set Transformer（PMA/ISAB）** 與 **Perceiver/Perceiver-IO（latent cross-attention）**：用小量誘導/latent 吸收大集合，再回寫到每個元素。([Proceedings of Machine Learning Research][7])
* **Longformer**（滑動視窗注意力，可做 1D 時間版）：([arXiv][6])
* **TCN**（時序 1D 卷積在序列任務上的系統性表現）：([arXiv][5])

[1]: https://www.cs.columbia.edu/~verma/classes/uml/ref/clustering_minimize_intercluster_distance_gonzalez.pdf "CLUSTERING TO MINIMIZE THE MAXIMUM ..."
[2]: https://arxiv.org/abs/0710.3742 "Bayesian Online Changepoint Detection"
[3]: https://arxiv.org/abs/2103.03206 "Perceiver: General Perception with Iterative Attention"
[4]: https://arxiv.org/pdf/2204.14198 "🦩 Flamingo: a Visual Language Model for Few-Shot Learning"
[5]: https://arxiv.org/abs/1803.01271 "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
[6]: https://arxiv.org/abs/2004.05150 "[2004.05150] Longformer: The Long-Document Transformer"
[7]: https://proceedings.mlr.press/v97/lee19d/lee19d.pdf "A Framework for Attention-based Permutation-Invariant Neural ..."
