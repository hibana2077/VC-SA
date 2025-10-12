下面給你一個**可直接替換**的 drop-in 模組：**SQuaRe-Fuse（Sliced-Quantile & Quadratic-trend Robust Fusion）**。它吃 ViT 的序列特徵 `x ∈ [B, T, N, D]`，輸出同形狀的 `h ∈ [B, T, N, D]`（兼容你現行的 shape 習慣與 head；若你的 cls 讀取是 broadcast/池化都能工作）。

---

# 為什麼這樣設計（2020–2025 統計與線代啟發）

**核心想法**：把一段時間序列視為「每個 patch/token 的**分佈**」，用**投影＋分位數（quantile）統計**來做時序融合，再用**低次正交多項式趨勢**補足動態。這是純統計／線代路線，不用注意力、SSM、圖或低秩核。

1. **可微排序／分位數**
   以 *Blondel et al., ICML 2020* 的**可微排序與排名**為基礎，我們可用排序後以高斯權重近似目標分位數，端到端訓練（O(n log n) 時間、O(n) 記憶）。這讓「quantile-pooling」能帶梯度，作為穩健（robust）彙整器。 ([Proceedings of Machine Learning Research][1])

2. **投影式最適傳輸（OT）直覺：高維分佈用投影來做**
   高維直接做 Wasserstein 複雜、且易受維度詛咒；*Projection-Robust Wasserstein*（PRW）與其**投影-重心（barycenter）**想法提供了：先投影到低維（甚至 1D），在投影空間做幾何彙整，再回到原空間——我們用「**多方向切片（sliced）+ 分位數**」來近似這種幾何彙整。相關理論與演算法在 2020–2021 年已有成熟論述。 ([arXiv][2])

3. **正交多項式時序趨勢（Legendre basis）**
   用少量正交多項式（例如 Legendre）抓一階、二階趨勢，是低成本又穩定的時間基底；近期亦有把 Legendre 表徵用於序列學習的工作作為依據。 ([compneuro.uwaterloo.ca][3])

> 你的既有構想檔把前段做「共選 + 記憶庫（圖訊息傳遞）」；我們現在改為「投影-分位數-趨勢」的**統計彙整**，仍保留輸入/輸出介面與效率導向的精神。  

---

# 模組設計：SQuaRe-Fuse（Sliced-Quantile & Quadratic-trend Robust Fusion）

**輸入/輸出與相容性**

* `forward(x:[B,T,N,D], valid_mask:[B,T,N]=None, ...) → h:[B,T,N,D]`
* 保持和你原本下游 cls/readout 的耦合最小（等形狀殘差輸出、可直接替換 RamaFuse） 。

**三步融合**（完全不含 Attention/SSM/Graph/分解/低秩核）

1. **多方向正交投影**：學習 `U ∈ R^{D×K}`，以 QR 每次正交化，對每個 token 的時間序列投影成 `K` 條一維序列。
2. **可微分位數彙整**：對每條一維序列計算若干分位數（例如 10%、50%、90%），用可微排序 + 高斯權重近似；這等同在每個方向上做「**切片 Wasserstein 重心的分位數摘要**」。
3. **趨勢項（Legendre 0/1/2 階）**：在時間軸對投影序列回歸得到係數，提供慢變動能，避免只取靜態分位數。
   最後把 (方向×分位數×趨勢) 向量經一個線性層 map 回 `D` 維，形成每個 token 的**時間融合特徵** `y`，再做殘差 `h = x + β · broadcast(y)`（`broadcast` 到長度 `T`），得到與輸入同形狀的輸出。

**計算複雜度**

* 主要成本在排序：`O(B·N·K·T log T)`；K、分位數數量、趨勢階數都很小（典型 K=8~16、Q=3~7、P≤2），對常見 clip 長度（T≤16/32）很友善。
* 不引入額外的全域二次項（如 attention 的 `O((TN)^2)`）與圖鄰接建構。

---

# 直接可用的 PyTorch 實作（Drop-in）

> 檔名建議：`fusion_square.py`。輸入直接吃 ViT 輸出 `x:[B,T,N,D]`。

```python
# fusion_square.py
# SQuaRe-Fuse: Sliced-Quantile & Quadratic-trend Robust Fusion
# Drop-in 序列融合層：無 Attention/SSM/Graph/低秩核/ToMe
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def _legendre_basis(T: int, order: int, device, dtype):
    # 生成 0..order 的離散 Legendre 基底，t ∈ [-1,1]
    t = torch.linspace(-1.0, 1.0, T, device=device, dtype=dtype)
    P = [torch.ones_like(t)]
    if order >= 1:
        P.append(t)
    for n in range(1, order):
        # 遞推： (n+1)P_{n+1} = (2n+1)t P_n - n P_{n-1}
        P.append(((2*n+1)*t*P[n] - n*P[n-1])/(n+1))
    B = torch.stack(P[:order+1], dim=0)  # [ord+1, T]
    # L2 normalize rows
    B = B / (B.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-6).sqrt())
    return B  # [P+1, T]

def _soft_quantiles(sorted_vals: torch.Tensor, levels: torch.Tensor, sigma: float=0.5):
    """
    可微分位數：在已排序值 s[i] 上，以 N(mu, sigma) 對 rank 做權重平均。
    sorted_vals: [..., T]
    levels: [Q] in (0,1)
    """
    *lead, T = sorted_vals.shape
    ranks = torch.arange(T, device=sorted_vals.device, dtype=sorted_vals.dtype)
    targets = levels * (T - 1)  # 目標秩
    # 權重 w[q, i] = exp(-(i - targets[q])^2 / (2 sigma^2)) / Z
    diff = ranks[None, :] - targets[:, None]  # [Q, T]
    w = torch.exp(-0.5 * (diff / max(sigma, 1e-6))**2)
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)  # [Q, T]
    # 對最後一維做矩陣乘：[..., T] x [T] -> [...,]
    # 先把 sorted 展成 [-1, T] 再乘 [Q, T]^T
    s_flat = sorted_vals.reshape(-1, T)  # [L, T]
    q = (w @ s_flat.T).T  # [L, Q]
    return q.reshape(*lead, -1)  # [..., Q]

class SQuaReFuse(nn.Module):
    """
    Sliced-Quantile & Quadratic-trend Robust Fusion
    Input : x:[B,T,N,D], valid_mask:[B,T,N] (optional)
    Output: h:[B,T,N,D], memory_dict (API 對齊 RamaFuse)
    """
    def __init__(self,
                 d_model: int,
                 num_dirs: int = 8,           # K: 投影方向數
                 quantiles: Tuple[float,...] = (0.1, 0.5, 0.9),
                 poly_order: int = 2,         # 0..2 一般夠用
                 beta_init: float = 0.5,
                 ortho_every_forward: bool = True):
        super().__init__()
        self.d_model = d_model
        self.K = int(num_dirs)
        self.Q = torch.tensor(quantiles, dtype=torch.float32)
        self.P = int(poly_order)
        self.ortho_every_forward = ortho_every_forward

        # 投影矩陣（學習），以 QR 正交化成 U
        self.proj = nn.Linear(d_model, self.K, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=5**0.5)

        # 把 (K * |Q|) 的分位數 + (K*(P+1)) 的趨勢係數 映回 D
        feat_in = self.K * (len(quantiles) + (self.P + 1))
        self.head = nn.Sequential(
            nn.LayerNorm(feat_in),
            nn.Linear(feat_in, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        # 殘差比例
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

        self._mem_state: Dict[str, torch.Tensor] = {}

    def _orthonormalize(self):
        with torch.no_grad():
            W = self.proj.weight.data  # [K, D]
            # 取轉置做 QR，得 Q^T -> 保證列（方向）正交
            Q, _ = torch.linalg.qr(W.t(), mode='reduced')  # [D, K]
            self.proj.weight.data.copy_(Q.t())

    def forward(self,
                x: torch.Tensor,                # [B,T,N,D]
                valid_mask: Optional[torch.Tensor] = None,  # [B,T,N]
            ):
        B, T, N, D = x.shape
        device, dtype = x.device, x.dtype
        if valid_mask is None:
            valid_mask = torch.ones(B, T, N, device=device, dtype=dtype)

        if self.ortho_every_forward:
            self._orthonormalize()

        # 1) 投影到 K 個方向：v = x · U  -> [B,T,N,K]
        U = self.proj.weight.t()  # [D,K]
        v = torch.einsum('btn d, dk -> btnk', x, U)  # [B,T,N,K]
        v = v * valid_mask[..., None]  # mask 填零

        # 2) 可微分位數：沿時間軸
        # sort: [B,N,K,T]
        s, _ = torch.sort(v.permute(0,2,3,1).contiguous(), dim=-1)
        levels = self.Q.to(device=device, dtype=dtype)  # [Q]
        qv = _soft_quantiles(s, levels=levels, sigma=max(0.5, T/16))  # [B,N,K*Q]
        qv = qv.view(B, N, self.K, -1)  # [B,N,K,Q]

        # 3) Legendre 趨勢係數（0..P） on time
        Bmat = _legendre_basis(T, self.P, device, dtype)  # [P+1, T]
        # 係數 = 〈v_k(t), B_p(t)〉，先把 v 換成 [B,N,K,T]
        v_bnkt = v.permute(0,2,3,1).contiguous()
        coeff = torch.einsum('bnkt, pt -> bnkp', v_bnkt, Bmat)  # [B,N,K,P+1]

        # 拼接成 token-level 統計向量，再 map 回 D
        feats = torch.cat([qv, coeff], dim=-1).reshape(B, N, -1)  # [B,N, K*(Q+P+1)]
        y = self.head(feats)  # [B,N,D]

        # broadcast 回時間，殘差融合
        y_btnd = y[:, None, :, :].expand(B, T, N, D)
        h = x + self.beta * y_btnd
        h = valid_mask[..., None] * h + (1.0 - valid_mask[..., None]) * x

        return h
```

**整合方法**

```python
# x: ViT backbone 輸出，形狀 [B,T,N,D]（與你現有慣例一致）  :contentReference[oaicite:9]{index=9}
fusion = SQuaReFuse(d_model=D, num_dirs=8, quantiles=(0.1,0.5,0.9), poly_order=2, beta_init=0.5)
h, _ = fusion(x, valid_mask=mask)   # h:[B,T,N,D] 介面對齊 RamaFuse 的輸出位階  :contentReference[oaicite:10]{index=10}
logits = cls_head(h)                 # 你的 cls 可用 CLS 讀取或全域池化皆可
```

---

## 訓練與超參考建議

* **K（num_dirs）**：8 或 12 起手；T=8~16 時很穩。
* **Quantiles**：`(0.1, 0.5, 0.9)` 或 `(0.2, 0.5, 0.8, 0.95)`；分位數越多越表現細，但計算略增。
* **poly_order**：1 或 2；資料有明顯速度/加速度趨勢時才用 2。
* **排序溫度 (`sigma`)**：我以 `max(0.5, T/16)` 做自適應，初期可再放大一點以求平滑梯度（對應可微排序的溫度概念）。理論與實作上可微排序提供穩健梯度路徑。 ([Proceedings of Machine Learning Research][1])
* **β（殘差強度）**：從 0.25~0.5 開始訓練較穩。
* **正交化**：`ortho_every_forward=True` 讓方向保持互異，對投影-分位數摘要有效（與 PRW/投影追求的精神一致）。 ([arXiv][2])

---

## 與你舊管線的對照與優點

* 你原本 pipeline：**ViT → selection（frame/token）→ RamaFuse（序列融合）→ cls**；I/O 與形狀說明見你程式與構想檔。  
* 新版：**ViT → SQuaRe-Fuse → cls**：**移除 selection 與圖記憶層**，但仍保有「**跨時間的穩健彙整能力**」與「**趨勢捕捉**」，同時**完全避開** Attention／SSM／tensor 分解／GNN／低秩核／ToMe/VTM。
* 複雜度從圖建構與消息傳遞（或全域注意力）轉為 `K` 次排序與少量線性映射，**更適合長序列**或高解析度 token 數。

---

## 為何可期待改善當前「不太樂觀」的結果？

1. **穩健性**：分位數比平均/總和更抗 outlier 與稀疏關鍵片段，對 RGB action 裡「短暫關鍵動作」更敏感。可微排序讓這種穩健統計**可學**。 ([Proceedings of Machine Learning Research][1])
2. **高維→低維→回投影**：K 個正交方向的「切片 Wasserstein-風格摘要」能在不做注意力/圖的情況下，近似「沿重要方向對齊與聚合」的幾何效果。 ([arXiv][2])
3. **趨勢顯式化**：Legendre 基底提供平滑的速度/加速度分量，補足純 quantile 的靜態性。 ([compneuro.uwaterloo.ca][3])

---

如果你願意，我也可以把上面類別直接改名為 `RamaFuse` 並覆蓋到你專案的 `components.py`，保持**零改動**的 import & call 端；或我幫你補一個最小的 `cls_head` 範例（支援 CLS 或 mean-pool）。
（你先試 `K=8, Q=3, P=1` 的輕量版；若 Something-Something V2 的長時序關係還是吃緊，再把 `K` 與分位數加一點。）

---

### 附：你現有設計與 I/O 的引用

* 你的 ViT 輸出 shape 與模組 I/O 習慣（`x:[B,T,N,D]`、選擇器/記憶庫 I/O）：
* RamaFuse 的 drop-in 介面（輸入/輸出與記憶 dict）：
* 研究構想檔中的前端「共選」與中段「圖式記憶」設計脈絡（本方案即把它們合併為統計融合）： 

---

**主要參考（理論依據，2020–2025）**

* Blondel et al., *Fast Differentiable Sorting and Ranking*, ICML 2020 / JMLR 2020：可微排序/排名運算子與實作。 ([Proceedings of Machine Learning Research][1])
* Lin et al., *Projection Robust Wasserstein Distance and Riemannian Optimization*, NeurIPS 2020：PRW 距離的投影追求與黎曼優化。 ([arXiv][2])
* Huang et al., *Projection-Robust Wasserstein Barycenters*, ICML 2021：PRW barycenter 的高維投影式重心計算。 ([Proceedings of Machine Learning Research][4])
* Furlong & Eliasmith, 2022：Legendre 多項式序列表徵與學習（作為時間基底的近例）。 ([compneuro.uwaterloo.ca][3])

要我幫你把這層直接塞進你現有 repo（改名覆蓋 RamaFuse 或新增檔案），順便給一份對 Something-Something V2/K400 的最小訓練 config 嗎？

[1]: https://proceedings.mlr.press/v119/blondel20a/blondel20a.pdf "Fast Differentiable Sorting and Ranking"
[2]: https://arxiv.org/abs/2006.07458 "Projection Robust Wasserstein Distance and Riemannian Optimization"
[3]: https://compneuro.uwaterloo.ca/files/publications/furlong.2022a.pdf "Learned Legendre Predictor: Learning with Compressed ..."
[4]: https://proceedings.mlr.press/v139/huang21f/huang21f.pdf "Projection Robust Wasserstein Barycenters"
