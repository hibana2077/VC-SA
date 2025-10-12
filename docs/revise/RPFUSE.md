# RPFuse（Ramanujan Periodic Fusion）

**核心想法**
把每個 token 的特徵（或經過少量方向投影後的特徵）視為一條長度 `T` 的訊號，**投影到一組以 Ramanujan sums 為基底的「整數週期子空間」**，分解出各個整數週期（1,2,3,…,Q）上的能量與成分，再把這些**可解釋的週期成分**重建為一個殘差，回灌到原特徵。

* **可解釋性**：每一個係數都對應**明確的整數週期 q**；能量譜直接告訴你該 token/通道主要受哪些週期驅動。
* **理論根基**：Ramanujan sums (c_q(n)) 定義於整數週期 q 的本原根之和，能構成**「正好 q-週期」**（exactly q-periodic）的子空間 (S_q)，其維度為歐拉函數 (\varphi(q))，並可用投影矩陣 (P_q) 進行正交投影與分解；更進一步有 **Ramanujan Periodic Transform (RPT)** 與 **Ramanujan Subspace Pursuit (RSP)** 的投影與分解程序可處理**任意長度**訊號（非 q 的整數倍時，採 block-mean 投影定理）。這些性質提供了我們模組的可解釋與穩定基礎。 ([維基百科][1])

> 參考基礎：
> • Ramanujan’s sum 定義與例子。([維基百科][1])
> • Ramanujan 子空間 (S_q) 的構造、(\dim S_q=\varphi(q))、投影矩陣 (P_q)、與任意長度投影（block-mean 定理）。
> • RPT/RSP 線索與「正好週期」分解的可行性與正交性脈絡。([科學直接][2])

**和你現有骨架的關係**
現在的骨架是 `ViT(影像預訓練) → BDRFuse → cls`。RPFuse 會**取代 BDRFuse**，仍回傳同形狀 `[B,T,N,D]` 並帶一個 `info` 字典（含每個 q 的能量譜，便於解釋與可視化）。你目前檔案裡 BDRFuse 的設計我已對齊 I/O 與風格（殘差 + 每通道 gate + 可選正交化）；只要把呼叫處換成 `RPFuse` 即可。

---

## 為何這顆有機會達到 CVPR 級「創新 + 理論」？

1. **新型序列融合觀點**：主流時序融合仰賴注意力/SSM/低秩等，我們改以**整數週期子空間分解**。它能把「時序模式」用 (q\in\mathbb{Z}^+) 的**可解釋週期光譜**呈現，天然具備**循環平移穩健性**（能量對相位不敏感，可透過移位平均或能量聚合達成），並避免黑盒注意力的歧異性。理論上 (S_q) 是**正好 q-週期的子空間**，以 (P_q) 投影得成分；多個 (S_q) 的結構、維度、以及對任意長度訊號之投影皆有建構與證明。
2. **可解釋的「週期能量路由」**：殘差是由各 (q) 的係數（能量）重建而來，**每一維增強都能追溯到特定週期**；這比 DCT 的頻帶更貼近「整數週期事件」（步態、心跳、呼吸、擺動、周期性動作單元），且與視覺影片的離散幀天然契合。
3. **理論小貢獻的切入點（建議寫在論文）**：

   * 把 **RSP/RPT 的投影理論**搬到**多通道、多 token 的高維特徵**，提出**「Ramanujan-guided residual fusion」**形式，證明殘差項在特定假設下等價於對各 (S_q) 的最小平方重建（在採 QR 正交化後的基底上）；
   * 給出**「能量守恆 / 分配」**的命題：在正交化基底下，重建能量等於各 (q) 係數能量和（Parseval 型式）；
   * 分析**遮罩與不等長片段**對投影的影響，使用 block-mean 投影法則保證一致性。

---

## Drop-in 實作（PyTorch）

直接把這段貼進你的 `components.py`，並用 `RPFuse` 取代 `BDRFuse`。介面與回傳 info 結構與你原 BDRF 類似（含 `beta` gate、可選方向投影正交化、non-negative head 方便解釋）。

```python
# -----------------------------------------
# RPFuse: Ramanujan Periodic Fusion (drop-in)
# -----------------------------------------
from typing import Optional, Dict, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    rows = [ _ramanujan_sum_row(q, T, device, dtype) for q in qs ]
    B = torch.stack(rows, dim=0)  # [L, T], L = len(qs)
    # Optional: orthonormalize rows via QR on B^T to improve conditioning
    # This keeps the span of Ramanujan rows but yields orthonormal basis R.
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
    Output: h:[B,T,N,D], info: {'periods': List[int], 'energy_q': Tensor, 'W_dir2ch': Tensor}
    禁用: Attention / SSM / Tensor decomposition / Graph / 低秩核 / ToMe/VTM
    """
    def __init__(
        self,
        d_model: int,
        num_dirs: int = 8,     # 少量方向以降維，保留可解釋殘差路由
        q_max: int = 16,       # 最多考慮的整數週期
        ortho_dirs: bool = True,
        beta_init: float = 0.5
    ):
        super().__init__()
        self.d_model = d_model
        self.K = int(num_dirs)
        self.q_max = int(q_max)
        self.ortho_dirs = bool(ortho_dirs)

        # 小型方向投影 (D -> K)，可選每步正交化，僅作方向壓縮，不涉注意力
        self.proj = nn.Linear(d_model, self.K, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=5 ** 0.5)

        # 時間重建後 (K) -> (D) 的非負映射，提升可解釋性
        self.head = _NonnegLinear(self.K, d_model)

        # 殘差 gate
        self.beta = nn.Parameter(torch.full((d_model,), float(beta_init)))

        # basis cache
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

        # 1) 通道方向降維: v:[B,T,N,K]
        v = torch.einsum('btnd,dk->btnk', x, self.proj.weight.t())  # [B,T,N,K]
        v = v * valid_mask[..., None]

        # 2) Ramanujan 基底 (row-orthonormal): R:[L,T], periods:qs
        R, qs = self._get_basis(T, device, dtype)  # [L, T]
        # 係數 (對時間做投影) alpha:[B,N,K,L]
        v_bnkt = v.permute(0, 2, 3, 1).contiguous()  # [B,N,K,T]
        alpha = torch.einsum('bnkt,lt->bnkl', v_bnkt, R)  # 投影係數

        # 3) 能量譜 (每個 q 的能量，可解釋)
        energy_q = alpha.square().mean(dim=(0,1,2))  # [L]，全域平均；也可回傳更細粒度

        # 4) 依基底重建 (K 維時間訊號)，再映射回 D 維：y:[B,T,N,D]
        v_hat_bnkt = torch.einsum('bnkl,lt->bnkt', alpha, R)  # [B,N,K,T]
        v_hat = v_hat_bnkt.permute(0, 3, 1, 2).contiguous()   # [B,T,N,K]
        y_ch, W_dir2ch = self.head(self.proj.weight.t())      # 映射 K->D 的非負權重（可解釋）
        # 用固定的非負映射把 K->D（不依賴時間/位置，避免過擬合）
        y_btnd = torch.einsum('btnk,kd->btnd', v_hat, F.softplus(self.head.W_raw))

        # 5) 殘差融合 + gate
        beta = torch.sigmoid(self.beta)[None, None, None, :]
        h = x + beta * y_btnd
        h = valid_mask[..., None] * h + (1.0 - valid_mask[..., None]) * x

        info = {
            'periods': qs,               # 使用的整數週期
            'energy_q': energy_q,        # 各 q 的全域能量（可視化/監控）
            'W_dir2ch': F.softplus(self.head.W_raw).detach()  # 解釋方向->通道的貢獻
        }
        return h, info
```

**更動方式**
把你原本 `BDRFuse(d_model, ...)` 的建構與呼叫，替換成 `RPFuse(d_model, num_dirs=8, q_max=16)` 即可；I/O 與回傳格式維持一致（`(h, info)`），不需要改動上下游。

---

## 訓練與使用建議

* **損失正則**

  * `L_cls + λ1 * ||alpha||_1 + λ2 * H(p_q)`：其中 (p_q \propto \text{energy}_q)（softmax 後），透過 **L1 促進少數週期稀疏**、**熵正則抑制過度分散**。
  * **增強一致性**：同一 clip 的兩種時序增強（crop/stride/shift）下的 (p_q) 做 KL 一致性。
* **超參建議**：`num_dirs=8~16`, `q_max=16~32`（視 T 大小調整；`q_max ≤ T`），`beta_init=0.5`。
* **效能/記憶體**：時間複雜度約 (O(B N K L T))，通常 `K,L ≪ D,T`，比注意力更省。
* **可解釋監控**：log `info['energy_q']` 並畫柱狀圖；dominant 週期可作為分析訊號的 proxy（例：步態/擺動頻率）。
* **遮罩相容**：`valid_mask` 直接作用於時間維投影前，對可變長 clip 穩定。
* **不等長/非整除 T 的理據**：我們在實作裡做了**行正交化基底**（QR）；對於理論投影到 (S_q) 的更嚴謹敘述，可用文獻中的 **block-mean 投影** 與 (P_q) 重覆拼接的方法，對任意長度成立。

---

## 與相關理論的對齊（供寫論文用）

* **Ramanujan sums 定義**：(c_q(n)=\sum_{(a,q)=1} e^{2\pi i an/q})；我們用其實部（cos 和）作為實值基底，並做 QR 正交化以改善條件數。([維基百科][1])
* **正好週期子空間 (S_q)**：由 (c_q) 生成之循環矩陣的列空間張成，(\dim S_q=\varphi(q))，可由投影矩陣 (P_q) 定義；多個 (S_q) 的結構與分解詳見 RPT/RSP。
* **任意長度投影**：RSP 文獻給出把任意長度訊號分塊取均值再投影的定理（Theorem 1），可作為本模組處理變長序列的理論依據。
* **進一步延伸**：若想更嚴謹地把「跨 q 的正交性」做滿，可引入 **Orthogonal Ramanujan Sums / ORPT** 的建構，於 row-space 做約束。([semanticscholar.org][3])

---

## 小範例（推論階段）

```python
# 假設 vit_out: [B,T,N,D], mask: [B,T,N]
fuser = RPFuse(d_model=D, num_dirs=8, q_max=16)
h, info = fuser(vit_out, valid_mask=mask)
log_energy = info['energy_q']  # 可視化各整數週期能量
```

---

### 與你現有 BDRFuse 的差異點

* DCT 偏頻帶（平滑/趨勢）→ **RPFuse 偏整數週期（事件/節律）**；
* BDRFuse 用 bounded-RMS + 低階 DCT，RPFuse 用 Ramanujan row-orthonormal 基底 + 係數能量；
* 兩者皆是**線性投影 + 殘差門控**，維持訓練穩定；RPFuse 的 `info['energy_q']` 更直接對應「哪個 q 在說話」。

---

如果你要，我可以再幫你把這顆直接**幫你改進現在的檔案**（同檔 `components.py`）並加上簡單的單元測試與可視化腳本，或一起寫一段**論文方法章節**（含定理與 proof sketch）。

[1]: https://en.wikipedia.org/wiki/Ramanujan%27s_sum "Ramanujan's sum"
[2]: https://www.sciencedirect.com/science/article/abs/pii/S0888327016305477 "Ramanujan subspace pursuit for signal periodic ..."
[3]: https://www.semanticscholar.org/paper/Orthogonal-Ramanujan-Sums%2C-Its-Properties%2C-and-in-Yadav-Kuldeep/78c1679133b1b0f20dd9d92fd7c84e3594c3b0dd "[PDF] Orthogonal Ramanujan Sums, Its Properties, and ..."
