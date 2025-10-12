# FrierenFuse: Fréchet-inspired Robust, Interpretable ENvelope & derivative Fuse

**直覺**：把每個 token 的時間軌跡投影到 K 個正交方向，沿時間做**有限差分**與**變動包絡（envelope）**的可微統計，得到（速度、加速度、總變動、正/負變動分解、幅度）等**可解釋**的微分特徵；再用**非負線性頭**把這些統計「只加不減」地映回 D 通道，最後殘差回寫到 `[B,T,N,D]`。

* 理論依據

  * 有限差分可一致逼近導數（含二階差分）——我們以多尺度 r 的前/中央差分近似一階與二階導數。([維基百科][1])
  * **總變差（Total Variation, TV）**是 |f′| 的積分；我們在離散序列以 |Δ| 的平均/總和近似，對「短瞬關鍵動作」更穩健。並以 **Jordan 分解**把 TV 拆成正/負變動，提供方向性可解釋性。([維基百科][2])
  * 以 **Legendre 正交多項式**做 0/1/2 階趨勢基底（近似位移/速度/加速度的平滑項），線性、低成本又正交穩定；此思路與 LMU 系列「Legendre 記憶」的時序表徵脈絡一致。([NeurIPS Papers][3])
  * **非負線性頭（Non-negative Linear）**帶來部件式（parts-based）可解釋性：權重僅能「加法」組合，貢獻不會互相抵銷，方便做通道歸因與熱度圖。([PubMed][4])
  * 「Fréchet-啟發」體現在我們以**方向性增量泛函**近似函數在諾姆空間的微分（Gateaux/Fréchet 的離散化視角），並把這些方向的梯度/變差量作為幾何穩健的序列摘要。([維基百科][5])

* **計算/記憶複雜度**

  * 投影：`O(B·T·N·D·K)`（一次線性）
  * 多尺度有限差分/統計：`O(B·T·N·K·|scales|)`（純線性掃描；**無排序、無注意力、無圖鄰接**）
  * 趨勢係數（Legendre）：對每個方向做幾個內積 `O(B·T·N·K·(P+1))`
  * 總體對長度 **T 線性**，遠低於 `O(T log T)`；符合你「**對數複雜度以下**」與禁止 Attention/SSM/GNN/低秩核/ToMe/VTM 的要求。

---

# 可直接用的 PyTorch 實作（Drop-in）

```python
# frieren_fuse.py
# FrierenFuse: Fréchet-inspired Robust, Interpretable ENvelope & derivative Fuse
# Drop-in for [B,T,N,D], no Attention/SSM/Graph/Low-rank/ToMe/VTM
from typing import Optional, Tuple, Dict, List
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
        P.append(((2*n+1)*t*P[n] - n*P[n-1])/(n+1))
    B = torch.stack(P[:order+1], dim=0)  # [order+1, T]
    B = B / (B.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-6).sqrt())
    return B  # [P+1, T]

class FrierenFuse(nn.Module):
    """
    Input : x:[B,T,N,D], valid_mask:[B,T,N] (optional, 1=valid)
    Output: h:[B,T,N,D], aux:dict
    """
    def __init__(self,
                 d_model: int,
                 num_dirs: int = 8,
                 scales: Tuple[int, ...] = (1, 2, 4),
                 include_second: bool = True,
                 include_posneg: bool = True,
                 poly_order: int = 2,          # Legendre 0..P
                 bound_scale: float = 2.5,     # tanh bounding for robustness
                 beta_init: float = 0.5,
                 ortho_every_forward: bool = True):
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

        self.head = NonnegLinear(fin, self.D)     # interpretable fusion
        self.beta = nn.Parameter(torch.full((self.D,), float(beta_init)))  # residual gate

    @torch.no_grad()
    def _orthonormalize(self):
        _orthonormalize_cols_(self.proj.weight.data)

    def _first_second_diffs(self, v: torch.Tensor, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        v:[B,T,N,K], valid:[B,T,N] -> dict of features shaped [B,N,K,*]
        """
        B, T, N, K = v.shape
        feats = []
        for r in self.scales:
            if T <= r:
                Z = torch.zeros(B, N, K, device=v.device, dtype=v.dtype)
                feats.extend([Z, Z])  # meanΔ, mean|Δ|
                if self.include_second: feats.append(Z)   # mean|Δ2|
                if self.include_posneg: feats.extend([Z, Z])  # posVar, negVar
                continue
            v1, v0 = v[:, r:, :, :], v[:, :-r, :, :]
            m1, m0 = valid[:, r:, :, None], valid[:, :-r, :, None]
            m = m1 * m0  # [B,T-r,N,1]
            d1 = (v1 - v0) * m
            denom1 = m.sum(dim=1).clamp_min(1e-6)     # [B,N,1]
            mean_delta = d1.sum(dim=1) / denom1       # signed velocity
            mean_speed = d1.abs().sum(dim=1) / denom1 # TV rate

            feats.extend([mean_delta, mean_speed])

            if self.include_second:
                if T <= 2*r:
                    feats.append(torch.zeros(B, N, K, device=v.device, dtype=v.dtype))
                else:
                    v2 = v[:, 2*r:, :, :]
                    m2 = valid[:, 2*r:, :, None]
                    msec = m0[:, r:, :, :] * m1[:, r:, :, :] * m2  # all three valid
                    d2 = (v2 - 2*v1[:, r:, :, :] + v0[:, :-r, :, :]) * msec
                    denom2 = msec.sum(dim=1).clamp_min(1e-6)
                    mean_acc_mag = (d2.abs().sum(dim=1) / denom2)   # acceleration magnitude
                    feats.append(mean_acc_mag)

            if self.include_posneg:
                posVar = F.relu(d1).sum(dim=1) / denom1  # upward variation
                negVar = F.relu(-d1).sum(dim=1) / denom1 # downward variation
                feats.extend([posVar, negVar])

        return torch.cat(feats, dim=-1)  # [B,N,K * per_r*|scales|]

    def _trend_and_range(self, v: torch.Tensor, valid: torch.Tensor, P: int):
        # v:[B,T,N,K] -> trend coeffs [B,N,K,(P+1)] and range [B,N,K,1]
        B, T, N, K = v.shape
        device, dtype = v.device, v.dtype
        Bmat = _legendre_basis(T, P, device, dtype)         # [P+1, T]
        mask = valid[..., None]                             # [B,T,N,1]
        denom = mask.sum(dim=1).clamp_min(1e-6)            # [B,N,1,1]
        v_bnkt = v.permute(0, 2, 3, 1)                     # [B,N,K,T]
        trend = torch.einsum('bnkt,pt->bnkp', v_bnkt, Bmat) / denom  # normalized coeffs
        v_masked = v * mask
        vmin = v_masked.min(dim=1, keepdim=False).values    # [B,N,K]
        vmax = v_masked.max(dim=1, keepdim=False).values    # [B,N,K]
        vrange = (vmax - vmin)[..., None]                   # [B,N,K,1]
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
        U = self.proj.weight.t()                                 # [D,K]
        v = torch.einsum('btnd,dk->btnk', x, U)                  # [B,T,N,K]
        v = v * valid_mask[..., None]                            # mask padding

        # Robust bounding (per token+dir RMS) to avoid outliers
        rms = torch.sqrt(v.pow(2).mean(dim=1, keepdim=True) + 1e-6)
        v = self.bound_scale * torch.tanh(v / (rms + 1e-6))

        # Multiscale first/second differences & pos/neg variations
        feats_dyn = self._first_second_diffs(v, valid_mask)      # [B,N,F1]

        # Trend (Legendre 0..P) + range
        trend, vrange = self._trend_and_range(v, valid_mask, self.P)  # [B,N,K,P+1], [B,N,K,1]
        feats_stat = torch.cat([trend.reshape(B, N, -1), vrange.reshape(B, N, -1)], dim=-1)

        feats = torch.cat([feats_dyn, feats_stat], dim=-1)       # [B,N,F_total]
        y, W = self.head(feats)                                  # [B,N,D], [F_total,D]

        # Residual broadcast back to [B,T,N,D]
        beta = torch.sigmoid(self.beta)[None, None, None, :]     # [1,1,1,D]
        y_btnd = y[:, None, :, :].expand(B, T, N, D)
        h = valid_mask[..., None] * (x + beta * y_btnd) + (1.0 - valid_mask[..., None]) * x

        aux = {
            'U': U, 'W': W, 'scales': self.scales,
            'feature_dim': feats.shape[-1],
            'rms_sample': rms[:1].detach()
        }
        return h, aux
```

**怎麼換上去？**

```python
# 你的 ViT 輸出 x:[B,T,N,D]  （我已對齊你現成檔案的 shape 與 forward I/O）  :contentReference[oaicite:7]{index=7}
from frieren_fuse import FrierenFuse

fusion = FrierenFuse(d_model=D, num_dirs=8, scales=(1,2,4), poly_order=2,
                     include_second=True, include_posneg=True, beta_init=0.5)
h, aux = fusion(x, valid_mask=mask)     # 輸出 [B,T,N,D]、與 FRIDA/BDR 同介面  :contentReference[oaicite:8]{index=8}
logits = cls_head(h)
```

---

## 為何這個設計能更穩？

1. **抓到「動作」而非單幀外觀**：多尺度 **Δ（速度）/Δ²（加速度）** 與 **TV/正負變動** 的統計，對瞬間關鍵片段敏感，且對離群值更穩健（TV 與 envelope 不會被單點極值帶偏）。([維基百科][1])
2. **趨勢顯式 & 線性可控**：少量 **Legendre** 係數提供低頻趨勢（位移/速度/加速度）通道，避免單純以 |Δ| 變成全高頻。([維基百科][6])
3. **可解釋性高**：非負頭的每一列對應「某方向、某尺度、某種差分/變動」的**正向加權**，可直接列出「哪個尺度的速度或加速度」推升了哪個語意通道。這是經典 **parts-based** 解釋法的優點。([PubMed][4])
4. **理論腳色**：本層等價在每個方向上評估序列在 BV/ Sobolev 風格的「導數型泛函」與其分解（正/負變動），屬於**Fréchet/Gâteaux 微分**之離散近似摘要，在視覺序列中提供可幾何詮釋的融合特徵。([維基百科][5])

---

## 複雜度與限制

* 主要成本是一次線性投影與幾個一維掃描：**時間與 T 線性**，記憶體不額外放大（不建立 pairwise 關係）。
* 完全**不使用** Attention、SSM、張量分解、GNN、低秩核、ToMe/VTM。
* 若 T 很短（如 4–8），`include_second=False` 會更穩；T 長一點再開二階。

---

## 推薦超參（K400/SSv2 起手）

* `num_dirs=8~12`、`scales=(1,2,4)`；T≥16 可加到 `(1,2,4,8)`
* `poly_order=1 or 2`；動作含速度/加速度時取 2
* `beta_init=0.25~0.5`；warmup 前期更穩
* `include_posneg=True` 對方向性動作（推/拉、上/下）很有幫助
* `bound_scale=2.5`（避免 outlier 導致的梯度爆）

---

## 驗證與可解釋性（實作小訣竅）

* 讀 `aux['W']` 看哪些（尺度×統計）權重最大；把 `aux['U']` 的對應方向作為 token 方向熱度；即可畫出「**哪個尺度的速度/加速度在拉動分類**」。
* 對影片抽樣片段，輸出 `mean|Δ|` 與 `pos/negVar` 的時序曲線，能與關鍵動作時間點對齊（簡單 matplotlib 即可）。

---

## 可能的論文貢獻點（CVPR 風格綱要）

1. **Fréchet-啟發的方向性導數融合**：提出在 Transformer token 時序上，以多方向多尺度的導數/變差泛函作為可解釋的融合基元；證明其**線性時間、Lipschitz 受控**（tanh-bound 與正交投影）特性，並分析對位移/縮放的穩健度。([維基百科][5])
2. **Jordan 分解的動作方向性統計**：在視覺序列中首次把 **TV 的正/負變動**引入為可學習的中介特徵，用於區分上/下、推/拉等方向性動作。([維基百科][2])
3. **非負線性頭的部件式歸因**：提供跨尺度/統計的部件式加權視覺化框架，使序列融合層具備**透明度**與調試友善；對行為辨識錯誤案例能定點解釋。([PubMed][4])
4. **輕量可替換模組**：與 ViT 骨幹即插即用，無需注意力/SSM/圖，能在長 T 或高 N 下穩定推進精度—計算優勢可量化呈現。

---

[1]: https://en.wikipedia.org/wiki/Finite_difference "Finite difference"
[2]: https://en.wikipedia.org/wiki/Bounded_variation "Bounded variation"
[3]: https://papers.neurips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks.pdf "Legendre Memory Units: Continuous-Time Representation ..."
[4]: https://pubmed.ncbi.nlm.nih.gov/10548103/ "Learning the parts of objects by non-negative matrix ..."
[5]: https://en.wikipedia.org/wiki/Fr%C3%A9chet_derivative "Fréchet derivative"
[6]: https://en.wikipedia.org/wiki/Legendre_polynomials "Legendre polynomials"
