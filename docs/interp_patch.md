# 極簡方案：**BDRF**（Bounded-DCT Residual Fusion）

**想法一句話**：
不做分位、不做軟排序、不做加權最小平方法；只做

1. 投影；2) 以**tanh** 做**有界化**（抑制外點）；3) 低階 **DCT** 係數當「趨勢特徵」；4) 兩個**有界矩**（均值、能量）；5) **單層非負線性頭** + β 殘差閘門。
   — 全部都是 O(B·T·N·K) 的向量化內積與均值，**沒有任何會出 NaN 的除法/反矩陣**。
   （DCT 以固定餘弦基近似趨勢，低頻係數可解釋成 level/斜率/緩慢起伏；有界化等價於穩健估計的「Huber 化精神」，但更便宜。([維基百科][1])）

---

## 可直接貼上的 PyTorch 模組（取代你現有那顆）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def _dct2_basis(T: int, P: int, device, dtype):
    # DCT-II（未做嚴格正交化，作為固定低頻基底已足夠穩）
    t = torch.arange(T, device=device, dtype=dtype) + 0.5  # [T]
    B = [torch.ones(T, device=device, dtype=dtype)]  # p=0 常數項
    for p in range(1, P + 1):
        B.append(torch.cos(torch.pi * p * t / T))
    B = torch.stack(B, dim=0)  # [P+1, T]
    # 粗略歸一化，避免係數規模爆掉
    B = B / (B.square().sum(dim=1, keepdim=True).sqrt() + 1e-6)
    return B  # [P+1, T]

class NonnegLinear(nn.Module):
    # 單層非負線性頭：可加、可解釋、最簡
    def __init__(self, fin, fout):
        super().__init__()
        self.W_raw = nn.Parameter(torch.empty(fin, fout))
        nn.init.xavier_uniform_(self.W_raw)
    def forward(self, x):
        W = F.softplus(self.W_raw)  # 保證非負，避免相互抵銷
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
        P: int = 2,                    # 取 2~3 即可，表示抓低頻趨勢
        beta_init: float = 0.5,
        ortho_every_forward: bool = True,
        bound_scale: float = 2.5       # tanh 有界尺度
    ):
        super().__init__()
        self.d_model = d_model
        self.K = int(num_dirs)
        self.P = int(P)
        self.ortho_every_forward = bool(ortho_every_forward)
        self.bound_scale = float(bound_scale)

        self.proj = nn.Linear(d_model, self.K, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=5 ** 0.5)

        # features = K * [ DCT係數(P+1) + 2個有界矩(mean, rms) ]
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

        # 1) 投影
        U = self.proj.weight.t()                      # [D,K]
        v = torch.einsum('btnd,dk->btnk', x, U)       # [B,T,N,K]
        v = v * valid_mask[..., None]
        v_bnkt = v.permute(0, 2, 3, 1).contiguous()   # [B,N,K,T]

        # 2) 有界化（抑制外點；全部有界 => 幾乎不可能 NaN）
        #    先以 RMS 做尺度，再用 tanh 限幅
        rms = torch.sqrt(v_bnkt.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        v_bounded = self.bound_scale * torch.tanh(v_bnkt / (rms + 1e-6))  # [B,N,K,T]

        # 3) 低階 DCT 係數作為趨勢特徵
        Bmat = _dct2_basis(T, self.P, device, dtype)                       # [P+1, T]
        coeff = torch.einsum('bnkt,pt->bnkp', v_bounded, Bmat)             # [B,N,K,P+1]

        # 4) 兩個有界矩：mean, rms（都在有界域內計算，超穩）
        mean_feat = v_bounded.mean(dim=-1, keepdim=True)                   # [B,N,K,1]
        rms_feat  = torch.sqrt(v_bounded.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

        feats = torch.cat([coeff, mean_feat, rms_feat], dim=-1).reshape(B, N, -1)  # [B,N,F]
        y, W = self.head(feats)                                            # [B,N,D], [F,D]

        # 5) 殘差融合（向量 gate）
        beta = torch.sigmoid(self.beta)[None, None, None, :]               # [1,1,1,D]
        y_btnd = y[:, None, :, :].expand(B, T, N, D)
        h = x + beta * y_btnd
        h = valid_mask[..., None] * h + (1.0 - valid_mask[..., None]) * x
        return h, {'W': W}
```

**解釋為何它穩又快**

* **沒有排序/分位/SoftSort** → 沒有 (O(T\log T)) 與梯度不穩問題。
* **沒有解線性方程或矩陣反轉** → 只做與固定餘弦基的內積。
* **tanh 有界化** 等價於「Huber 化」的極簡版本：小殘差近似線性，大殘差被平滑截斷，實務上就夠穩健（Huber 精神）。([維基百科][2])
* **低階 DCT** 天生可解釋成低頻趨勢能量與相位，早就在時序/訊號界證明能以少量係數近似趨勢。([維基百科][1])

**可解釋性**

* 每個方向 (k) 的 DCT 係數=「該方向的趨勢分量」，mean/rms=「該方向整體水平與能量」。非負線性頭讓各統計對通道 (d) 的貢獻是**疊加且單調**，畫熱圖超直覺。
* 若想更「論文味」，在附錄談「投影追蹤/切片逆迴歸的脈絡 + 固定頻域基底作低維摘要」，審稿人很買單。([泰晤士健康][3])

---

## 你剛剛 NaN 的主因（快速定位）

1. `robust_legendre_coeff` 裡的 `torch.inverse` + 權重近奇異，最容易把梯度炸掉。
2. `skew = ... / (var + 1e-6).pow(1.5)` 在半常數段仍可能極小→放大噪聲；再經 head 放大後連鎖 NaN。
3. AMP/bfloat16 情境下，softmax/除法/平方和等若沒良好縮放也會 NaN（先關 AMP 訓練，穩後再開）。([PyTorch Forums][4])

---

## 如果你**一定**想修舊版（但我建議直接換 BDRF）

* 把所有 `/sum` 換成 `softmax` 正規化（直方圖層尤其安全）。([arXiv][5])
* 所有矩陣別 `inverse`，一律 `torch.linalg.solve` 並加 `+eps*I`。
* `var` 相關的分母一律 `clamp_min(1e-6)`；`median/quantile` 統計改用有界 `tanh` 前置處理（先降尺度再算）。
* 混合精度先停用、或採動態 loss scaling；必要時 clip 梯度。([PyTorch Forums][4])

---

## 為何這樣有 **CVPR 可用的新穎點**

* 我們把**投影式魯棒融合**改為**「有界化 + 固定頻域基底 + 非負線性可加頭」**的極簡流水線：避開可微排序與穩健回歸求解的老問題，卻保留統計可解釋性（方向 × 低頻係數/均值/能量）。
* 文獻上 DCT 與魯棒估計是經典，但**這種無排序、無求解、全有界的殘差融合**在時序特徵融合領域是乾淨、直接的組合，可抓住「效率 vs. 可解釋」這個賣點。([維基百科][1])

---

## 三個超短訓練守則（防 NaN + 快速收斂）

1. **先 FP32、後 AMP**；若要 AMP，選 bf16 + 動態 loss scaling；保持 logit 尺度合理。([Medium][6])
2. **梯度裁剪**（如 1.0）+ **權重衰退**（1e-4）+ **warmup** 幾百 step（可用 cosine）。
3. **監控髒樣本**（啟用 `terminate_on_nan` 或 batch 檢查，必要時跳過）。([GitHub][7])

---

如果要，我可以把 BDRF 幫你接到你現有訓練腳本、補上 `r`/`r_feat` 的貢獻統計與可視化（direction×feature 的熱圖），再給一份簡短 ablation（DCT 階數/有界尺度/是否非負頭）。你先把上面的 `BDRFuse` 丟進去跑一版，看看 acc/loss 是否立刻回正。

[1]: https://en.wikipedia.org/wiki/Discrete_cosine_transform "Discrete cosine transform"
[2]: https://en.wikipedia.org/wiki/Huber_loss "Huber loss"
[3]: https://www.tandfonline.com/doi/abs/10.1080/01621459.1991.10475035 "Sliced Inverse Regression for Dimension Reduction"
[4]: https://discuss.pytorch.org/t/nan-loss-issues-with-precision-16-in-pytorch-lightning-gan-training/204369 "NaN Loss Issues with Precision 16 in PyTorch Lightning ..."
[5]: https://arxiv.org/abs/2005.03995 "Differentiable Joint and Color Histogram Layers for Image- ..."
[6]: https://medium.com/%40Modexa/7-pytorch-mixed-precision-rules-that-avoid-nans-3d1c7dfaa4f5 "7 PyTorch Mixed-Precision Rules That Avoid NaNs"
[7]: https://github.com/Lightning-AI/pytorch-lightning/issues/4956 "how to properly skip samples that cause inf/nan gradients ..."
