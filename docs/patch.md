下面給你一個遵守「現有 I/O」的 drop-in：**RamaFuse（Ramanujan 序列特徵融合層）**。它不使用 attention／SSM／張量分解／圖方法／低秩核／ToMe/VTM，而是用 **Ramanujan sums** 的「整數週期基底」在時間軸上做分析–合成式的週期投影與殘差融合，專抓「重複節律／週期性動作」訊號。Ramanujan sums (c_q(n)) 來自數論，定義為對所有與 (q) 互質的 (a) 的原始 (q) 次單位根冪次求和；它們天然對「週期 (q)」有選擇性，因此在訊號處理上被用來做 **Ramanujan Periodicity Transform / Filter Banks** 以偵測時變週期結構（例如生醫、語音、腦波）——我們把這套基底變成可微分、端到端的序列融合層即可。 ([維基百科][1])

---

# RamaFuse：Ramanujan Sequence Feature Fusion（可直接替換 StatMem）

**I/O 完全對齊你現有的 `StatMem`**：輸入 `z:[B,T,M,D]`（T 為片段長度、每格 M 個 token、D 通道），可選 `valid_mask:[B,T,M]`；輸出 `h:[B,T,M,D]` 與 `memory_dict`。這與你檔案中 `StatMem` 的介面一致（forward 簽名與張量形狀註解）——你可以在原位置直接替換。  

**作法摘要（無注意力／無SSM）：**

1. **Ramanujan 分析（analysis）**：為一組週期集合 (q=1..Q) 預先產生長度 (W) 的捲積核 (c_q[0..W-1])（Ramanujan sums，零均值、(L_2) 正規化）。對每個 token 的**壓縮表徵**（在 D 維上線性降維或取均值）做 1D 捲積得到「每個時間步的週期響應」 (r_{q,t})。
2. **融合（synthesis）**：用 (r_{q,t}) 維度上的輕量 gating（Sigmoid/Swish + 1×1）產生權重，對**原始通道**做同一組 Ramanujan 核的可分離 1D 捲積並加權求和，得到周期性殘差 (p_t)。
3. **殘差輸出**：(h_t = z_t + \beta \cdot p_t)（(\beta) 可學標量或 per-channel 參數）。`valid_mask` 會保持 padding 位置不變。

> Ramanujan sums（(c_q(n)=\sum_{(a,q)=1} e^{2\pi i an/q})）具備「對應週期 (q) 的選擇性」、與多個 (q) 的（近）正交性；**Ramanujan filter banks (RFB)** 在訊號處理中用這些核掃過時間軸以追蹤局部、時變的週期，這正好符合影片中反覆動作的需求。([維基百科][1])
> 相關還有 **Ramanujan subspace / RSP** 可把序列分解為「精確週期成分」並提供貪婪選擇策略，我們在這裡等價地用固定（可學縮放）基底+可微 gating 來近似。([arXiv][2])

---

## 直接可用的 PyTorch 模組（貼到你的 `components.py`，與 `__all__` 並存）

> 完全不依賴外部套件（僅 `torch`），遵守 `StatMem` 介面；保持你原始檔案的形狀慣例與說明。

```python
# ---------- RamaFuse: Ramanujan Sequence Feature Fusion (drop-in for StatMem) ----------
from math import gcd, pi
import torch
import torch.nn as nn
import torch.nn.functional as F

class RamaFuse(nn.Module):
    """
    Ramanujan-based sequence fusion layer (drop-in for StatMem)
    Input : z:[B,T,M,D], valid_mask:[B,T,M] (optional)
    Output: h:[B,T,M,D], memory_dict (kept for API compatibility)
    No attention / No SSM / No tensor decomposition / No graphs.
    """
    def __init__(self,
                 d_model: int,
                 max_period: int = 16,
                 window: int = 16,
                 proj_dim: int = 0,          # 0 = use channel mean for analysis
                 causal: bool = True,
                 beta_init: float = 0.5,
                 eps: float = 1e-6):
        super().__init__()
        self.Q = int(max_period)
        self.W = int(window)
        self.causal = causal
        self.eps = eps

        # optional low-dim projection for analysis branch
        self.proj_dim = int(proj_dim)
        if self.proj_dim > 0:
            self.analysis_proj = nn.Linear(d_model, self.proj_dim, bias=False)
        else:
            self.analysis_proj = None

        # learnable mixer on Q periodic channels (lightweight, per-token shared across D)
        self.gate = nn.Sequential(
            nn.Conv1d(self.Q, self.Q, kernel_size=1, groups=self.Q, bias=True),  # depthwise 1x1
            nn.GELU(),
            nn.Conv1d(self.Q, self.Q, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # learnable residual scale
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

        # precompute Ramanujan filters: [Q, 1, W]
        self.register_buffer("rama_kernels", self._make_rama_kernels(self.Q, self.W), persistent=False)

        # API-compatible memory dict (not used; kept for drop-in parity)
        self._mem_state = {}

    @staticmethod
    def _ramanujan_sum_vec(q: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # c_q(n) = sum_{1<=a<=q, gcd(a,q)=1} exp(2π i a n / q); use real part (integer-valued)
        n = torch.arange(W, device=device, dtype=dtype)  # 0..W-1
        ks = [a for a in range(1, q + 1) if gcd(a, q) == 1]
        if len(ks) == 0:
            return torch.ones(W, device=device, dtype=dtype)
        angles = torch.outer(torch.tensor(ks, device=device, dtype=dtype), n) * (2.0 * pi / q)
        c = torch.cos(angles).sum(dim=0)  # real part; sin-sum cancels by symmetry
        # zero-mean & l2-normalize (improves stability)
        c = c - c.mean()
        denom = torch.sqrt(torch.clamp(torch.sum(c * c), min=1e-6))
        c = c / denom
        return c

    def _make_rama_kernels(self, Q: int, W: int) -> torch.Tensor:
        # Stack Q periods into a conv bank: [Q, 1, W]
        ker = []
        # q = 1..Q
        for q in range(1, Q + 1):
            ker.append(self._ramanujan_sum_vec(q, W, device=torch.device("cpu"), dtype=torch.float32))
        K = torch.stack(ker, dim=0).unsqueeze(1)  # [Q,1,W]
        return K

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        # Causal "same" length output
        if self.causal and self.W > 1:
            return F.pad(x, (self.W - 1, 0))
        else:
            # symmetric padding to keep length T
            pad = (self.W - 1) // 2
            return F.pad(x, (pad, self.W - 1 - pad))

    def forward(self,
                z: torch.Tensor,                  # [B,T,M,D]
                pos: torch.Tensor = None,
                valid_mask: torch.Tensor = None,  # [B,T,M]
                memory_id: str = None,
                reset_memory: bool = False):
        B, T, M, D = z.shape
        device, dtype = z.device, z.dtype
        if valid_mask is None:
            valid_mask = torch.ones(B, T, M, device=device, dtype=dtype)

        # ---- Analysis branch (compute r_{q,t} per token) ----
        if self.analysis_proj is None:
            z_anl = z.mean(dim=-1)  # [B,T,M]
        else:
            z_anl = torch.einsum('btmd,df->btmf', z, self.analysis_proj.weight.t()).mean(dim=-1)  # [B,T,M]

        z_anl = z_anl * valid_mask  # mask padded tokens
        x = z_anl.permute(0, 2, 1).contiguous()          # [B,M,T]
        x = x.view(B * M, 1, T)                          # [BM,1,T]
        x = self._pad(x)                                  # pad for "same" conv

        rama_k = self.rama_kernels.to(device=device, dtype=dtype)  # [Q,1,W]
        r = F.conv1d(x, rama_k, bias=None, stride=1, padding=0)    # [BM,Q,T]
        r = r.view(B, M, self.Q, T).permute(0, 2, 1, 3).contiguous()  # [B,Q,M,T]

        # gating over Q (per token/time)
        g = self.gate(r.view(B * M, self.Q, T))  # [BM,Q,T]
        g = g.view(B, M, self.Q, T).permute(0, 2, 1, 3).contiguous()  # [B,Q,M,T]

        # ---- Synthesis branch (apply same filters to full D channels) ----
        z_syn = (z * valid_mask.unsqueeze(-1)).permute(0, 2, 3, 1).contiguous()  # [B,M,D,T]
        y = z_syn.view(B * M * D, 1, T)
        y = self._pad(y)
        R = F.conv1d(y, rama_k, bias=None, stride=1, padding=0)  # [B*M*D, Q, T]
        R = R.view(B, M, D, self.Q, T).permute(0, 3, 1, 2, 4).contiguous()  # [B,Q,M,D,T]

        # weighted sum across periods
        p = (R * g.unsqueeze(3)).sum(dim=1)  # [B,M,D,T]
        p = p.permute(0, 3, 1, 2).contiguous()  # [B,T,M,D]

        # residual output + keep padding positions unchanged
        h = z + self.beta * p
        h = valid_mask.unsqueeze(-1) * h + (1.0 - valid_mask.unsqueeze(-1)) * z

        # API-compatible memory dict (no recurrent state by default)
        key = memory_id or "default"
        if reset_memory or (key not in self._mem_state):
            self._mem_state[key] = torch.zeros(B, M, D, device=device, dtype=dtype)
        return h, {key: self._mem_state[key]}
```

**複雜度與特性**

* 時間複雜度 ~ (O(Q \cdot W \cdot B \cdot M \cdot D \cdot T))（1D 可分離卷積），不引入注意力的二次方成本；`Q` 與 `W` 是你可控的超參，通常 (Q,W\le 16) 即可。**Ramanujan filter banks** 的文獻指出其能在短序列中檢出時變局部週期，這正好覆蓋 many human-action 的節律片段。([Eurasip][3])
* 完全**無注意力／無SSM**；僅有固定基底 + 逐點非線性 gating。
* 介面與你現有 `StatMem` 兼容；可直接在 `cls` 前替換。你的檔案已在說明中界定了形狀慣例與 forward 的輸入輸出。 

---

## 參數建議 & 整合步驟

1. **先小步測試**：`RamaFuse(d_model=D, max_period=8, window=8, proj_dim=0, causal=True)`；在 `SimpleFrameTokenSelector` 之後、`cls` 之前替換 `StatMem`。
2. **與 ARP/EMA 對照**：你現有 `StatMem` 是 ARP + EMA（近似 rank pooling + 指數平滑）。先用同樣的訓練設定跑 A/B，比較 Top-1 與吞吐。
3. **調 `Q` / `W`**：若動作更長週期（如 Diving48），把 `max_period`/`window` 增到 16–32。RFB/RPT 文獻顯示更大的 period grid 有助於解析較慢的週期結構。([Eurasip][3])
4. **輕量 gating**：若過擬合，把 `self.gate` 簡化為單層 `Conv1d(Q,Q,1)` + `Sigmoid`。
5. **數值穩定**：Ramanujan 核做了零均值與 (L_2) 正規化；你可視需求把 `beta_init` 調小（0.1）再 warm-up。

---

## 為什麼「Ramanujan」在這裡有效？

* (c_q(n)) 等價於「所有原始 (q) 次頻率的和」，因此對**週期 = q** 的結構有強響應；它還有接近正交的性質、可用來做**週期子空間分解**（RSP/FRSP），被證實能把序列拆成**精確週期成分**。我們把這些基底做卷積掃描 + gating，自然得到**時變週期性**的融合訊號，完全不需要注意力或狀態空間模型。([維基百科][1])

---

如果你想，我可以幫你把 `components.py` 的 `__all__` 加上 `'RamaFuse'`，並替換你現有 `StatMem` 的建構處；或再做一版「非因果（centered）」與「流式因果」比較的 ablation 表。
（背景脈絡與目前管線摘要見你上傳的設計筆記與元件說明。） 

[1]: https://en.wikipedia.org/wiki/Ramanujan%27s_sum "Ramanujan's sum"
[2]: https://arxiv.org/abs/1512.08112 "Ramanujan subspace pursuit for signal periodic decomposition"
[3]: https://www.eurasip.org/Proceedings/Eusipco/Eusipco2015/papers/1570091833.pdf "Properties of Ramanujan filter banks"
