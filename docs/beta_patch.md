簡短答：**不要只看 `beta.mean()`。**
改成「向量 + 門控」後，建議同時看**分佈、有效通道數、實際影響、是否飽和**四件事。下面是最實用的一套指標（拎著就能用）：

## 要看的 4 個指標

1. **分佈形狀**（是不是被少數極端值誤導）

   * `mean`、`median`、`p10/p90`（或畫直方圖）
   * 若 β 允許負值 → 再看 `mean(abs(beta))`，避免正負抵消。

2. **有效通道數（Participation Ratio）**
   衡量「有多少 channel 真正在出力」：
   [
   N_{\text{eff}}=\frac{(\sum_i \beta_i)^2}{\sum_i \beta_i^2}
   ]
   觀察 (N_{\text{eff}}/D)（D=通道數）。太低＝幾乎都關掉；太高＝沒選擇性。

3. **殘差實際影響比**
   [
   r=\frac{|\beta\odot y|}{|x|}
   ]
   看平均或中位數。**1%～5%** 常見且健康；<0.5% 可能太弱；>20% 可能過強。（經驗值，視任務調整）

4. **門控飽和 / 熵**（只對 sigmoid 門控 0～1）

   * 飽和率：`((beta<0.05)|(beta>0.95)).float().mean()`（太高→梯度易消失）
   * 熵：(H=-[b\log b+(1-b)\log(1-b)]) 的平均。低熵＝決策果斷；高熵＝接近 0.5、沒主見。

---

## PyTorch 小抄

**靜態向量門控（β ∈ ℝ^D）**

```python
b = beta.detach()
mean = b.mean().item()
mean_abs = b.abs().mean().item()
p = torch.quantile(b, torch.tensor([0.1, 0.5, 0.9], device=b.device))
sparsity = (b < 0.05).float().mean().item()     # 若 b≥0
neff = (b.sum()**2 / (b.pow(2).sum() + 1e-9)).item()
```

**資料自適應門控（β̂ = sigmoid(g(x))，形狀 [B,T,N,D]）**

```python
gate = beta_hat.detach()
m = gate.mean(dim=(0,1,2))      # 每個通道的平均 [D]
v = gate.var(dim=(0,1,2))
entropy = (-gate*(gate+1e-9).log() - (1-gate)*(1-gate+1e-9).log()).mean().item()
sat = ((gate<0.05)|(gate>0.95)).float().mean().item()
```

**殘差影響比 r**

```python
num = ((beta_hat * y).pow(2).sum(dim=-1).sqrt().mean())
den = (x.pow(2).sum(dim=-1).sqrt().mean() + 1e-9)
r = (num/den).item()
```

---

## 怎麼解讀（快篩準則）

* `beta.mean()` 小但 **`neff/D` 不小**、且 **r 有 1%～5%**、指標有提升 → **健康**（少量但精準的修正）。
* 飽和率超高（例如 >80% 在 0 或 1）或熵極低 → 可能「過度關/過度開」，留意梯度。
* r 幾乎 0 且拿掉殘差表現不變 → 殘差可能沒在工作，需要調尺度或正則。

> 一句話：**看平均值不夠，至少再配 `neff`、`r`、飽和/熵**，你才能知道「有多少通道在出力、出力多大、決策是不是卡死」。
