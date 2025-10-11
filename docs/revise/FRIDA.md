# FRIDA (**Fréchet-Inspired Difference Aggregator**, pronounced "Frida")

> Core idea: project each token's time series onto K orthogonal directions, then extract two sets of features — signed change and magnitude — by taking finite differences (Δ_r) at multiple scales (r). Finally, fuse them with a non-negative linear head (to ensure additivity and interpretability) as a residual.

---

## Why this design is sound (and lightweight)

* **Theoretical motivation**: The Fréchet derivative linearizes "changes of a function in vector spaces." In practice we approximate it with direction derivatives (Gâteaux) combined with finite differences, which can be seen as a JVP estimate of linear functionals. ([Wikipedia][1])
* **Interpretability**: Difference templates ([1,-1], [1,0,-1], ...) are zero-sum filters; the detail filter of the Haar wavelet is essentially a normalized first difference, so each scale r's Δ_r is the change across time step r. ([staff.washington.edu][2])
* **Parts-based weights**: The final layer uses non-negative weights so contributions from each scale/direction are additive and interpretable (parts-based), aligning with classic NMF interpretability motivations. ([PubMed][3])
* **Computational complexity**: For a fixed scale set (𝓡) — usually 3–4 scales — and K directions, time complexity is O(B T N K |𝓡|) (linear in T). If you want a denser orthogonal basis, replace the differences with a Walsh–Hadamard basis and accelerate with FWHT to get O(T log T). ([Wikipedia][4])
* **No restricted components**: The whole pipeline uses only matrix multiplies, elementwise ops, and shifted differences; there is no attention, state-space models, graph structure, low-rank kernels, ToMe/VTM, etc.

---

## Algorithm (outline)

1. **Directional projection**: Learn U ∈ R^{D×K} and QR-orthogonalize it each forward pass, producing v = ⟨x, U⟩ ∈ R^{B×T×N×K}.

2. **Stabilization**: RMS-normalize each token+direction and bound with tanh to be robust to outliers.

3. **Multi-scale difference features**: For each r ∈ 𝓡:

    * **Signed change**: μ_r = mean * t, [v_t - v_{t-r}]
    * **Magnitude**: RMS_r or ℓ1 (optional)

    These are approximations of directional derivatives for several zero-sum linear functionals, equivalent to computing JVPs with difference filters. ([odlgroup.github.io][5])

4. **Non-negative linear head + residual fusion**: Concatenate the above features, pass through NonnegLinear(W ≥ 0), then gate per channel (σ(β)) and add back into x.

---

## A ready-to-use PyTorch module (Drop-in)

I generated a file you can import directly; I/O matches your existing BDRFuse (`forward(x: [B,T,N,D], valid_mask)` returns `(h, aux)`):

Usage:

```python
from frida import FRIDA

fuse = FRIDA(d_model=D, num_dirs=8, scales=(1,2,4), use_rms=True)
h, aux = fuse(x, valid_mask)   # x: [B,T,N,D]
```

---

## Differences and complements with BDRFuse

* **Feature type**: BDRFuse leans toward "frequency/trend" (low-order DCT + moment boundaries), while FRIDA focuses on "change/roughness spectrum" (multi-scale differences). The two can be used for ablations or even chained.
* **Interpretability**: Each scale r and direction k in FRIDA corresponds to the r-step change along U_k, which is intuitive (e.g., whether a key joint moves sharply within 2–4 frames). The Haar = difference connection gives a clear signal-processing interpretation. ([staff.washington.edu][2])

---

## Training notes (lightweight & stable)

* **Direction orthogonalization**: QR each forward pass prevents direction collapse and improves interpretability.
* **Sparsification**: Apply ℓ1 regularization to NonnegLinear weights to encourage only a few scales/directions to be active.
* **Scale selection**: Recommend 𝓡 = {1,2,4} or {1,3,6}; add 8 if T is very long.
* **Gate initialization**: β ≈ 0.5 (set by default) helps early stability.
* **Experiment suggestions**: Run ablations for (i) signed-change only, (ii) magnitude only, (iii) both combined; sweep number of directions K ∈ {4,8,12} and number of scales ∈ {2,3,4}.

---

## Pronounceable acronym shortlist (pick your favorite)

* **FRIDA** — Fréchet-Inspired Difference Aggregator (recommended)
* **FROG** — Fréchet-Residual Orthogonal Gradients (memorable)
* **FADO** — Fréchet Approximate Difference Operator

---

### References (key supports)

* Fréchet derivative: its generalized definition and intuition. ([Wikipedia][1])
* Gâteaux/directional derivatives and finite-difference approximations — implementation link. ([odlgroup.github.io][5])
* Haar wavelet detail filter ≈ normalized first difference (motivation for multi-scale differencing). ([staff.washington.edu][2])
* Fast Walsh–Hadamard transform is O(n log n) (for extending the difference family to an orthogonal sequence). ([Wikipedia][4])
* Non-negative weights provide parts-based interpretability (classic NMF evidence). ([PubMed][3])

---

[1]: https://en.wikipedia.org/wiki/Fr%C3%A9chet_derivative "Fréchet derivative"
[2]: https://staff.washington.edu/dbp/PDFFILES/3-Lund-A4.pdf "Part III: Basic Theory for Discrete Wavelet Transform (DWT)"
[3]: https://pubmed.ncbi.nlm.nih.gov/10548103/ "Learning the parts of objects by non-negative matrix ..."
[4]: https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform "Fast Walsh–Hadamard transform"
[5]: https://odlgroup.github.io/odl/math/derivatives_guide.html "On the different notions of derivative — odl 0.8.1 documentation"
