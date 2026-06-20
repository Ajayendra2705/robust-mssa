# SSA & MSSA — Mechanics Note (Day 2)

> Working note grounding the implementation. Notation follows Golyandina/Rodrigues conventions.
> Anything marked **[verify]** should be cross-checked against the cited paper before it
> enters the technical report.

---

## 1. Univariate SSA

Let `F = (f_0, f_1, …, f_{N-1})` be a real-valued series of length `N`.
Choose a **window length** `L` with `1 < L < N`, and set `K = N − L + 1`.

### Stage 1 — Embedding
Form the **trajectory matrix** by sliding a length-`L` window across the series:

```
        ⎡ f_0    f_1    …  f_{K-1} ⎤
        ⎢ f_1    f_2    …  f_{K}   ⎥
  X  =  ⎢  ⋮      ⋮          ⋮     ⎥   ∈ R^{L×K}
        ⎣ f_{L-1} f_L   …  f_{N-1} ⎦
```

Each column `X_i = (f_{i-1}, …, f_{i+L-2})ᵀ` is a lagged vector. `X` is a **Hankel matrix**:
its entries are constant along anti-diagonals, i.e. `X[i, j]` depends only on `i + j`
(`X[i, j] = f_{i+j}` with 0-based `i, j`). This Hankel structure is the property our
embedding code must satisfy exactly, and is what diagonal averaging will later restore.

### Stage 2 — Decomposition (SVD)
Let `S = X Xᵀ ∈ R^{L×L}`. Let `λ_1 ≥ λ_2 ≥ … ≥ λ_L ≥ 0` be its eigenvalues and
`U_1, …, U_L` the orthonormal eigenvectors. With `d = rank(X) = #{ i : λ_i > 0 }`, define
`V_i = Xᵀ U_i / √λ_i`. Then the SVD is

```
  X = Σ_{i=1}^{d} √λ_i · U_i V_iᵀ = Σ_{i=1}^{d} X_i ,     X_i := √λ_i U_i V_iᵀ.
```

- `(√λ_i, U_i, V_i)` is the `i`-th **eigentriple**.
- `√λ_i` is the `i`-th singular value; `U_i` are **left singular vectors** (the empirical
  orthonormal basis / "EOFs"), `V_i` the **right singular vectors** (principal components /
  factor scores).
- Each `X_i` is a rank-1 **elementary matrix**. Note `‖X‖_F² = Σ λ_i`, so `λ_i / Σλ_j` is
  the share of variance captured by component `i` (the scree plot).

This is the step Robust MSSA replaces: standard SVD solves
`min_{rank(M) ≤ r} ‖X − M‖_F²` (Eckart–Young), an **L2** objective that a few outlying
columns can dominate.

### Stage 3 — Grouping
Partition the index set `{1, …, d}` into `m` disjoint groups `I_1, …, I_m` and sum the
elementary matrices in each group:

```
  X = Σ_{k=1}^{m} X_{I_k} ,     X_{I_k} = Σ_{i ∈ I_k} X_i.
```

Grouping is where interpretation happens: indices are assigned to *trend*, *oscillatory
pairs* (seasonality usually appears as a pair of eigentriples with similar singular values
and phase-shifted sinusoidal singular vectors), and *noise*.

### Stage 4 — Reconstruction (diagonal averaging / Hankelization)
A grouped matrix `Y = X_{I_k}` is generally **not** Hankel. Diagonal averaging projects it
back to a Hankel matrix (equivalently, a series) by averaging each anti-diagonal. For a
matrix `Y ∈ R^{L×K}` with `L* = min(L,K)`, `K* = max(L,K)`, the reconstructed series
`g = (g_0, …, g_{N-1})` is, writing `Y*` for `Y` if `L<K` else `Yᵀ`:

```
            ⎧ (1/(s+1)) Σ_{l=0}^{s}      Y*[l, s−l]              0 ≤ s < L*−1
  g_s  =    ⎨ (1/L*)    Σ_{l=0}^{L*−1}   Y*[l, s−l]              L*−1 ≤ s < K*
            ⎩ (1/(N−s)) Σ_{l=s−K*+1}^{N−K*} Y*[l, s−l]          K* ≤ s < N
```

i.e. each value of the output series is the **mean of the corresponding anti-diagonal** of
the matrix. Applied to `X` itself it is the identity; applied to each `X_{I_k}` it yields
the reconstructed component series, and `Σ_k g^{(k)} = F` exactly (full grouping).
**This is the reconstruction identity the Day-6 tests will assert.**

### Separability & the w-correlation
Two reconstructed series `g^{(1)}, g^{(2)}` are **(weakly) separable** if their inner product
under the weights `w = (w_0, …, w_{N-1})`, `w_s = #{(i,j): i+j = s}` (the anti-diagonal
length, `w_s = min(s+1, L*, N−s)`), is zero. The **weighted correlation**

```
  ρ_w(g^{(1)}, g^{(2)}) = ⟨g^{(1)}, g^{(2)}⟩_w / (‖g^{(1)}‖_w · ‖g^{(2)}‖_w)
```

near 0 ⇒ well separated; large `|ρ_w|` ⇒ the two components should likely be in the same
group. The `m × m` w-correlation matrix is the primary separability diagnostic (Day 9).

### Choice of `L`
- For a series with a periodicity `p`, choosing `L` a multiple of `p` improves separation.
- `L ≈ N/2` maximises resolution but raises cost; a common range is `L ∈ [N/4, N/2]`.
- Trade-off: larger `L` ⇒ finer frequency resolution but fewer columns `K` (noisier
  estimate of the column space). To be revisited empirically (Day 21).

---

## 2. Multivariate SSA (MSSA)

Now `p` series `F^{(1)}, …, F^{(p)}`. We use the **horizontal / "stacked" (common-window)**
form, which matches the block trajectory matrix `H ∈ R^{L×K_tot}` in the proposal.

Assume (for now) equal lengths `N` and a common window `L`, so each channel has the same
`K = N − L + 1`. Build each channel's trajectory matrix `X^{(j)} ∈ R^{L×K}` exactly as in §1,
then **concatenate columns**:

```
  H = [ X^{(1)} | X^{(2)} | … | X^{(p)} ]  ∈ R^{L × K_tot},   K_tot = p·K.
```

- **Embedding:** block-Hankel — each block `X^{(j)}` is individually Hankel.
- **Decomposition:** one SVD of the `L × K_tot` matrix `H = Σ √λ_i U_i V_iᵀ`. The left
  singular vectors `U_i ∈ R^L` are **shared across channels** — this is exactly how MSSA
  exploits cross-sectional co-movement: a single common temporal basis is fit jointly to all
  series. The right singular vector `V_i ∈ R^{K_tot}` splits into `p` channel blocks.
- **Grouping:** as in §1, on the joint eigentriples.
- **Reconstruction:** split the grouped matrix `H_{I_k}` back into its `p` column blocks of
  width `K`, and diagonal-average **each block separately** to recover that channel's
  reconstructed component.

Why MSSA over `p` independent SSAs: the shared basis `U_i` pools information across series
that move together, so common structure (a market factor, a business-cycle factor) is
estimated from `p·K` columns instead of `K`, improving signal recovery when series share
latent drivers — the precise claim quantified in **Rodrigues & Mahmoudvand (2018)** **[verify
exact conditions]**.

### Vertical MSSA (alternative, not our default)
Stacking *rows* gives `(pL) × K`. This treats the channels as extra rows rather than extra
columns and changes which subspace is shared. The proposal's `H ∈ R^{L×K_tot}` is the
horizontal form, so that is our baseline; vertical MSSA is noted only as a possible ablation.

### Unequal lengths
If series have different lengths `N_j`, the `K_j = N_j − L + 1` differ and the blocks simply
have different widths; `K_tot = Σ_j K_j`. Missing data is a separate concern (spectral
imputation, Rodrigues & de Carvalho 2013) handled in Phase 3.

---

## 3. Implementation contract (what the code must guarantee)

1. **Hankel structure:** `embed(F, L)[i, j]` depends only on `i + j`; equals `f_{i+j}`.
2. **Block structure:** `mssa_embed([F1,…,Fp], L)` returns `H` of shape `(L, p·K)` whose
   `j`-th width-`K` block equals `embed(Fj, L)`.
3. **Decomposition fidelity:** `Σ_i √λ_i U_i V_iᵀ = H` to machine precision (Day-5 test).
4. **Reconstruction identity:** diagonal-averaging the full (all-component) grouped matrix
   returns the original series; `Σ_k g^{(k)} = F` (Day-6 test).
5. **Backend interchangeability:** swapping `StandardSVD → RobustSVD` changes only the
   `decompose` call; embedding/grouping/reconstruction code is untouched.

---

## References
- Golyandina, Nekrutkin & Zhigljavsky (2001), *Analysis of Time Series Structure: SSA and
  Related Techniques* — canonical reference for the four-stage scheme and diagonal averaging.
- Rodrigues, P.C. & Mahmoudvand, R. (2018). *The benefits of MSSA over the univariate
  version.* J. Franklin Institute.
- Golyandina & Zhigljavsky (2013), *Singular Spectrum Analysis for Time Series* — w-correlation
  and separability.
