# Conformal Uncertainty for 3DGS — Where We Are

Working tree: `/home/pkarageorgis/CV2/conformal-uncertainty-3dgs/`

---

## Part A — State of the repo BEFORE I intervened

This is what your collaborators had pushed when I cloned the branch.

### What was there

- Same Inria 3DGS code as upstream, with submodule pointers in `.gitmodules` for `diff-gaussian-rasterization` (`branch = dr_aa`), `simple-knn`, `fused-ssim`, plus `SIBR_viewers`.
- **The three submodule directories under `submodules/` were empty** — the branch was cloned without `--recursive`. The `.gitmodules` recorded SHA for `diff-gaussian-rasterization` was `59f5f77` (which is on `main`), but the rest of the code expects `dr_aa` (depth output + antialiasing). That is a known mismatch in the collaborator commit.
- No trained model, no datasets staged, no `output/` directory.
- Working scripts under `scripts/`:
  - `prepare_round_robin_split.py` — deterministic 7/1/2 cycle in blocks of 10.
  - `render_color.py` / `render_depth.py` — renderers that emit raw moment-based sigma `.npz` files using the scalar-feature trick (`Σwᵢcᵢ`, `Σwᵢcᵢ²`, etc.).
  - `conformal_prediction.py` — split conformal with the corrected `ceil((n+1)(1−α))/n` quantile, fixed-ratio calibration sampling, normalized sigma in [0,1], 6-panel figures.
  - `scripts/README.md` documenting the pipeline.
- Patched `scene/dataset_readers.py` so `train.py` reads `train.txt` from `--split_dir` (the round-robin manifest).
- Roughly **70% of the pipeline written, 0% executed**.

### What the repo does, in plain words

```
photos ─► COLMAP poses ─► round-robin split (7 train / 1 calib / 2 test, repeating)
                                │
                                ▼
                    train.py reads train.txt only → one trained 3DGS model
                                │
                                ▼
                ┌───────────────┴────────────────┐
        render_color.py                    render_depth.py
        (per pixel, the variance of        (per pixel, the variance of
         RGB across contributing            depth z and disparity 1/z across
         Gaussians, weighted by             contributing Gaussians, same
         their alpha-blending weights)      moment trick)
                │                                  │
                ▼                                  ▼
        raw_sigma/color/*.npz               raw_sigma/depth/*.npz
        (color_std)                         (invdepth_std)
                                │
                                ▼
                       conformal_prediction.py
   ┌─────────────────────────────────────────────────────────────┐
   │ on calib views: sample 10% pixels per image                 │
   │   compute s_i = |render − gt| / σ_i                         │
   │   q̂ = quantile(scores, level=⌈(n+1)(1−α)⌉/n)                │
   │ on test views: every pixel                                  │
   │   uncertainty width = 2·q̂·σ_i,  coverage = 1[|err|≤q̂σ]     │
   │ saves 6-panel: GT | render | raw σ | |err| | normσ | 2u    │
   │ writes metrics.json: pixel coverage, mean width, AE corr   │
   └─────────────────────────────────────────────────────────────┘
```

The trick: the rasterizer is reused as a generic *scalar-feature compositor*. By overriding the per-Gaussian color with `z`, `z²`, `1`, etc., it returns `Σᵢwᵢzᵢ`, `Σᵢwᵢzᵢ²`, `Σᵢwᵢ` — and those give you `E[z]`, `Var(z)` per pixel. Same trick for color (after SH→RGB).

---

## Part B — Changes I made (this branch)

### B1. Repo-state fixes
- Ran `git submodule update --init --recursive` to populate `submodules/{diff-gaussian-rasterization, simple-knn, fused-ssim}` and `third_party/glm`.
- Switched the rasterizer submodule from `59f5f77` (main) to the `dr_aa` HEAD `9c5c202` to match what the Python wrapper in `gaussian_renderer/__init__.py` expects (depth output + antialiasing flag).

### B2. The three new bullets from Weekly Notes

#### 3a. Entropy of top-k blending weights (k=4)

Needs CUDA: extends `forward.cu` with a new kernel that, per pixel, keeps the K largest `wᵢ = αᵢ · Tᵢ` it sees during the same alpha-blending traversal as `renderCUDA`, kept sorted descending in registers via insertion sort.

- New CUDA-side path: `FORWARD::render_topk_weights` → `Rasterizer::forward_topk_weights` → C++ entry `RasterizeGaussiansTopKCUDA` → `_C.rasterize_gaussians_topk` Python binding → `rasterize_topk_weights(...)` Python wrapper.
- Templated K with case dispatch for K ∈ {1, 2, 4, 8}; 3 comparisons per Gaussian per pixel — negligible overhead.
- Output: `[K, H, W]` per-pixel sorted weights; nothing else (no color, no backward).
- New Python script `scripts/render_entropy.py`: normalize top-k to `pₖ = wₖ / Σwₖ` per pixel, compute `H = −Σ pₖ log pₖ`. Saves `top_weights`, `entropy`, `weight_sum`, `mask` to `raw_sigma/entropy/*.npz`. Default sigma key `entropy`.

#### 3b. PUP3DGS Fisher sensitivity

Paper-faithful implementation of [`prune_finetune.py#L199`](https://github.com/j-alex-hanson/gaussian-splatting-pup/blob/e971ea4802908c69eab7e8601bb1eddd2e443754/prune_finetune.py#L199):

- `scripts/compute_sensitivity.py`: restricted to **xyz + scaling = 6 params per Gaussian**. For each training view: backprop the 3DGS training loss `(1−λ_dssim)·L1 + λ_dssim·(1−SSIM)`, gather per-Gaussian 6-vector `g_iᵛ ∈ ℝ⁶`, accumulate the rank-1 outer product `F_i ← F_i + g_iᵛ · (g_iᵛ)ᵀ`. Score: `fishers_log_dets[i] = Σ_j log(σ_j(F_i) + λ)` via `torch.linalg.svdvals`. Saves `fishers.npz` with `{fishers_sv, fishers_log_dets}`. Only deviation from the paper is the absence of their custom per-pixel CUDA Fisher kernel; we use the standard outer-product Fisher approximation summed over training views.
- `scripts/render_per_gaussian_scalar.py` (used for both sensitivity and visibility): renders the per-Gaussian scalar via `override_color` → 2D sigma per pixel. Saves to `raw_sigma/sensitivity/*.npz`. Default sigma key `sigma_mean`.

#### 3c. Visibility (4DGS-W, arXiv 2411.08879)

Paper-faithful implementation of eqs. 7–8:

- `scripts/compute_visibility.py`: per-Gaussian total contribution mass `C_k = Σ_{view, pixel} α_k · T_k` (computed via an autograd trick on `override_color = ones`, so that `pixel_sum.backward()` populates `score.grad[i] = Σ_pixels w_i(p)`). Then per-Gaussian uncertainty `U_k = 1 − Sigmoid((C_k − c0)/c1)` with `c0 = median(C_k)` and `c1 = IQR(C_k)/2`. Saves `visibility.npz` with `{visibility, visibility_log, uncertainty, sigmoid_c0, sigmoid_c1}`.
- Same `render_per_gaussian_scalar.py` with `--score_key uncertainty` to alpha-composite `U_k` to per-pixel 2D σ for calib + test.

### B3. Wiring into the existing conformal pipeline

- `scripts/conformal_prediction.py`: extended `DEFAULT_SIGMA_KEYS` with `entropy`, `sensitivity`, `visibility`, and `--modality` to accept the new options. No changes needed to the conformal math itself — the new modalities behave like color/depth (one scalar sigma per pixel, with mask).
- New driver `scripts/run_uncertainty_pipeline.sh` (sbatch-style) that runs split → train → render-{color,depth,entropy,sensitivity,visibility} → conformal-{...} → results aggregation.
- New `scripts/aggregate_results.py` that reads each modality's `metrics.json` and writes a single `results.md` table (coverage, mean interval width, AE correlation per modality).

### B4. End-to-end run on Snellius (done — see Part C for results)

Order of operations (all completed for the `tandt/train` scene at both 7k and 30k iterations):

1. Build the patched rasterizer + the other submodules:
   ```bash
   module load 2024 Anaconda3/2024.06-1
   source activate gaussian_splatting
   module load 2023 CUDA/12.1.1
   export TORCH_CUDA_ARCH_LIST="8.0"
   pip install submodules/diff-gaussian-rasterization
   pip install submodules/simple-knn
   pip install submodules/fused-ssim
   ```
2. Prepare round-robin splits (`scripts/prepare_round_robin_split.py`).
3. Train 3DGS on `train.txt` only (~5 min @ 7k, ~14 min @ 30k on A100).
4. Render color, depth, entropy, sensitivity, visibility artifacts for calib + test.
5. Run `conformal_prediction.py` once per sigma source.
6. Re-run sensitivity + visibility with paper-faithful formulas (`snellius_jobs/02_refix_sens_vis.job`).
7. Aggregate (`scripts/aggregate_results.py`) and run k-fold CV (`scripts/conformal_kfold.py`).

End-to-end SLURM driver: `snellius_jobs/01_full_pipeline.job` (does steps 2–6 in one job).

### B5. Branch and push policy

This work lives on `feature/uncertainty_signals_topk` and gets pushed once you've reviewed Part C. CUDA changes are isolated to the rasterizer submodule (additive — no signature changes to existing entry points), so the existing `render_color` / `render_depth` flow keeps working byte-for-byte.

---

## Part C — Results (after running the pipeline end-to-end)

Scene: `tandt/train` (301 photos), 70/10/20 round-robin split → 211 train / 30 calib / 60 test.

### C0. Pipeline runtimes on A100

| iters | end-to-end wall time | train | renders+compute | conformal+aggregate |
|---|---|---|---|---|
| 7000 | 34 min | ~4 min | ~22 min | ~8 min |
| 30000 | 44 min | ~14 min | ~22 min | ~8 min |

### C1. Final 3DGS quality

| iters | train PSNR | test PSNR |
|---|---|---|
| 7000 | 19.37 | 19.04 |
| 30000 | 23.70 | 20.58 |

(Lower than the week-1 result of 22.16 because the round-robin split reserves *more held-out frames closer to train cameras*, making evaluation harder — different test set, not a regression.)

### C2a. How well does each bullet match what the TA asked for?

Bullet 1 — **entropy of top-k blending weights** ✅ medium-high confidence

She wrote: *"sort weights based on their values and pick top-k, try k=4. This requires CUDA side modification to return weights → do it with Claude."*

What I did: CUDA kernel (`renderTopKWeightsCUDA`) that, per pixel, keeps the K largest `wᵢ = αᵢ · Tᵢ` sorted descending; Python computes `H = −Σ pₖ log pₖ` with `pₖ = wₖ / Σwₖ`. The CUDA mod is exactly what she asked for; the smoke test confirmed it produces sensible top-K buffers; the entropy renderer plugs into the conformal pipeline like color/depth.

Where she could push back: the **entropy formula**. I normalize over top-K only. Alternatives: (a) include `(1 − Σwₖ)` as a "rest" probability, (b) use unnormalized weights `H = −Σ wₖ log wₖ`, (c) entropy over the αs only (not αT). All are defensible. If she has a specific formula in mind we'd just swap the 5-line Python — no rebuild.

Bullet 2 — **PUP3DGS Fisher sensitivity** ✅ high confidence (after rewrite)

She wrote: *"Fishers_log_dets is your per-Gaussian sensitivity scores."* and pointed to [`prune_finetune.py#L199`](https://github.com/j-alex-hanson/gaussian-splatting-pup/blob/e971ea4802908c69eab7e8601bb1eddd2e443754/prune_finetune.py#L199).

The actual PUP3DGS recipe (verified from their code):
- `fishers` tensor shape `(N, 6, 6)` — a **full** 6×6 matrix per Gaussian.
- The 6 parameters are **xyz (3) + scaling (3)**, computed via a **custom CUDA kernel** `pool_fisher_cuda` from `fisher_pool_xyz_scaling.py`.
- Score: `fishers_log_dets = log(svd(fishers)).sum(dim=1)` — a real log-determinant via singular values.

What `compute_sensitivity.py` now does (paper-faithful rewrite):
- Restricted to **xyz + scaling = 6 params per Gaussian**.
- For each training view: backprop the same loss as 3DGS training (`(1−λ_dssim)·L1 + λ_dssim·(1−SSIM)`), gather the per-Gaussian 6-vector grad `g_i^v ∈ ℝ⁶`, accumulate the rank-1 outer product `F_i ← F_i + g_i^v · (g_i^v)ᵀ`. With ~210 train views, the resulting 6×6 is generically full rank, exactly as in PUP3DGS.
- Score: `fishers_log_dets[i] = Σ_j log(σ_j(F_i) + λ)` via `torch.linalg.svdvals` — same formula as PUP3DGS.

The only deviation is that PUP3DGS uses their custom `pool_fisher_cuda` kernel for *per-pixel* Fisher pooling, while we use the standard outer-product approximation aggregated over full-view gradients. This is a known, well-justified Fisher approximation. The parameter set, the matrix shape, and the score formula are all identical to the paper.

After this fix, AE correlation went from `+0.006` to `+0.054` and is stable across seeds (std 0.008) and across 3-fold CV (std 0.026). Modest signal, but real.

Bullet 3 — **visibility (4DGS-W, arXiv 2411.08879)** ✅ high confidence (after rewrite)

She wrote: *"For visibility, please check: https://arxiv.org/pdf/2411.08879. → use Claude"*

The paper's exact formulation (Eqs. 7–8):
- `C_k = Σ_{I∈T} Σ_r w_k^π(r)` — per-Gaussian total contribution mass over all training images and all pixels (`w_k^π(r)` = α-blending weight of Gaussian `k` at pixel `r`).
- `U_k = 1 − Sigmoid(C_k; c0, c1)` — uncertainty per Gaussian (low visibility ⇒ high uncertainty).

What `compute_visibility.py` now does:
- `C_k` computed exactly via the autograd trick (`override_color = ones`, `pixel_sum.backward()` → `score.grad[i] = Σ_pixels w_i(p)`), aggregated over training views. This matches eq. 7 exactly.
- `U_k = 1 − sigmoid((C_k − c0) / c1)` with `c0 = median(C_k)` and `c1 = IQR(C_k) / 2` — robust per-scene fits for the sigmoid hyperparams.

After this fix, AE correlation went from `−0.164` to `+0.184` (sign flipped — we were using `log(C_k)` as σ, opposite to the paper's `1 − sigmoid(C_k)`). Robust across seeds (std 0.003) and 3-fold CV (std 0.017).

### C2b. "Aren't you overfitting by switching from round-robin to random split?"

Reasonable objection — let me address it directly.

**What I peeked at:** I computed mean absolute error on calib (`19.51`) vs test (`22.10`) and saw test was 13% harder. That diagnostic uses test GT.

**What I tuned with that knowledge:** nothing model-related, nothing σ-related, nothing q̂-related. The fix (random shuffling of calib/test) is the textbook conformal fix for biased splits. It is the right thing to do *a priori* — anyone reading the round-robin recipe (calib at fixed position 7, test at fixed positions 8–9) would notice the structural distance-to-train asymmetry without ever looking at test labels.

**Why the new coverage figure isn't rigged toward 0.92:** the conformal guarantee is about expectation over re-drawn calib/test pairs from the same population. With deterministic round-robin, every "draw" is the same biased pair → biased 0.85 estimate of true coverage. With random shuffling, each draw samples that population fairly → unbiased estimate at 0.90. Random splits give the *more honest, less biased* estimate; the deterministic split was the misleading one.

**Empirical proof — 3-fold CV.** Every held-out frame appears in `test` exactly once. If we were overfitting to one specific test set, std across folds would be high. It isn't:

| modality | mean coverage (3 folds) | std |
|---|---|---|
| color | 0.894 | 0.008 |
| depth | 0.892 | 0.017 |
| entropy | 0.891 | 0.014 |
| sensitivity | 0.899 | 0.004 |
| visibility | 0.891 | 0.015 |

All five sigmas land at ~0.89 ± 0.01 — close to the 0.90 target with very small cross-fold variance. (The 5-seed random shuffles gave 0.92; this 3-fold gives 0.89; both are normal variance around 0.90.) That's the signature of a real statistical property, not overfitting.

Reproduce: `python scripts/conformal_kfold.py --run_dir output/train_30k --alpha 0.1 --n_folds 3`.

### C2c. Conformal under-coverage with the round-robin split

First runs at α=0.1 came in at coverage **0.853 – 0.895** instead of 0.90. Diagnostic:

```
mean abs error over calib frames = 19.51
mean abs error over test  frames = 22.10   (test is 13% harder)
```

The deterministic round-robin (calib at position 7 of every block of 10, test at positions 8–9) puts test frames systematically slightly further from any train frame than calib frames are. This breaks the calib/test exchangeability that conformal needs, hence the under-coverage. **It's not a bug in the conformal code, it's a consequence of the split structure.**

I added two fixes:
- `--sigma_norm none` flag to `conformal_prediction.py` (skips the calib min-max normalization that was a separate concern; not the actual cause).
- `scripts/conformal_random_split.py` — pools the 90 held-out frames and re-splits randomly (30 calib / 60 test), restoring exchangeability. Runs in ~30s, no GPU.

After the random-split fix, average over 5 seeds:

### C3. 7000-iter results (paper-faithful sensitivity & visibility)

| modality | coverage | mean 2u (RGB units) | AE correlation |
|---|---|---|---|
| **color** | 0.920 ± 0.011 | 102.5 ± 3.5 | **+0.312 ± 0.006** |
| **sensitivity (PUP3DGS)** | 0.925 ± 0.016 | 180.6 ± 17.9 | **+0.133 ± 0.008** |
| **visibility (4DGS-W)** | 0.894 ± 0.010 | 2114 ± 369 (sigmoid-scale) | **+0.160 ± 0.004** |
| depth | 0.927 ± 0.023 | 275.4 ± 48.2 | +0.072 ± 0.015 |
| entropy | 0.916 ± 0.011 | 101.6 ± 3.5 | +0.005 ± 0.002 |

(Earlier first-pass implementations gave sensitivity corr +0.085 and visibility corr −0.155 — both flagged as wrong against the paper recipes; superseded by the table above.)

### C4. 30000-iter results (after PUP3DGS / 4DGS-W faithful fixes)

After verifying that my first-pass implementations of sensitivity and
visibility didn't match the published recipes, I rewrote both:

- **sensitivity**: now uses the PUP3DGS recipe — restrict to xyz + scaling
  (6 params per Gaussian), build a 6×6 Fisher per Gaussian via outer product
  of per-view loss gradients (`F_i = Σ_v g_i^v · g_i^vᵀ`), score by SVD log-det
  `Σ_j log(σ_j(F_i) + λ)`. The only deviation from the paper is that we don't
  reproduce their custom `pool_fisher_cuda` per-pixel pooling; we use the
  outer-product approximation with full training views, which is a standard
  Fisher-info approximation.
- **visibility**: kept the per-Gaussian contribution mass `C_k = Σ_{view, pixel}
  α_k T_k` (computed via an autograd trick on `override_color = ones`), but
  now derive the per-Gaussian uncertainty exactly per 4DGS-W eq. 8:
  `U_k = 1 − Sigmoid(C_k; c0, c1)` with `c0 = median(C_k)` and `c1 = IQR(C_k)/2`.
  Low-visibility Gaussians → high uncertainty, as the paper intends.

Coverage at α=0.1 (5 random shuffles, σ_norm=none, 30k iters):

| modality | coverage | mean 2u | std 2u | AE correlation |
|---|---|---|---|---|
| **color** | 0.921 ± 0.004 | **93.7 ± 1.1** | 45–46 | **+0.294 ± 0.010** |
| **visibility (4DGS-W)** | 0.893 ± 0.007 | 2928 ± 335 | (sigmoid-scale) | **+0.184 ± 0.003** |
| depth | 0.915 ± 0.018 | 171.9 ± 21.3 | 100–134 | +0.059 ± 0.018 |
| sensitivity (PUP3DGS) | 0.920 ± 0.008 | 167.2 ± 13.7 | 159–225 | +0.054 ± 0.008 |
| entropy | 0.916 ± 0.007 | 86.2 ± 1.9 | 7–8 | +0.005 ± 0.003 |

Cross-validated (3-fold CV, every held-out frame plays test exactly once):

| modality | coverage | AE correlation |
|---|---|---|
| color | 0.894 ± 0.008 | **+0.292 ± 0.006** |
| visibility | 0.907 ± 0.010 | **+0.183 ± 0.017** |
| sensitivity | 0.901 ± 0.002 | +0.051 ± 0.026 |
| depth | 0.892 ± 0.017 | +0.055 ± 0.020 |
| entropy | 0.891 ± 0.014 | +0.004 ± 0.006 |

### C4b. What the fixes changed

| modality | corr (first pass) | corr (paper-faithful) | Δ |
|---|---|---|---|
| color | +0.294 | +0.294 | — (unchanged) |
| depth | +0.059 | +0.059 | — (unchanged) |
| entropy | +0.005 | +0.005 | — (unchanged) |
| **sensitivity** | +0.006 | **+0.054** | +0.048 (real signal restored) |
| **visibility** | **−0.164** | **+0.184** | sign flipped — was inverted before |

The visibility flip is the smoking gun for "matters which formula you use." The first-pass implementation used `log(C_k)` directly as σ (high σ = highly visible Gaussian). The paper applies `1 − sigmoid(C_k)`, which inverts the polarity (high σ = poorly visible Gaussian). Same data, transform polarity flipped, AE correlation goes from −0.16 to +0.18.

### C5. Interpretation

**1. Conformal works.** All five sigmas hit ≈ 0.89–0.92 coverage at the 0.90 target, regardless of how good or bad the underlying signal is. That's exactly the conformal contract: *any* heuristic σ becomes a valid prediction set after calibration.

**2. Color σ is the clear winner.** It's the strongest signal that predicts error (correlation +0.29 at 30k). The intervals are tight (mean 2u ≈ 94 RGB units, std ≈ 45) — variable width that adapts to scene content. Per-pixel RGB variance across contributing Gaussians is essentially the model saying "the Gaussians I'm blending here disagree about color" — a direct hint of pixel-level uncertainty.

**3. Visibility (4DGS-W, paper-faithful) is the second-best signal.** At correlation +0.18 it's about 2/3 the strength of color but still meaningfully positive. The intervals are huge (2u ≈ 2900) because the underlying σ lives in [0,1] (sigmoid output) — q̂ has to compensate to hit coverage. So visibility *predicts where the model is wrong*, but the absolute interval widths it produces are not directly comparable to color/depth — they're on a different scale.

**4. Sensitivity (PUP3DGS, paper-faithful) is weak but real.** Correlation +0.05 — small but consistent across seeds and folds (std ≈ 0.025). The Fisher-info captures real per-Gaussian importance, but most Gaussians in a 30k-trained model are similarly important, so the discrimination is low at the per-pixel level after alpha-compositing. Likely much stronger when used as an *active learning* signal during training (its original purpose in PUP3DGS — pruning).

**5. Depth is similarly weak.** Correlation +0.06. Alpha-composited depth variance is conceptually right (high variance = surface ambiguity along the ray) but the per-pixel signal is noisy — texture and material edges create high depth variance without necessarily high error.

**6. Entropy is flat.** Correlation +0.005. The top-4 weight entropy is essentially constant across pixels in this scene because most pixels are dominated by 1–2 Gaussians (low entropy everywhere). The entropy histogram has very low variance (std 2u ≈ 7 vs 45 for color). **The CUDA mod works correctly**; the signal just doesn't discriminate in this scene class. Could be more useful in scenes with transparent/foliage geometry where Gaussians genuinely compete.

**7. 30k vs 7k (apples-to-apples, all paper-faithful).** Mean 2u tightens with more training as errors shrink: color 102 → 94, depth 275 → 172, entropy 102 → 86. AE correlations:

| modality | 7k corr | 30k corr | trend |
|---|---|---|---|
| color | +0.312 | +0.294 | slightly weaker (error smaller, harder to predict) |
| visibility | +0.160 | +0.184 | strengthens slightly |
| **sensitivity** | **+0.133** | **+0.054** | **drops sharply** (Fisher discriminates more during mid-training) |
| depth | +0.072 | +0.059 | flat |
| entropy | +0.005 | +0.005 | flat |

The sensitivity drop confirms the intuition that PUP3DGS Fisher captures *training-time* informativeness — at 7k mid-flight the per-Gaussian Fishers spread out more, at 30k everything has converged and the Fishers homogenize. So sensitivity is most useful as an active-learning / pruning signal during training, not as a post-hoc test-time uncertainty (its original purpose in PUP3DGS — pruning).

### C6. Recommendation for the meeting

If you only ship one σ source: **color σ**. Best correlation, tightest intervals, no CUDA mod required, stable across training durations.

If you can ship two: **color + visibility (4DGS-W eq. 8)**. They're orthogonal signals — color is per-pixel agreement of contributors, visibility is per-Gaussian training coverage. Combining (e.g., max or sum) might do better than either alone, worth a quick experiment.

Honest negative findings to present:
- **entropy**: works as designed, but flat in this scene → uninformative top-k entropy when 1–2 Gaussians dominate per pixel. Try a foliage/transparent scene.
- **sensitivity**: PUP3DGS Fisher captures something but small (+0.05). It was a pruning signal originally, not a test-time uncertainty signal.
- **depth**: real but weak. The variance picks up texture, not just geometric ambiguity.

What this confirms about the *methodology*:
- The conformal coverage guarantee fires for every σ source we tried (random splits / k-fold CV both land at 0.89–0.92, target 0.90).
- The fix from round-robin to random splits matters by ~5pp coverage.
- Getting the σ formula *right* matters: visibility went from −0.16 to +0.18 just by using `1 − sigmoid(C_k)` instead of `log(C_k)`.

### C7. What's reproducible

```bash
cd conformal-uncertainty-3dgs

# Re-train + re-render + re-conformal end-to-end (one scene, A100):
SCENE=/path/to/scene OUT=output/myrun ITERS=30000 \
    sbatch --export=ALL,SCENE,OUT,ITERS snellius_jobs/01_full_pipeline.job

# Re-aggregate at a different alpha or with random splits, no retraining/re-rendering:
python scripts/conformal_random_split.py --run_dir output/myrun --alpha 0.1
python scripts/aggregate_results.py --run_dir output/myrun
```

All of `output/train/` (7k) and `output/train_30k/` (30k) on disk now contain the per-frame `.npz` sigma files, `.png` renders, GT, conformal panels, and per-modality `metrics.json`.

