"""
3-fold cross-validation of conformal coverage on the 90 held-out frames.

Every held-out frame appears in `test` exactly once across the 3 folds. For
each fold:
  - calib = 30 random frames (drawn from the 60 not-in-test frames)
  - test = 30 frames (the fold's hold-out)

Reports per-fold coverage + calibrated full width + AE correlation, plus the mean
and std across folds. If coverage is honest, std should be small.
"""

import argparse
import math
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, "scripts")
from conformal_random_split import run_one, DEFAULT_SIGMA_KEYS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_folds", type=int, default=3)
    p.add_argument("--calib_n", type=int, default=30)
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    iter_dirs = []
    for split in ("calib", "test"):
        for d in (run_dir / split).iterdir():
            if d.name.startswith("ours_"):
                iter_dirs.append(int(d.name.split("_", 1)[1]))
    iteration = max(iter_dirs)

    # Load pool of all 90 held-out frames.
    from conformal_random_split import load_frame, conformal_quantile, pearson, ause, fit_minmax, normalize_sigma
    pool = []
    for split in ("calib", "test"):
        sd = run_dir / split / f"ours_{iteration}"
        for path in sorted((sd / "render").glob("*.png")):
            pool.append((sd, path.name))
    n_pool = len(pool)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_pool)
    fold_size = n_pool // args.n_folds

    print(f"Pool size: {n_pool}, fold size: {fold_size}, n_folds: {args.n_folds}, seed: {args.seed}")
    print(f"Iteration: {iteration}, alpha: {args.alpha}, target cov: {1 - args.alpha:.3f}")
    print()

    results_per_mod = {mod: [] for mod in DEFAULT_SIGMA_KEYS}
    for fold in range(args.n_folds):
        test_idx = perm[fold * fold_size : (fold + 1) * fold_size].tolist()
        train_idx = [i for i in perm if i not in test_idx]
        calib_idx = train_idx[: args.calib_n]
        # Run each modality on this fold.
        print(f"--- fold {fold + 1}/{args.n_folds} (test_n={len(test_idx)}, calib_n={len(calib_idx)}) ---")
        for mod, key in DEFAULT_SIGMA_KEYS.items():
            calib_pool = [pool[i] for i in calib_idx]
            test_pool = [pool[i] for i in test_idx]
            normalization = fit_minmax(calib_pool, mod, key, 1e-6)

            calib_scores = []
            for sd, name in calib_pool:
                render, gt, sigma, mask = load_frame(sd, mod, name, key)
                err = np.abs(render - gt).mean(axis=2)
                sigma_norm, sigma_safe = normalize_sigma(sigma, mask, normalization, 1e-6)
                n_pix = err.size
                n_samp = max(1, int(math.ceil(0.1 * n_pix)))
                idx = rng.choice(n_pix, size=n_samp, replace=False)
                fe = err.reshape(-1)[idx]
                fs = sigma_safe.reshape(-1)[idx]
                fm = mask.reshape(-1)[idx]
                keep = fm & np.isfinite(fe) & np.isfinite(fs)
                calib_scores.extend((fe[keep] / fs[keep]).tolist())
            q_hat, _ = conformal_quantile(calib_scores, args.alpha)

            covs, widths, corrs, auses = [], [], [], []
            for sd, name in test_pool:
                render, gt, sigma, mask = load_frame(sd, mod, name, key)
                err = np.abs(render - gt).mean(axis=2)
                sigma_norm, _ = normalize_sigma(sigma, mask, normalization, 1e-6)
                u = q_hat * sigma_norm
                full_w = 2.0 * u
                valid = np.isfinite(err) & np.isfinite(full_w) & mask
                covs.append((err[valid] <= u[valid]).mean() if np.any(valid) else 0.0)
                widths.append(full_w[valid].mean() if np.any(valid) else 0.0)
                corrs.append(pearson(err[valid], full_w[valid]))
                auses.append(ause(err, full_w, mask=valid))
            mean_cov = float(np.mean(covs))
            mean_w = float(np.mean(widths))
            mean_corr = float(np.mean(corrs))
            mean_ause = float(np.mean(auses))
            results_per_mod[mod].append((mean_cov, mean_w, mean_corr, mean_ause))
            print(f"  {mod:12} cov={mean_cov:.4f}  full_width={mean_w:8.2f}  corr={mean_corr:+.4f}  ause={mean_ause:.4f}")
        print()

    print("=" * 64)
    print("CROSS-VALIDATED RESULTS (mean ± std over folds)")
    print("=" * 64)
    print(f"{'modality':12} {'cov':>16} {'mean full width':>16} {'AE corr':>14} {'AUSE':>14}")
    for mod, vals in results_per_mod.items():
        c = np.array([v[0] for v in vals]); w = np.array([v[1] for v in vals]); r = np.array([v[2] for v in vals]); a = np.array([v[3] for v in vals])
        print(f"{mod:12} {c.mean():.4f}±{c.std():.4f}  {w.mean():9.2f}±{w.std():6.2f}    {r.mean():+.4f}±{r.std():.4f}    {a.mean():.4f}±{a.std():.4f}")


if __name__ == "__main__":
    main()
