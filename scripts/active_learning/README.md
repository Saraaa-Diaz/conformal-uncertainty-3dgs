# Active Learning Pipeline

This folder contains the active-learning utilities. They are separate from the
basic uncertainty pipeline so AL runs can write to `output/active_learning/...`
without overwriting standard runs.

## Split Layout

Each AL round has:

```text
splits/train.txt       currently labeled/training views
splits/calib.txt       fixed conformal calibration views
splits/test.txt        fixed held-out evaluation views
splits/candidate.txt   unlabeled/selectable candidate views
```

Round 0 is initialized by `splits.py init`:

```text
INIT_TRAIN -> train.txt
CALIB      -> calib.txt
TEST       -> test.txt
remaining  -> candidate.txt
```

The default launcher uses:

```text
INIT_TRAIN=10%
CALIB=10%
TEST=20%
candidate=remaining 60%
```

For later rounds, `calib.txt` and `test.txt` stay fixed. The selected top `K`
candidate views are appended to `train.txt` and removed from `candidate.txt`.

## Acquisition Methods

`splits.py update --method ...` supports:

```text
random                  random candidate views
uniform                 evenly spaced candidate views in sorted image order
conformal_color         color mean conformal interval width
conformal_visibility    visibility mean conformal interval width
conformal_sensitivity   sensitivity mean conformal interval width
conformal_combined      min-max normalized conformal color/visibility/sensitivity average
raw_sensitivity         raw rendered Fisher/sensitivity mean
raw_color               raw rendered color uncertainty mean
raw_visibility          raw rendered visibility uncertainty mean
raw_combined            min-max normalized raw color/visibility/sensitivity average
fisher, pup             aliases for raw_sensitivity
color, visibility       aliases for conformal_color/conformal_visibility
sensitivity             alias for conformal_sensitivity
combined                alias for conformal_combined
```

The raw sensitivity baseline is PUP-style Fisher acquisition: it ranks by the
mean rendered Fisher/sensitivity map. The conformal sensitivity method instead
uses calibration views to estimate `q_hat`, then ranks candidate views by mean
`2*q_hat*sigma`.

## Combination Rule

`export_rankings.py` computes both raw and conformal per-view statistics over
valid candidate pixels:

```text
color_raw_mean
sensitivity_raw_mean
visibility_raw_mean
color_conformal_mean_2u
sensitivity_conformal_mean_2u
visibility_conformal_mean_2u
```

Conformal scores use calibration views only:

```text
q_hat = conformal quantile(|render - gt| / sigma on calib)
candidate_score = mean(2 * q_hat * sigma_candidate)
```

Candidate GT is not used for acquisition.

For combined methods, each available signal is min-max normalized across the
candidate views for that round:

```text
norm_signal(view) = (signal(view) - min_signal) / (max_signal - min_signal)
```

If a signal is constant across candidates, its normalized values are set to 0.

The exported combined scores are:

```text
raw_combined_mean(view) = mean(norm_raw_color, norm_raw_sensitivity, norm_raw_visibility)
conformal_combined_mean(view) = mean(norm_conf_color, norm_conf_sensitivity, norm_conf_visibility)
```

The AL loop's `METHOD=combined` is an alias for `conformal_combined`.

## Running On Snellius

Run default AL jobs for one scene:

```bash
./snellius_jobs/submit_active_learning_db_drjohnson.sh
./snellius_jobs/submit_active_learning_db_playroom.sh
./snellius_jobs/submit_active_learning_tandt_train.sh
./snellius_jobs/submit_active_learning_tandt_truck.sh
```

Defaults:

```text
ITERS=30000
ROUNDS=5
ADD_K=5
METHODS="conformal_color conformal_visibility conformal_sensitivity raw_sensitivity"
```

Run one scene and one method manually:

```bash
SCENE=tandt/train \
AL_ROOT=output/active_learning/tandt_train \
METHOD=conformal_color \
SEED=0 \
ROUNDS=3 \
ADD_K=5 \
ITERS=7000 \
sbatch snellius_jobs/02_active_learning_loop.job
```

Each method renders only the signal it needs. For example,
`conformal_color` renders color only, `raw_sensitivity` renders Fisher
sensitivity only, and `conformal_visibility` renders visibility only. Depth and
entropy are not rendered in the active-learning loop unless a future method is
added that uses them.

## Outputs

Per method/seed/round:

```text
output/active_learning/<scene>/<method>/seed_<seed>/round_<rr>/
  splits/
  results.md
  results.json
  conformal/
  active_learning/ours_<iter>/
    view_signal_scores.csv
    view_rankings.json
```

Scene-level summaries:

```text
output/active_learning/<scene>/active_learning_summary.csv
output/active_learning/<scene>/active_learning_summary.md
output/active_learning/<scene>/figures/
```

The figures include PSNR/SSIM/LPIPS learning curves and per-modality
coverage/interval-width/AE-correlation/AUSE curves against number of training
views.

## Results for Playroom

| method | seed | round | train views | PSNR | SSIM | LPIPS | modality | coverage | mean 2u | AE corr | AUSE |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| conformal_color | 0 | 0 | 22 | 19.141 | 0.7706 | 0.3945 | color | 0.9431 | 111.80 | 0.4357 | 0.2931 |
| conformal_color | 0 | 1 | 27 | 20.768 | 0.7981 | 0.3692 | color | 0.9442 | 90.05 | 0.4438 | 0.2948 |
| conformal_color | 0 | 2 | 32 | 21.454 | 0.8094 | 0.3560 | color | 0.9366 | 75.19 | 0.4635 | 0.2790 |
| conformal_color | 0 | 3 | 37 | 22.948 | 0.8267 | 0.3354 | color | 0.9176 | 52.38 | 0.4687 | 0.2663 |
| conformal_color | 0 | 4 | 42 | 23.687 | 0.8400 | 0.3191 | color | 0.9214 | 47.87 | 0.4810 | 0.2447 |
| conformal_color | 0 | 5 | 47 | 25.013 | 0.8520 | 0.3066 | color | 0.9208 | 42.16 | 0.4754 | 0.2553 |
| conformal_sensitivity | 0 | 0 | 22 | 19.547 | 0.7734 | 0.3966 | sensitivity | 0.9388 | 385.94 | 0.0222 | 0.6218 |
| conformal_sensitivity | 0 | 1 | 27 | 20.478 | 0.7956 | 0.3729 | sensitivity | 0.8932 | 284.52 | 0.0005 | 0.6866 |
| conformal_sensitivity | 0 | 2 | 32 | 21.459 | 0.8075 | 0.3565 | sensitivity | 0.9365 | 579.00 | -0.0001 | 0.7314 |
| conformal_sensitivity | 0 | 3 | 37 | 22.757 | 0.8237 | 0.3375 | sensitivity | 0.9404 | 339.76 | 0.0289 | 0.6533 |
| conformal_sensitivity | 0 | 4 | 42 | 23.618 | 0.8394 | 0.3207 | sensitivity | 0.9175 | 191.63 | -0.0015 | 0.6284 |
| conformal_sensitivity | 0 | 5 | 47 | 24.634 | 0.8507 | 0.3061 | sensitivity | 0.9366 | 265.01 | -0.0103 | 0.6423 |
| conformal_visibility | 0 | 0 | 22 | 19.239 | 0.7722 | 0.3942 | visibility | 0.8906 | 5951.72 | 0.1927 | 0.4051 |
| conformal_visibility | 0 | 1 | 27 | 20.540 | 0.7971 | 0.3693 | visibility | 0.8978 | 5218.68 | 0.2338 | 0.3485 |
| conformal_visibility | 0 | 2 | 32 | 21.551 | 0.8105 | 0.3536 | visibility | 0.8997 | 8357.56 | 0.2667 | 0.3452 |
| conformal_visibility | 0 | 3 | 37 | 22.532 | 0.8224 | 0.3378 | visibility | 0.9131 | 17341.60 | 0.2423 | 0.3926 |
| conformal_visibility | 0 | 4 | 42 | 23.461 | 0.8354 | 0.3224 | visibility | 0.9192 | 14733.40 | 0.2669 | 0.4049 |
| conformal_visibility | 0 | 5 | 47 | 24.730 | 0.8497 | 0.3074 | visibility | 0.9006 | 12641.77 | 0.2848 | 0.3294 |
| raw_sensitivity | 0 | 0 | 22 | 19.440 | 0.7720 | 0.3971 | sensitivity | 0.9434 | 975.68 | 0.0095 | 0.7194 |
| raw_sensitivity | 0 | 1 | 27 | 20.397 | 0.7957 | 0.3711 | sensitivity | 0.9339 | 392.78 | 0.0244 | 0.6748 |
| raw_sensitivity | 0 | 2 | 32 | 21.428 | 0.8090 | 0.3576 | sensitivity | 0.9325 | 424.34 | 0.0179 | 0.6201 |
| raw_sensitivity | 0 | 3 | 37 | 22.867 | 0.8265 | 0.3354 | sensitivity | 0.9260 | 283.76 | 0.0245 | 0.6670 |
| raw_sensitivity | 0 | 4 | 42 | 23.592 | 0.8363 | 0.3214 | sensitivity | 0.9189 | 203.96 | -0.0052 | 0.6407 |
| raw_sensitivity | 0 | 5 | 47 | 24.850 | 0.8511 | 0.3064 | sensitivity | 0.9235 | 282.94 | 0.0002 | 0.6468 |
