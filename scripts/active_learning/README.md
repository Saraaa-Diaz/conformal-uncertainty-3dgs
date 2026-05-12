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
random       random candidate views
uniform      evenly spaced candidate views in sorted image order
fisher       PUP/Fisher baseline, using sensitivity_mean
pup          alias for fisher
sensitivity  alias for fisher, useful for signal ablation naming
color        color_mean
visibility   visibility_mean
combined     combined_mean
```

The PUP/Fisher baseline follows the PUP 3D-GS idea of ranking by Fisher
sensitivity over Gaussian xyz+scaling parameters. In this repo the per-view
candidate score is obtained by rendering the per-Gaussian Fisher log-det signal
into candidate views and averaging the resulting sensitivity map.

## Combination Rule

`export_rankings.py` computes per-view signal statistics over valid pixels:

```text
color_mean
sensitivity_mean
visibility_mean
```

For each AL round, it min-max normalizes each available signal across the
candidate views for that round:

```text
norm_signal(view) = (signal(view) - min_signal) / (max_signal - min_signal)
```

If a signal is constant across candidates, its normalized values are set to 0.

The exported combined scores are:

```text
combined_mean(view) = mean(norm_color, norm_sensitivity, norm_visibility)
combined_max(view)  = max(norm_color, norm_sensitivity, norm_visibility)
```

The AL loop's `METHOD=combined` uses `combined_mean`.

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
ITERS=7000
ROUNDS=3
ADD_K=5
METHODS="random fisher color visibility combined"
```

Run a fuller comparison:

```bash
ITERS=30000 ROUNDS=5 ADD_K=10 METHODS="random uniform fisher color visibility combined" ./snellius_jobs/submit_active_learning_tandt_train.sh
```

Run one scene and one method manually:

```bash
SCENE=tandt/train \
AL_ROOT=output/active_learning/tandt_train \
METHOD=combined \
SEED=0 \
ROUNDS=3 \
ADD_K=5 \
ITERS=7000 \
sbatch snellius_jobs/02_active_learning_loop.job
```

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
