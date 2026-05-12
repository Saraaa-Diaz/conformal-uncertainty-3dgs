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
ITERS=7000
ROUNDS=3
ADD_K=5
METHODS="conformal_color conformal_visibility conformal_sensitivity raw_sensitivity"
```

Run a fuller comparison:

```bash
ITERS=30000 ROUNDS=5 ADD_K=10 METHODS="conformal_color conformal_visibility conformal_sensitivity raw_sensitivity" ./snellius_jobs/submit_active_learning_tandt_train.sh
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
