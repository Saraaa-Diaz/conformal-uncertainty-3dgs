# 3DGS Rendering + Conformal Prep Scripts

This document explains the current script workflow in the `scripts` folder.

The scripts generate the raw artifacts needed for conformal uncertainty estimation:

- RGB render + GT images
- raw depth/disparity sigma sources as `.npz`
- raw color sigma sources as `.npz`
- deterministic train/calib/test split manifests
- conformal outputs for either color- or depth-based sigma maps

## Current Pipeline

### 1. Round-robin split

We use a deterministic 10-frame cycle to avoid bias on consecutive video frames:

- positions `0..6` in each block of 10: `train`
- position `7`: `calib`
- positions `8..9`: `test`

If the sorted image list is

$$I_0, I_1, I_2, \ldots,$$

then image $I_k$ is assigned by

$$r = k \bmod 10$$

and

$$
\text{split}(I_k)=
\begin{cases}
\text{train} & \text{if } r \in \{0,1,2,3,4,5,6\} \\
\text{calib} & \text{if } r = 7 \\
\text{test} & \text{if } r \in \{8,9\}
\end{cases}
$$

The script [scripts/prepare_round_robin_split.py](/Users/navalbhagat/projects/conformal-uncertainty-3dgs/scripts/prepare_round_robin_split.py) writes:

- `train.txt`
- `calib.txt`
- `test.txt`
- `summary.txt`

These files are passed into the original 3DGS loading code through `--split_dir`.

### 2. Training on the original 3DGS pipeline

The repo was patched so the scene loader can read `--split_dir` from `ModelParams`.

At load time:

- `train.txt` becomes the actual training camera set
- `calib.txt` and `test.txt` are combined into the loader's held-out set

That means `train.py` now does the right thing for this project:

- optimize Gaussians using only the train views
- keep calibration and test views out of optimization

This is implemented in [scene/dataset_readers.py](/Users/navalbhagat/projects/conformal-uncertainty-3dgs/scene/dataset_readers.py), where the explicit split manifests override the repo's usual LLFF holdout logic.

### 3. Rendering held-out views

After training, the new scripts:

- [scripts/render_color.py](/Users/navalbhagat/projects/conformal-uncertainty-3dgs/scripts/render_color.py)
- [scripts/render_depth.py](/Users/navalbhagat/projects/conformal-uncertainty-3dgs/scripts/render_depth.py)

load the trained model, read the same split manifests, and separate the held-out views back into:

- `calib`
- `test`

So the original repo still treats held-out views as one group internally, but the custom renderers write separate calibration and test outputs for later conformal work.

## Output Layout

By default, outputs are written under:

```text
output/<run_name>/<split>/ours_<iteration>/
```

For each split, the scripts write:

```text
gt/<frame>.png
render/<frame>.png
raw_sigma/depth/<frame>.npz
raw_sigma/color/<frame>.npz
preview/depth/<frame>.png
preview/color/<frame>.png
metadata.json
```

Important conventions:

- RGB images are saved as PNG in `[0,255]`
- raw sigma arrays are saved as `.npz`
- raw sigma arrays are not normalized or clipped
- preview PNGs are visualization-only and may be normalized/clipped
- the conformal stage should consume `.npz`, not the preview PNGs

## Script Roles

### `render_common.py`

[scripts/render_common.py](/Users/navalbhagat/projects/conformal-uncertainty-3dgs/scripts/render_common.py) contains the shared plumbing:

- argument parsing
- scene/model loading
- split manifest loading
- view selection
- output directory creation
- RGB saving
- preview normalization
- metadata merging
- train/test exposure safety checks
- `train_test_exp` crop application

The important helper is `select_views(...)`:

- train views come from `scene.getTrainCameras()`
- calib/test views are looked up by image name from `scene.getTestCameras()`

That is how one trained 3DGS model can be rendered separately on calib and test.

### `render_depth.py`

[scripts/render_depth.py](/Users/navalbhagat/projects/conformal-uncertainty-3dgs/scripts/render_depth.py) computes raw depth/disparity uncertainty quantities using alpha-composited scalar feature rendering.

For each Gaussian, let its camera-space depth be

$$z_i.$$

The script renders the following scalar fields with a black background and `override_color`:

- $z$
- $z^2$
- $1/z$
- $(1/z)^2$
- $1$

These give the per-pixel weighted sums:

$$
S_1(\mathbf{p}) = \sum_i w_i(\mathbf{p}) z_i
$$

$$
S_2(\mathbf{p}) = \sum_i w_i(\mathbf{p}) z_i^2
$$

$$
S_{1,\mathrm{inv}}(\mathbf{p}) = \sum_i w_i(\mathbf{p}) \frac{1}{z_i}
$$

$$
S_{2,\mathrm{inv}}(\mathbf{p}) = \sum_i w_i(\mathbf{p}) \frac{1}{z_i^2}
$$

and the accumulated weight map

$$
W(\mathbf{p}) = \sum_i w_i(\mathbf{p}).
$$

With mask

$$
\mathrm{mask}(\mathbf{p}) = W(\mathbf{p}) > \tau
$$

where $\tau$ is `weight_threshold`, the script computes

$$
\mathbb{E}[z](\mathbf{p}) = \frac{S_1(\mathbf{p})}{W(\mathbf{p})}
$$

$$
\mathrm{Var}(z)(\mathbf{p}) = \frac{S_2(\mathbf{p})}{W(\mathbf{p})} - \mathbb{E}[z](\mathbf{p})^2
$$

$$
\mathrm{Std}(z)(\mathbf{p}) = \sqrt{\max(\mathrm{Var}(z)(\mathbf{p}), 0)}
$$

and analogously

$$
\mathbb{E}[1/z](\mathbf{p}) = \frac{S_{1,\mathrm{inv}}(\mathbf{p})}{W(\mathbf{p})}
$$

$$
\mathrm{Var}(1/z)(\mathbf{p}) = \frac{S_{2,\mathrm{inv}}(\mathbf{p})}{W(\mathbf{p})} - \mathbb{E}[1/z](\mathbf{p})^2
$$

$$
\mathrm{Std}(1/z)(\mathbf{p}) = \sqrt{\max(\mathrm{Var}(1/z)(\mathbf{p}), 0)}.
$$

Each frame saves:

- `Ez`
- `depth_var`
- `depth_std`
- `E_inv`
- `invdepth_var`
- `invdepth_std`
- `W`
- `mask`

in one `.npz`.

### `render_color.py`

[scripts/render_color.py](/Users/navalbhagat/projects/conformal-uncertainty-3dgs/scripts/render_color.py) follows the same moment-based idea for color.

First, the script converts SH features to view-dependent RGB:

$$
\mathbf{c}_i = \mathrm{SH2RGB}_\text{view}(i).
$$

For each channel $c \in \{R,G,B\}$ it renders:

- $c_i$
- $c_i^2$
- $1$

which gives

$$
\mu_c(\mathbf{p}) = \frac{\sum_i w_i(\mathbf{p}) c_i}{W(\mathbf{p})}
$$

$$
\mathrm{Var}_c(\mathbf{p}) = \frac{\sum_i w_i(\mathbf{p}) c_i^2}{W(\mathbf{p})} - \mu_c(\mathbf{p})^2
$$

$$
\mathrm{Std}_c(\mathbf{p}) = \sqrt{\max(\mathrm{Var}_c(\mathbf{p}), 0)}.
$$

The script saves:

- `color_mean_rgb`
- `color_var_rgb`
- `color_std_rgb`
- `color_var`
- `color_std`
- `W`
- `mask`

where

$$
\mathrm{color\_var}(\mathbf{p}) = \frac{\mathrm{Var}_R(\mathbf{p}) + \mathrm{Var}_G(\mathbf{p}) + \mathrm{Var}_B(\mathbf{p})}{3}
$$

and

$$
\mathrm{color\_std}(\mathbf{p}) = \sqrt{\mathrm{color\_var}(\mathbf{p})}.
$$

This scalar `color_std` is the preview/default sigma-like quantity for later conformal processing, while the per-channel tensors are preserved in the `.npz` for flexibility.

### `conformal_prediction.py`

[scripts/conformal_prediction.py](/Users/navalbhagat/projects/conformal-uncertainty-3dgs/scripts/conformal_prediction.py) consumes the saved `calib` and `test` outputs and runs conformal prediction for one modality at a time.

Supported modes:

- `--modality color`
- `--modality depth`

Default raw sigma keys:

- `color -> color_std`
- `depth -> invdepth_std`

You can override these with `--sigma_key`.

The script follows the requirements from the notes:

- compute error in RGB space, staying in the saved `[0,255]` range
- sample a fixed percentage of calibration pixels from each calibration image
- use every pixel from every test image
- learn sigma normalization on calibration sigma values and reuse the exact same normalization on test
- save absolute error maps
- save normalized sigma maps
- save conformalized uncertainty maps
- compute mean per-view correlation between absolute error maps and conformalized uncertainty maps

For RGB rendering error:

$$
\mathrm{error}(\mathbf{p}) = \frac{1}{3}\sum_{c \in \{R,G,B\}} |\hat{y}_c(\mathbf{p}) - y_c(\mathbf{p})|
$$

If you want the worst-channel definition instead:

$$
\mathrm{error}(\mathbf{p}) = \max_{c \in \{R,G,B\}} |\hat{y}_c(\mathbf{p}) - y_c(\mathbf{p})|
$$

Use:

- `--rgb_error_mode mean`
- `--rgb_error_mode max`

The nonconformity score is:

$$
s_i = \frac{|\hat{y}_i - y_i|}{\sigma_i}
$$

The corrected conformal quantile is:

$$
n = \mathrm{len}(\mathrm{scores})
$$

$$
\ell = \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}
$$

Then clip $\ell$ to `[0,1]` and compute:

$$
\hat{q}_{1-\alpha} = \mathrm{Quantile}(\mathrm{scores}, \ell, \text{method}=\text{"higher"})
$$

Coverage on test uses the interval half-width:

$$
u(x) = \hat{q}_{1-\alpha}u_{\text{norm}}(x)
$$

with criterion:

$$
|\hat{y}(x) - y(x)| \le u(x)
$$

The saved uncertainty map is the full conformal width:

$$
2u(x) = 2\hat{q}_{1-\alpha}u_{\text{norm}}(x)
$$

By default, `u_norm` is min-max normalized with min/max fit on calibration
pixels only before calibration.

For each frame, the script saves a 6-panel figure in this order:

1. GT
2. render
3. rendered raw sigma
4. absolute error map
5. normalized sigma map
6. uncertainty map after conformal

## How To Run

### Minimal workflow

1. Prepare split manifests
```bash
python3 scripts/prepare_round_robin_split.py \
  --source_path /path/to/scene \
  --images images \
  --output_dir output/my_run/splits
```

2. Train 3DGS on train views only
```bash
python3 train.py \
  -s /path/to/scene \
  -m output/my_run \
  --split_dir output/my_run/splits
```

3. Render held-out calibration and test RGB/sigma artifacts
```bash
python3 scripts/render_color.py \
  -m output/my_run \
  --split_dir output/my_run/splits \
  --skip_train

python3 scripts/render_depth.py \
  -m output/my_run \
  --split_dir output/my_run/splits \
  --skip_train
```

4. Run conformal prediction
```bash
python3 scripts/conformal_prediction.py \
  --run_dir output/my_run \
  --modality color \
  --alpha 0.1 \
  --calib_sample_ratio 0.1

python3 scripts/conformal_prediction.py \
  --run_dir output/my_run \
  --modality color \
  --rgb_error_mode max \
  --alpha 0.1 \
  --calib_sample_ratio 0.1

python3 scripts/conformal_prediction.py \
  --run_dir output/my_run \
  --modality depth \
  --alpha 0.1 \
  --calib_sample_ratio 0.1
```

Using a different depth sigma source:
```bash
python3 scripts/conformal_prediction.py \
  --run_dir output/my_run \
  --modality depth \
  --sigma_key depth_std
```

## Notes And Caveats

- `render.py` from the original repo can also be used with `--split_dir`, but it only knows `train` and `test`, so its `test` output is the union of calib and test.
- The new custom renderers are the ones that re-separate held-out views into `calib` and `test`.
- Raw sigma arrays are saved without normalization or clipping on purpose.
- Preview PNGs are only for inspection.
- The conformal script reuses the exact same epsilon and sigma normalization rules between calibration and test.
