# Conformal Uncertainty Quantification for 3D Gaussian Splatting

This document explains the two uncertainty quantification scripts and the conformal prediction framework they implement.

## What is 'Error'?

The **error** used in both scripts is the **per-pixel absolute difference** between the rendered image and ground truth:

$$\text{error}_{i} = \text{mean}_{\text{channels}} \left| \text{render}_{i} - \text{gt}_{i} \right|$$

For RGB images, we average across color channels. For grayscale, it's the simple absolute difference. Pixel values are normalized to [0, 1].

---

## Conformal Prediction Framework

Both scripts implement **distribution-free conformal prediction** for uncertainty quantification:

### Core Idea

Given a rendering error and an uncertainty estimate σ at each pixel, we construct a prediction interval:
$$[\text{error} - q \cdot \sigma, \text{error} + q \cdot \sigma]$$

where $q$ is a learned quantile threshold that guarantees coverage at level $1 - \alpha$.

### Two-Phase Approach

**Phase 1: Calibration** (on calibration set)
- Compute non-conformity scores: $s_i = \frac{|\text{error}_i|}{\sigma_i}$
- Compute threshold: $q = \text{quantile}_{1-\alpha}(s_1, \ldots, s_n)$

**Phase 2: Prediction** (on test set)
- Prediction interval width: $w = q \cdot \sigma$
- Check coverage: does $|\text{error}| \leq w$?
- Guaranteed: ≥ $(1-\alpha)$ of test pixels satisfy coverage

---

## Script 1: `uncertainty_maps.py`

Estimates uncertainty from **per-image error statistics**.

### Workflow

```
1. Load Test Images
   ├─ Read rendered images from renders/ directory
   └─ Read ground truth images from gt/ directory
   
2. Split into Calibration & Test
   └─ First calib_ratio fraction → calibration set
   └─ Remaining → test set
   
3. Calibration Phase
   ├─ For each calibration image:
   │  ├─ Compute per-pixel errors
   │  ├─ Estimate σ = std(errors) across all pixels
   │  ├─ Sample num_pixels random pixel locations
   │  └─ Compute non-conformity scores: s = error / σ
   ├─ Aggregate scores from all images
   └─ Compute threshold: q = quantile_{1-α}(s)
   
4. Test Phase
   ├─ For each test image:
   │  ├─ Compute error map
   │  ├─ Apply conformal prediction interval: error ≤ q · σ
   │  └─ Track coverage
   └─ Compute average coverage rate
   
5. Visualization & Metrics
   └─ Save 5-column visualization per test image
```

### Mathematical Details

#### Sigma Estimation

For each calibration image:
$$\sigma_{\text{img}} = \text{std}\left(\left\{\text{error}_{i,j} : (i,j) \in \text{image}\right\}\right)$$

This is a **single scalar per image**, not per-pixel.

#### Non-conformity Score

For sampled pixels in calibration images:
$$s = \frac{\text{error}}{\sigma_{\text{img}}}$$

#### Prediction Interval

For test pixels, using the learned threshold $q$:
$$w = q \cdot \sigma_{\text{img}}$$

Coverage: $\text{error} \leq w$

---


## Script 2: TODO

Extracts uncertainty estimates from **3D Gaussian point cloud properties**.

### Workflow

```
1. Load 3D Gaussian Point Cloud
   └─ Read point_cloud.ply from 3DGS training
   
2. Extract Gaussian Properties
   ├─ Positions: xyz coordinates
   ├─ Opacities: σ_a (from log-space)
   └─ Scales: σ_xyz (from log-space)
   
3. Parse Camera Parameters
   └─ Extract K (intrinsic), R, t (extrinsic) from cameras.json
   
4. Project Gaussians → Uncertainty Map
   ├─ Transform Gaussians to camera space
   ├─ Project to 2D image plane
   ├─ Accumulate per-pixel properties using 2D Gaussian weights
   └─ Compute per-pixel σ from Gaussian properties
   
5. Calibration Phase
   ├─ Project Gaussians to calibration images
   ├─ Compute per-pixel errors
   ├─ Compute non-conformity scores: s = error / σ
   └─ Compute threshold: q = quantile_{1-α}(s)
   
6. Test Phase & Evaluation
   ├─ Check coverage on test images
   ├─ Compute confidence intervals
   └─ Visualize and save metrics
```

### Mathematical Details

#### Gaussian Projection (Step 4)

For each Gaussian $j$ at world position $\mathbf{P}_j$:

1. **Transform to camera space:**
   $$\mathbf{P}_j^{\text{cam}} = (\mathbf{P}_j - \mathbf{t}) \mathbf{R}^T$$

2. **Project to 2D image:**
   $$\mathbf{u}_j = \mathbf{K} \mathbf{P}_j^{\text{cam}} / z_j$$
   
   where $\mathbf{K}$ is the intrinsic matrix:
   $$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

3. **Compute projected 2D Gaussian:**
   - Pixel standard deviation: $\sigma_{\text{pixel}} = \frac{\|\mathbf{s}_j\| \cdot f_x}{z_j}$
   - where $\|\mathbf{s}_j\|$ is the magnitude of the 3D scale
   - Projection radius: $r = 3\sigma_{\text{pixel}}$

4. **Accumulate per-pixel properties:**
   
   For each pixel $\mathbf{p}$ within radius $r$ of $\mathbf{u}_j$:
   $$w_j(\mathbf{p}) = \exp\left(-\frac{\|\mathbf{p} - \mathbf{u}_j\|^2}{2\sigma_{\text{pixel}}^2}\right)$$
   
   Accumulate weighted contributions:
   - $C(\mathbf{p}) \leftarrow C(\mathbf{p}) + w_j(\mathbf{p})$
   - $A(\mathbf{p}) \leftarrow A(\mathbf{p}) + \alpha_j \cdot w_j(\mathbf{p})$
   - $Z(\mathbf{p}) \leftarrow Z(\mathbf{p}) + z_j \cdot w_j(\mathbf{p})$
   - $Z^2(\mathbf{p}) \leftarrow Z^2(\mathbf{p}) + z_j^2 \cdot w_j(\mathbf{p})$
   - $S(\mathbf{p}) \leftarrow S(\mathbf{p}) + \|\mathbf{s}_j\| \cdot w_j(\mathbf{p})$

5. **Normalize:**
   $$\bar{A}(\mathbf{p}) = \frac{A(\mathbf{p})}{C(\mathbf{p})}$$
   $$\bar{Z}(\mathbf{p}) = \frac{Z(\mathbf{p})}{C(\mathbf{p})}$$
   $$\text{Var}_Z(\mathbf{p}) = \frac{Z^2(\mathbf{p})}{C(\mathbf{p})} - \bar{Z}(\mathbf{p})^2$$

#### Sigma Computation (Step 5)

At each pixel, σ combines three uncertainty sources:

$$\sigma(\mathbf{p}) = 0.4 \cdot (1 - \bar{A}(\mathbf{p})) + 0.4 \cdot \frac{\text{Var}_Z(\mathbf{p})}{\max(\text{Var}_Z)} + 0.2 \cdot \left(1 - \frac{C(\mathbf{p})}{\max(C)}\right)$$

Where:
- **Opacity uncertainty:** $1 - \bar{A}$ — regions with low opacity are uncertain
- **Depth variance:** normalized spread in depths across Gaussians
- **Count uncertainty:** pixels hit by fewer Gaussians are uncertain

#### Non-conformity Score

For each pixel in calibration images:
$$s(\mathbf{p}) = \frac{|\text{render}(\mathbf{p}) - \text{gt}(\mathbf{p})|}{\sigma(\mathbf{p})}$$

#### Quantile Threshold

Aggregate all calibration non-conformity scores and compute:
$$q = \text{quantile}_{1-\alpha}(\{s_1, s_2, \ldots, s_n\})$$

For example, if $\alpha = 0.1$ (90% coverage), compute the 90th percentile.

#### Coverage Evaluation

On test images, prediction interval width is:
$$w = q \cdot \sigma(\mathbf{p})$$

Pixel is within interval if:
$$|\text{render}(\mathbf{p}) - \text{gt}(\mathbf{p})| \leq w$$

Coverage rate: fraction of test pixels satisfying this condition.

---


## Usage Commands

### Setup

The script requires the standard dependencies:
```bash
pip install numpy matplotlib pillow scipy
```

### 1. Uncertainty Maps

**Basic usage:**
```bash
python scripts/uncertainty_maps.py --scene train
```

**With custom parameters:**
```bash
python scripts/uncertainty_maps.py \
    --scene playroom \
    --calib-ratio 0.6 \
    --alpha 0.05 \
    --num-pixels 10000 \
    --output-dir my_uncertainty_output
```

**Arguments:**
- `--scene`: Scene name (drjohnson, playroom, train, truck) [required]
- `--num-pixels`: Pixels sampled per calibration image (default: 5000)
- `--alpha`: Significance level; coverage = 1 - alpha (default: 0.1)
- `--calib-ratio`: Fraction of images for calibration (default: 0.5)
- `--output-dir`: Output directory (default: uncertainty_maps)

**Output files:**
- `uncertainty_maps/{scene}/map_{scene}.png` — 5-column visualization
- `uncertainty_maps/{scene}/metrics.json` — Results and statistics

---

## Output Visualization

### From Images (5 columns)

1. **Render** — Rendered image
2. **Ground Truth** — Reference image
3. **Absolute Error** — Pixel-wise error (hot colormap)
4. **Conformal Coverage** — Green=inside interval, Red=outside (RdYlGn)
5. **Normalized Error** — error/interval_width (viridis colormap)

---

## Metrics Output (JSON)

The scripts saves metrics as JSON with structure:

```json
{
  "scene": "train",
  "target_coverage": 90.0,
  "actual_coverage": 89.5,
  "calibration_threshold": 1.282,
  "calibration_ratio": 0.5,
  "alpha": 0.1,
  "num_calibration_images": 50,
  "num_test_images": 50,
  "calibration_score_stats": {
    "mean": 1.001,
    "std": 0.345,
    "min": 0.001,
    "max": 5.234
  },
  "sigma_map_stats": {
    "mean": 0.025,
    "std": 0.018,
    "min": 0.010,
    "max": 0.089
  }
}
```

**Key metrics:**
- **target_coverage**: Desired coverage level (1 - α)
- **actual_coverage**: Empirical coverage on test set
- **calibration_threshold**: The learned quantile $q$
- **calibration_score_stats**: Distribution of non-conformity scores used to learn $q$

---

## Example Workflow

```bash
# 1. Generate uncertainty from image statistics
python scripts/uncertainty_maps.py \
    --scene train \
    --alpha 0.1 \
    --calib-ratio 0.5

# 2. Compare results
# - uncertainty_maps/train/metrics_from_gaussians.json
# - uncertainty_maps/train/metrics.json
```

---

## Theoretical Guarantees

**Conformal Prediction Guarantee:**

If the calibration and test data are exchangeable (identically distributed), then:

$$\mathbb{P}(|\hat{y}_i - y_i| \leq w_i) \geq 1 - \alpha - \frac{1}{n+1}$$

where $w_i$ is the prediction interval width at test point $i$, and $n$ is the calibration set size.

For large $n$, this approaches the desired coverage level $1 - \alpha$.

**Key property:** This guarantee holds **distribution-free** — no assumptions about error distribution!

---

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning in a random world.
- Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction under covariate shift.
