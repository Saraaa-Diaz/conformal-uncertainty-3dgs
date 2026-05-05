# Rasterizer patches

The CUDA modifications for the top-K blending weights live as a patch file
because the rasterizer is an upstream Inria submodule that we don't own.

## Apply the patch on a fresh clone

```bash
# After git submodule update --init --recursive
cd submodules/diff-gaussian-rasterization
git checkout dr_aa
git apply ../../patches/0001-rasterizer-topk-weights.patch
cd ../..

# Then build it
module load 2024 Anaconda3/2024.06-1
source activate gaussian_splatting
module load 2023 CUDA/12.1.1
export TORCH_CUDA_ARCH_LIST="8.0"
pip install --no-build-isolation submodules/diff-gaussian-rasterization
```

## What the patch adds

A new `_C.rasterize_gaussians_topk(...)` C++/CUDA entry that, per pixel,
returns the `K` largest alpha-blending weights `wᵢ = αᵢ · Tᵢ` sorted
descending. Templated on `K ∈ {1, 2, 4, 8}`. Inference-only — no backward.

Touched files:
- `cuda_rasterizer/forward.{cu,h}` — new `renderTopKWeightsCUDA` kernel
- `cuda_rasterizer/rasterizer.h` and `rasterizer_impl.cu` — new `Rasterizer::forward_topk_weights`
- `rasterize_points.{cu,h}` — new `RasterizeGaussiansTopKCUDA` C++ entry
- `ext.cpp` — bound as `rasterize_gaussians_topk`
- `diff_gaussian_rasterization/__init__.py` — `GaussianRasterizer.rasterize_topk_weights(...)` Python wrapper

The mod is purely additive — existing entry points are unchanged, so the
existing `render_color.py` / `render_depth.py` flow keeps working byte-for-byte.
