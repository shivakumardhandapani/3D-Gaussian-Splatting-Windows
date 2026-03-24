# 3D Gaussian Splatting — Windows Setup & Learning Reference (In progress)

> Forked from [jonstephens85/gaussian-splatting-Windows](https://github.com/jonstephens85/gaussian-splatting-Windows), which builds on the original [INRIA implementation](https://github.com/graphdeco-inria/gaussian-splatting) by Kerbl et al. (SIGGRAPH 2023).

This repo documents my hands-on journey learning 3D Gaussian Splatting — from setting up the pipeline on a consumer GPU to understanding the math well enough to eventually implement it from scratch.

> For the full installation walkthrough, follow [Jon Stephens' guide](https://github.com/jonstephens85/gaussian-splatting-Windows) and his [YouTube tutorial](https://youtu.be/UXtuigy_wYc).
---

## What is 3D Gaussian Splatting?

Traditional Neural Radiance Fields (NeRFs) use neural networks to represent 3D scenes, which makes them slow to both train and render. Gaussian Splatting takes a fundamentally different approach: it represents scenes as collections of millions of **3D Gaussian primitives**, each defined by:

| Property | Description |
|---|---|
| **Position** (μ) | 3D mean — where the Gaussian sits in space |
| **Covariance** (Σ) | 3×3 matrix defining shape & orientation (anisotropic) |
| **Opacity** (α) | Transparency of the Gaussian |
| **Color** (SH coefficients) | View-dependent appearance via spherical harmonics (up to degree 3) |

These Gaussians are **differentiable** and rendered via a tile-based rasterizer — no ray marching required. The result: **real-time rendering (≥30 fps) at 1080p** with quality competitive to or exceeding NeRF-based methods.

### Pipeline at a Glance

```
Video of a scene
        │
        ▼
   ┌──────────┐
   │  ffmpeg  │  Extracts individual frames from the video at a chosen fps
   └────┬─────┘
        │
        ▼
   Photos (input images)
        │
        ▼
   ┌──────────┐
   │ COLMAP   │  Structure-from-Motion → camera poses + sparse point cloud
   └────┬─────┘
        │
        ▼
   ┌───────────────┐
   │  Optimizer    │  PyTorch + CUDA — iteratively refines 3D Gaussians
   │  (train.py)   │  from sparse points via differentiable rasterization
   └────┬──────────┘
        │
        ▼
   ┌──────────────┐
   │  SIBR Viewer │  OpenGL real-time viewer — navigate the trained scene
   └──────────────┘
```

### Key Concepts I'm Learning

- **Differentiable Rasterization** — The custom CUDA rasterizer (`diff-gaussian-rasterization` submodule) projects 3D Gaussians to 2D, sorts them by depth per-tile, and alpha-composites them. Gradients flow back through this entire process to optimize Gaussian parameters.

- **Adaptive Density Control** — During training, Gaussians are cloned (where detail is needed), split (where they're too large), or pruned (where opacity is near-zero). This happens between iterations 500–15,000 by default.

- **Spherical Harmonics for View-Dependent Color** — Rather than storing a single RGB value, each Gaussian stores SH coefficients (up to degree 3 = 48 coefficients), enabling view-dependent lighting effects like specular highlights.

- **Loss Function** — A weighted combination of L1 photometric loss and D-SSIM structural similarity: `L = (1 - λ) * L1 + λ * D-SSIM`, where λ = 0.2 by default.

- **Depth Regularization & Antialiasing** — Extensions to the base method that improve geometric consistency and reduce aliasing artifacts, evaluated across standard benchmarks (see [results.md](results.md)).

- **Exposure Compensation** — Optimizing a per-image 3×4 affine color transform to handle exposure variations across input photographs, improving training coherence without affecting real-time rendering.

---

## Evaluation Results

Detailed benchmark evaluations are documented by [taming-3dgs](https://github.com/graphdeco-inria/diff-gaussian-rasterization/tree/3dgs_accel) in [`results.md`](results.md), covering:

- **Default rasterizer** vs. **Accelerated rasterizer**
- **Optimizer comparison** — Default optimizer vs. Sparse Adam
- **Feature ablations** — Depth regularization (DR), antialiasing (AA), exposure compensation
- **Metrics** — PSNR, SSIM, LPIPS across MipNeRF360, Tanks&Temples, and Deep Blending datasets
- **Training time comparisons** — Baseline vs. accelerated rasterizer with both optimizer variants

---

## Repository Structure

```
├── train.py                # Main training script — optimizes Gaussians from SfM data
├── render.py               # Offline rendering of trained models at specified viewpoints
├── convert.py              # Prepares raw images → COLMAP SfM dataset
├── metrics.py              # Computes PSNR, SSIM, LPIPS evaluation metrics
├── full_eval.py            # End-to-end evaluation on standard benchmarks
├── results.md              # Benchmark evaluation results with charts
├── environment.yml         # Conda environment specification
│
├── gaussian_renderer/      # Differentiable Gaussian rasterization pipeline (Python)
├── scene/                  # Scene representation, dataset loaders, Gaussian model
├── arguments/              # CLI argument definitions for all scripts
├── submodules/
│   ├── diff-gaussian-rasterization/  # Custom CUDA rasterizer (core engine)
│   └── simple-knn/                   # K-nearest-neighbor utility
├── lpipsPyTorch/           # LPIPS perceptual metric
├── utils/                  # Loss functions, image I/O, general helpers
│
├── input/                  # Input images / scene data for training
├── install/                # Installation helpers and setup scripts
├── viewers/                # Pre-built SIBR real-time viewer binaries (Windows)
├── SIBR_viewers/           # SIBR viewer source (git submodule)
└── assets/                 # Documentation images and evaluation charts
```
>>>>>>> 8ddfa65cfe299da818f984fa5d7e509c05e2d55b

---

## My Setup

| Component | Spec |
| --- | --- |
| **CPU** | Intel Core i9 |
| **RAM** | 32 GB |
| **GPU** | NVIDIA RTX 4070 Laptop (8 GB VRAM) |
| **CUDA Toolkit** | 11.8 |
| **OS** | Windows |

---

## What I Learned Along the Way

### Understanding the Pipeline

3D Gaussian Splatting reconstructs photorealistic 3D scenes from photographs. Instead of neural networks (like NeRFs), it represents scenes as millions of 3D Gaussian primitives — each with a position, shape, opacity, and view-dependent color encoded via spherical harmonics. A custom CUDA rasterizer projects these to 2D and alpha-composites them, enabling real-time rendering at 30+ fps.

The full pipeline:

```
Photos/Video → COLMAP (SfM) → Sparse Point Cloud → Gaussian Optimization (train.py) → SIBR Viewer
```

Key concepts I've been working through:
- **Differentiable Rasterization** — EWA splatting projects 3D Gaussians to 2D; gradients flow back through the entire rasterizer to optimize Gaussian parameters end-to-end, much like backprop in deep learning.
- **Adaptive Density Control** — Gaussians are cloned, split, or pruned during training (iterations 500–15,000) to add detail where needed and remove noise.
- **Spherical Harmonics** — Each Gaussian stores up to 48 SH coefficients (degree 3) instead of plain RGB, enabling view-dependent effects like specular highlights.
- **Loss** — `L = (1 - λ) * L1 + λ * D-SSIM` with λ = 0.2 by default.

### Issues I Hit & How I Fixed Them

**1. CUDA/PyTorch version mismatch** — The default `environment.yml` didn't match my CUDA 11.8 setup. Fixed by manually creating the conda env with `pytorch==2.0.0` and `pytorch-cuda=11.8`.

**2. MSVC compiler errors during submodule compilation** — Two problems: `cl.exe` wasn't on PATH (fixed by running `vcvars64.bat` first), and CUDA 11.8 rejected my newer MSVC version (fixed by editing the `_MSC_VER` ceiling in `host_config.h`).

**3. Out-of-memory on 8 GB VRAM** — The defaults target 24 GB GPUs. My working flags:

```bash
python train.py -s <dataset> -r 2 --densify_grad_threshold 0.0004 --densify_until_iter 10000 --data_device cpu --test_iterations -1
```

`-r 2` (half resolution) is the biggest saver. Quality drops ~20–30% vs full training, but the scene is fully navigable.

**4. SIBR viewer needed CUDA 12** — The pre-built binaries didn't work with my CUDA 11.8. Built from source with CMake pointing to my CUDA 11.8 toolkit path instead.

---

## Evaluation Results

Benchmark evaluations are documented in [`results.md`](results.md) — comparing the default vs. accelerated rasterizer, Sparse Adam optimizer, depth regularization, antialiasing, and exposure compensation across MipNeRF360, Tanks&Temples, and Deep Blending datasets.

---

## Credits

* **Paper:** Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023) — [[PDF]](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) [[Project Page]](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
* **Original Repo:** [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
* **Windows Guide:** [jonstephens85/gaussian-splatting-Windows](https://github.com/jonstephens85/gaussian-splatting-Windows)

## License

Inherited from the original repo. See [LICENSE.md](LICENSE.md).
