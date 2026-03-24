# 3D Gaussian Splatting — My Learning Journey

> Forked from [jonstephens85/gaussian-splatting-Windows](https://github.com/jonstephens85/gaussian-splatting-Windows), which builds on the original [INRIA implementation](https://github.com/graphdeco-inria/gaussian-splatting) by Kerbl et al. (SIGGRAPH 2023).

This repo documents my hands-on journey learning 3D Gaussian Splatting — from setting up the pipeline on a consumer GPU to understanding the math well enough to eventually implement it from scratch.

> For the full installation walkthrough, follow [Jon Stephens' guide](https://github.com/jonstephens85/gaussian-splatting-Windows) and his [YouTube tutorial](https://youtu.be/UXtuigy_wYc).

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

## Learning Roadmap

| Phase | Focus | Status |
| --- | --- | --- |
| **1** | 3D Transformations & Gaussian Math Foundations | In Progress |
| **2** | Structure from Motion & COLMAP Internals | Upcoming |
| **3** | Core 3DGS — Splatting Projection, Differentiable Rasterizer, Training Loop | Upcoming |
| **4** | CUDA Rasterization, Spherical Harmonics, Export & Evaluation | Upcoming |
| **5** | Full From-Scratch Implementation & Research Extensions | Upcoming |

---

## Credits

* **Paper:** Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023) — [[PDF]](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) [[Project Page]](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
* **Original Repo:** [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
* **Windows Guide:** [jonstephens85/gaussian-splatting-Windows](https://github.com/jonstephens85/gaussian-splatting-Windows)

## License

Inherited from the original repo. See [LICENSE.md](LICENSE.md).
