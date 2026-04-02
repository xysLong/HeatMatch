# HeatMatch

**Orientation-Aware Visualization and Comparison of Very Dense Saccade Patterns**

This repository contains the Python implementation accompanying the paper:

> Xingyu Long, Jozsef Arato, Sophia Kury, Anna Miscena, and Raphael Rosenberg. 2026.
> **HeatMatch: Orientation-Aware Visualization and Comparison of Very Dense Saccade Patterns.**
> *Proceedings of the ACM on Computer Graphics and Interactive Techniques* (PACMCGIT), 9(2), Article 18 (June 2026).
> https://doi.org/10.1145/3803539

The preprocessed saccade dataset used in the paper is archived at **https://osf.io/f2xhj/** and is not included in this repository. To run [`demo.ipynb`](demo.ipynb), download `data_anonymized.csv` from OSF and place it in a `tests/` folder at the repo root.

<img src="assets/crea_logo.jpg" height="100" alt="CReA Lab"/>&nbsp;&nbsp;&nbsp;&nbsp;<img src="assets/cogscihub_logo.png" height="100" alt="Vienna Cognitive Science Hub"/>

---

## Overview

Scanpath comparison is challenging for long free-viewing of complex stimuli (e.g., paintings), where saccades are dense, idiosyncratic in order, and numerous. HeatMatch addresses this with two contributions:

1. **OOI-coded heatmaps.** Saccades are aggregated into a continuous orientation field over a reference grid. Each grid point receives a kernel-weighted mean orientation, a coherence score (mean resultant length), and a density estimate. Heatmaps are rendered by anchoring the colormap to researcher-defined *Orientations of Interest* (OOIs) and modulating opacity by local confidence — all without AOI annotation.

2. **HeatMatch similarity.** Two saccade patterns are compared via a composite score *S* = (*S*_loc + *S*_dir) / 2, where *S*_loc is a Pearson-correlation-based density similarity and *S*_dir is a density- and coherence-weighted mean of per-point axial angular similarities.

The method is permutation-invariant, segmentation-free, and scales to thousands of saccades.

---

## Installation

```bash
git clone https://github.com/xysLong/HeatMatch.git
cd HeatMatch
pip install -e .
```

Dependencies: `numpy`, `matplotlib` (see `pyproject.toml`).

---

## Quick Start

```python
import matplotlib.pyplot as plt
from heatmatch import make_reference_grid, make_orientation_field, Heatmap, compute_similarity

# Saccade data: onset/offset coordinates, shape (J, 2)
onset = ...  # array of saccade starting points (x, y)
offset = ...  # array of saccade end points      (x, y)

W, H = 2880, 2160  # stimulus dimensions in pixels

# 1. Build reference grid and compute orientation field
pts, xx, yy = make_reference_grid(W, H, nx=200, ny=200)
field = make_orientation_field(pts, onset, offset, sigma=50.0, grid_shape=yy.shape)

# 2. Render OOI-coded heatmap (ooi=0 → horizontal; ooi=90 → vertical)
fig, ax = plt.subplots()
heatmap = Heatmap(field, W, H)
heatmap.draw(ax, ooi=0.0)
plt.show()

# 3. Compare two saccade patterns A and B
field_a = make_orientation_field(pts, onset_a, offset_a)
field_b = make_orientation_field(pts, onset_b, offset_b)
result = compute_similarity(field_a, field_b)
print(result.s_loc, result.s_dir)
```

A worked example over all 12 stimuli from the paper is in [`demo.ipynb`](demo.ipynb).

---

## API

### `heatmatch.fields`

| Symbol | Description |
|---|---|
| `make_reference_grid(w, h, nx, ny)` | Create uniform N×N reference grid — returns `(pts, xx, yy)` |
| `make_orientation_field(pts, onset, offset, sigma, grid_shape, ...)` | Compute field statistics (ω̄, R, ρ) — Eqs. (2)–(4); returns `OrientationField` |
| `OrientationField` | Dataclass holding `omega_mean`, `R`, `rho` (each shape `(P,)`) and `grid_shape` |

### `heatmatch.heatmapping`

| Symbol | Description |
|---|---|
| `Heatmap(field, w, h)` | Renderer wrapping an `OrientationField`; caches colormap and opacity arrays |
| `Heatmap.draw(ax, ooi, density_weight, cmap, degree, quantile)` | Render OOI-coded heatmap onto `ax` — §3.2 |

`Heatmap.draw` accepts OOI in **degrees** with compass convention: 0 = east, 90 = north, 180 = west, 270 = south. Orientations are unsigned, so `ooi` and `ooi ± 180` are equivalent. The colormap is rebuilt only when `cmap` changes; the opacity array is rebuilt only when `density_weight`, `degree`, or `quantile` changes.

### `heatmatch.matching`

| Symbol | Description |
|---|---|
| `compute_similarity(field_a, field_b, t)` | Compute HeatMatch similarity — returns `SimilarityResult` |
| `SimilarityResult` | Dataclass with `s_loc` (Eq. 5) and `s_dir` (Eq. 6) |

The parameter `t ∈ [0, 1]` trades off density (`t=1`) against coherence (`t=0`) in the directional similarity weight ρ̄^t · R̄^(1−t).

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `sigma` | `50.0` | Gaussian kernel bandwidth σ in pixels. Controls the spatial scale of aggregation. For similarity analysis, the paper evaluates σ ∈ {50, 100, 150} px. |
| `nx`, `ny` | `200` | Reference grid resolution. The paper uses N = 200. |
| `ooi` | `0.0` | Orientation of Interest in degrees (0 = horizontal). |
| `density_weight` | `1.0` | Blend between density and coherence for heatmap opacity: 1.0 → density only, 0.0 → coherence only. |
| `t` | `0.5` | Density–coherence trade-off for *S*_dir in similarity scoring — Eq. (6). |

---

## Citation

```bibtex
@article{long2026heatmatch,
  author    = {Long, Xingyu and Arato, Jozsef and Kury, Sophia and Miscena, Anna and Rosenberg, Raphael},
  title     = {HeatMatch: Orientation-Aware Visualization and Comparison of Very Dense Saccade Patterns},
  journal   = {Proc. ACM Comput. Graph. Interact. Tech.},
  year      = {2026},
  volume    = {9},
  number    = {2},
  articleno = {18},
  doi       = {10.1145/3803539},
}
```
