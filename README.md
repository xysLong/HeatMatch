# HeatMatch

**Orientation-Aware Visualization and Comparison of Very Dense Saccade Patterns**

<table border="0" cellspacing="0" cellpadding="0">
<tr>
<td width="72%" valign="top">

This repository contains the Python implementation accompanying the paper:

<blockquote>
Xingyu Long, Jozsef Arato, Sophia Kury, Anna Miscena, and Raphael Rosenberg. 2026.<br/>
<strong>HeatMatch: Orientation-Aware Visualization and Comparison of Very Dense Saccade Patterns.</strong><br/>
<em>Proceedings of the ACM on Computer Graphics and Interactive Techniques</em> (PACMCGIT), 9(2), Article 18 (June 2026).<br/>
<a href="https://doi.org/10.1145/3803539">https://doi.org/10.1145/3803539</a>
</blockquote>

</td>
<td width="28%" align="center" valign="middle">
<img src="assets/logos.png" width="200" alt="CReA Lab and Vienna Cognitive Science Hub"/>
</td>
</tr>
</table>

---

## Overview

HeatMatch provides two tools for dense free-viewing saccade data:

1. **OOI-coded heatmaps** — orientation fields visualized with researcher-defined color anchors and confidence-weighted opacity.
2. **HeatMatch similarity** — a composite score *S* = (*S*_loc + *S*_dir) / 2 comparing saccade density and orientation between two patterns.

See the [paper](https://doi.org/10.1145/3803539) for the full methodology.

---

## Example Output

OOI-coded heatmaps for four paintings, shown per individual participant (columns 1–5) and aggregated (column 6). Hue encodes mean saccade orientation relative to the OOI; opacity encodes local confidence.

![HeatMatch heatmap example](assets/heatmaps.svg)

<sub>Stimulus images are public domain paintings sourced from [Wikimedia Commons](https://commons.wikimedia.org). See the paper for provenance and rights details.</sub>

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
pts, xx, yy = make_reference_grid(W, H, grid_resolution=200)
field = make_orientation_field(pts, onset, offset, sigma=50.0, grid_shape=yy.shape)

# 2. Render OOI-coded heatmap (ooi=0 → horizontal; ooi=90 → vertical)
fig, ax = plt.subplots()
heatmap = Heatmap(field, W, H)
heatmap.draw(ax, ooi=0.0, opacity_density_weight=1.0)
plt.show()

# 3. Compare two saccade patterns A and B
field_a = make_orientation_field(pts, onset_a, offset_a)
field_b = make_orientation_field(pts, onset_b, offset_b)
result = compute_similarity(field_a, field_b)
print(result.s_loc, result.s_dir)
```

A worked example over all 12 stimuli from the paper is in [`demo.ipynb`](demo.ipynb). The preprocessed saccade dataset is not included in this repository — download `data_anonymized.csv` from **[osf.io/f2xhj](https://osf.io/f2xhj/)** and place it in a `tests/` folder at the repo root.

---

## API

### `heatmatch.fields`

| Symbol | Description |
|---|---|
| `make_reference_grid(w, h, grid_resolution)` | Create reference grid — returns `(pts, xx, yy)`; `grid_resolution` is an int (square) or `(ny, nx)` tuple |
| `make_orientation_field(pts, onset, offset, sigma, grid_shape, ...)` | Compute field statistics (ω̄, R, ρ) — Eqs. (2)–(4); returns `OrientationField` |
| `OrientationField` | Dataclass holding `omega_mean`, `R`, `rho` (each shape `(P,)`) and `grid_shape` |

### `heatmatch.heatmapping`

| Symbol | Description |
|---|---|
| `Heatmap(field, w, h, image=None)` | Renderer wrapping an `OrientationField`; optional stimulus image is converted to grayscale once and cached |
| `Heatmap.draw(ax, ooi, opacity_density_weight, base_opacity, cmap, degree, quantile)` | Render OOI-coded heatmap onto `ax` — §3.2 |

`Heatmap.draw` accepts OOI in **degrees** with compass convention: 0 = east, 90 = north, 180 = west, 270 = south. Orientations are unsigned, so `ooi` and `ooi ± 180` are equivalent. The colormap is rebuilt only when `cmap` changes; the opacity array is rebuilt only when `opacity_density_weight`, `degree`, or `quantile` changes.

`opacity_density_weight` blends density and coherence into the opacity signal (1.0 = density only, 0.0 = coherence only). Recommended range: 0.5–1.0.

### `heatmatch.matching`

| Symbol | Description |
|---|---|
| `compute_similarity(field_a, field_b, density_coherence_tradeoff)` | Compute HeatMatch similarity — returns `SimilarityResult` |
| `SimilarityResult` | Dataclass with `s_loc` (Eq. 5) and `s_dir` (Eq. 6) |

`density_coherence_tradeoff` controls the balance between density and coherence in the S_dir weight (1.0 = density only, 0.0 = coherence only). At 1.0 the weight reduces to pure density, making S_dir statistically redundant with S_loc — angular coherence has no influence on the score. Recommended ~0.5.

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `sigma` | `50.0` | Gaussian kernel bandwidth σ in pixels. Controls the spatial scale of aggregation. For similarity analysis, the paper evaluates σ ∈ {50, 100, 150} px. |
| `grid_resolution` | `200` | Reference grid resolution. An int gives a square grid; a `(ny, nx)` tuple allows non-square grids. The paper uses 200. |
| `ooi` | `0.0` | Orientation of Interest in degrees (0 = horizontal). |
| `opacity_density_weight` | `1.0` | Blend between density and coherence for heatmap opacity. Recommended 0.5–1.0. |
| `base_opacity` | `0.85` | Global opacity ceiling multiplied with the data-derived opacity. Useful when a background image is present. |
| `density_coherence_tradeoff` | `0.5` | Density–coherence trade-off for *S*_dir weighting. At 1.0, angular coherence has no influence — recommended ~0.5. |

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
