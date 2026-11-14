"""Measure stain density, stain colour and edge blur on a real H&E image.

Usage:
    python -I src/measure_real.py <image> <out_json>

Tissue statistics use the step-1 mask (segment.tissue_mask with the approved
default offset 0.10), on the *unsmoothed* OD. The same definition is applied to
synthetic renders so the two can be compared like for like.

Edge blur: at a random sample of mask-boundary points, the raw OD profile is
sampled along the gradient normal and fitted with a blurred step
    OD(s) = A * Phi((s - s0) / sigma)
using only edges with clean air on one side and a tissue plateau on the
other. The median sigma is reported (10-90% edge width = 2.563 sigma).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.special import ndtr
from skimage import filters, io, segmentation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment  # noqa: E402


def od_image(rgb):
    return -np.log((rgb[..., :3].astype(np.float64) + 1.0) / 256.0)  # per channel


def tissue_stats(od_sum, mask):
    v = od_sum[mask]
    q = np.percentile(v, [5, 10, 25, 50, 75, 90, 95])
    return {"n": int(v.size), "mean": float(v.mean()),
            **{f"p{p}": float(x) for p, x in zip([5, 10, 25, 50, 75, 90, 95], q)}}


def stain_colour(od_rgb, mask):
    """Mean OD direction (unit vector over R,G,B) inside tissue."""
    v = od_rgb[mask]
    v = v[v.sum(1) > 0.05]
    d = v.mean(0)
    return (d / d.sum()).tolist()


def _step(s, A, s0, sigma):
    return A * ndtr((s - s0) / sigma)


def edge_blur(od_sum, mask, rng, n_points=3000, half=8.0, step=0.25):
    sm = filters.gaussian(od_sum, 1.5)
    gy, gx = np.gradient(sm)
    bnd = np.argwhere(segmentation.find_boundaries(mask, mode="inner"))
    pick = bnd[rng.choice(len(bnd), size=min(n_points, len(bnd)), replace=False)]
    s = np.arange(-half, half + step / 2, step)
    sig = []
    for r, c in pick:
        g = np.array([gy[r, c], gx[r, c]])
        n = np.linalg.norm(g)
        if n < 1e-3:
            continue
        g /= n  # points uphill, i.e. from air into tissue
        rr, cc = r + s * g[0], c + s * g[1]
        prof = ndi.map_coordinates(od_sum, [rr, cc], order=1, mode="nearest")
        air, tis = prof[s <= -4], prof[(s >= 3) & (s <= 6)]
        if air.mean() > 0.03 or tis.mean() < 0.15 or tis.std() > 0.35 * tis.mean():
            continue  # need clean air and a reasonably flat tissue plateau
        fit_sel = s <= 6
        try:
            p, _ = curve_fit(_step, s[fit_sel], prof[fit_sel], p0=[tis.mean(), 0.0, 1.0],
                             bounds=([0, -5, 0.2], [5, 5, 5]), maxfev=2000)
        except RuntimeError:
            continue
        resid = prof[fit_sel] - _step(s[fit_sel], *p)
        ss = ((prof[fit_sel] - prof[fit_sel].mean()) ** 2).sum()
        if ss > 0 and 1 - (resid ** 2).sum() / ss > 0.9:
            sig.append(p[2])
    sig = np.array(sig)
    return {"n_fitted": int(sig.size), "n_sampled": int(len(pick)),
            "sigma_median": float(np.median(sig)),
            "sigma_p25": float(np.percentile(sig, 25)),
            "sigma_p75": float(np.percentile(sig, 75)),
            "edge_10_90_px_median": float(2.563 * np.median(sig))}


def render_profile(rgb, stats):
    """Inputs for synthetic.render_rgb_real: interior tissue OD quantiles (less
    affected by edge blur, used as the pre-blur distribution), stain colour, blur."""
    mask, _, _ = segment.tissue_mask_global(rgb)
    interior = ndi.binary_erosion(mask, iterations=2)
    v = od_image(rgb).sum(2)[interior]
    return {"od_quantiles": np.percentile(v, np.arange(101)).tolist(),
            "colour": stats["stain_od_direction_rgb"],
            "blur_sigma": stats["edge_blur"]["sigma_median"]}


def describe(rgb, rng):
    od_rgb = od_image(rgb)
    od_sum = od_rgb.sum(2)
    mask, thr, _ = segment.tissue_mask_global(rgb)  # v1 mask: fixed definition for stats
    interior = ndi.binary_erosion(mask, iterations=2)
    return {
        "shape": list(rgb.shape[:2]),
        "threshold_od": float(thr),
        "tissue_fraction": float(mask.mean()),
        "tissue_od_raw": tissue_stats(od_sum, mask),
        "tissue_od_raw_interior2px": tissue_stats(od_sum, interior),
        "stain_od_direction_rgb": stain_colour(od_rgb, mask),
        "background_white_fraction_of_air": float((rgb[..., :3][~mask] >= 254).all(-1).mean()),
        "edge_blur": edge_blur(od_sum, mask, rng),
    }


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    rgb = io.imread(sys.argv[1])
    stats = describe(rgb, np.random.default_rng(0))
    stats["image"] = Path(sys.argv[1]).name
    stats["render_profile"] = render_profile(rgb, stats)
    Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[2]).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
