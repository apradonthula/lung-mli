"""Step 2: exclude non-parenchymal structures (vessels, connective bands) from Lm.

Usage:
    python -I src/exclude.py <image> <annotation.json> <out_dir>

The annotation holds rough polygons around each structure (x, y in native px).
A polygon only has to enclose its structure without cutting into other alveoli:
the excluded region is the polygon intersected with the structure's hull,

    hull = fill_holes(closing(tissue & polygon, disk(CLOSE_R))), dilated by PAD px

so it follows the outer edge of the structure, includes its lumen and the gaps
in loose connective tissue, and gives the surrounding airspace back to the
measurement. Test lines are then split at the excluded region (mli.measure_lm
valid=...): only line length and air/tissue transitions in the valid region
count.

Writes to out_dir:
  <stem>_valid.png           white = measured, black = excluded
  <stem>_exclusion_overlay.png  half-size overview: excluded region tinted,
                             polygons dashed
  <stem>_exclusion_zoom.png  full-resolution view of each region
  exclusion_lm.csv           Lm before/after, whole image and left/right halves
                             (also without regions marked confidence "low")
  exclusion_regions.csv      per region: criterion, confidence, area
Which structures to exclude is decided by the criteria in README.md; each
region in the annotation cites its criterion (V, A or C) and a confidence.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import draw, io, morphology  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mli  # noqa: E402
import segment  # noqa: E402

CLOSE_R, PAD = 10, 6
SPACING, MIN_TISSUE = 8, 3


def load_regions(path):
    return json.loads(Path(path).read_text())["regions"]


def polygon_mask(poly, shape):
    xy = np.asarray(poly, float)
    rr, cc = draw.polygon(xy[:, 1], xy[:, 0], shape)
    m = np.zeros(shape, bool)
    m[rr, cc] = True
    return m


def exclusion_mask(tissue, regions):
    """Returns (excluded mask, {label: region mask})."""
    out, per = np.zeros(tissue.shape, bool), {}
    for r in regions:
        poly = polygon_mask(r["polygon"], tissue.shape)
        # pad so structures touching the image edge are not eroded by the closing
        pad = np.pad(tissue & poly, CLOSE_R + 1)
        hull = ndi.binary_closing(pad, morphology.disk(CLOSE_R))[CLOSE_R + 1:-CLOSE_R - 1, CLOSE_R + 1:-CLOSE_R - 1]
        hull = ndi.binary_fill_holes(hull)
        hull = ndi.binary_dilation(hull, morphology.disk(PAD)) & poly
        per[r["label"]] = hull
        out |= hull
    return out, per


def lm_table(tissue, valid):
    H, W = tissue.shape
    left = np.zeros((H, W), bool)
    left[:, : W // 2] = True
    rows = []
    for name, v in (("before", np.ones_like(tissue)), ("after", valid)):
        row = {"exclusion": name, "excluded_area_frac": round(float(1 - v.mean()), 4)}
        for part, p in (("whole", np.ones_like(tissue)), ("left", left), ("right", ~left)):
            r = mli.measure_lm(tissue, SPACING, valid=v & p, min_tissue=MIN_TISSUE)
            row[f"Lm_{part}"] = round(float(r["Lm"]), 1)
            row[f"tissue_frac_{part}"] = round(float(tissue[v & p].mean()), 4)
        row["left_over_right"] = round(float(row["Lm_left"] / row["Lm_right"]), 3)
        rows.append(row)
    return rows


def overview(rgb, tissue, excluded, regions, path):
    s = 2
    img = rgb[::s, ::s].astype(float)
    ex = excluded[::s, ::s]
    img[ex] = 0.45 * img[ex] + 0.55 * np.array([40, 90, 255])
    edge = ex & ~ndi.binary_erosion(ex, iterations=2)
    img[edge] = (0, 40, 200)
    fig, ax = plt.subplots(figsize=(rgb.shape[1] / s / 100, rgb.shape[0] / s / 100), dpi=100)
    ax.imshow(img.astype(np.uint8), extent=(0, rgb.shape[1], rgb.shape[0], 0))
    for r in regions:
        xy = np.asarray(r["polygon"] + [r["polygon"][0]], float)
        ax.plot(xy[:, 0], xy[:, 1], ls="--", lw=1.2, color="#222")
        cx, cy = xy[:-1].mean(0)
        ax.text(xy[:, 0].max() + 15, cy, r["label"], fontsize=10, color="#0028a0", weight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.75))
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def zoom_views(rgb, per, regions, path, margin=40):
    """One full-resolution panel per region: excluded area tinted, edge in blue,
    polygon dashed."""
    H, W = per[regions[0]["label"]].shape
    n = len(regions)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.8 * rows))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, r in zip(np.ravel(axs), regions):
        xy = np.asarray(r["polygon"], float)
        x0, y0 = (np.maximum(xy.min(0) - margin, 0)).astype(int)
        x1, y1 = np.minimum(xy.max(0) + margin, [W, H]).astype(int)
        e = per[r["label"]][y0:y1, x0:x1]
        c = rgb[y0:y1, x0:x1].astype(float)
        c[e] = 0.6 * c[e] + 0.4 * np.array([40, 90, 255])
        ax.imshow(c.astype(np.uint8), extent=(x0, x1, y1, y0), interpolation="nearest")
        if e.any() and not e.all():
            ax.contour(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5, e.astype(float),
                       levels=[0.5], colors=["#0028a0"], linewidths=1.0)
        cl = np.vstack([xy, xy[:1]])
        ax.plot(cl[:, 0], cl[:, 1], ls="--", lw=0.8, color="#333")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.axis("on")
        ax.tick_params(labelsize=6)
        ax.set_title(f"{r['label']}  [{r['criterion']}, {r['confidence']}]", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    img_path, ann, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    rgb = io.imread(img_path)[..., :3]
    tissue = segment.tissue_mask(rgb)[0]
    regions = load_regions(ann)
    excluded, per = exclusion_mask(tissue, regions)
    region_rows = []
    for r in regions:
        m = per[r["label"]]
        region_rows.append({"label": r["label"], "criterion": r["criterion"], "confidence": r["confidence"],
                            "area_px": int(m.sum()), "tissue_px": int((tissue & m).sum()),
                            "tissue_frac_inside": round(float(tissue[m].mean()), 3) if m.any() else 0})
        print(f"{r['label']:28s} {r['criterion']} {r['confidence']:6s} excluded {m.sum():7d} px, "
              f"tissue inside {region_rows[-1]['tissue_frac_inside']:.0%}")
    print(f"tissue inside exclusions: {(tissue & excluded).sum() / tissue.sum():.1%} of all tissue")
    rows = lm_table(tissue, ~excluded)
    keep_low = [r for r in regions if r["confidence"] != "low"]
    if len(keep_low) < len(regions):
        alt = lm_table(tissue, ~exclusion_mask(tissue, keep_low)[0])[1]
        alt["exclusion"] = "after, without low-confidence regions"
        rows.append(alt)
    for r in rows:
        print(f"{r['exclusion']:40s} Lm whole {r['Lm_whole']:6.1f}  left {r['Lm_left']:6.1f}  "
              f"right {r['Lm_right']:6.1f}  L/R {r['left_over_right']:.3f}  excluded {r['excluded_area_frac']:.1%}")
    stem = img_path.stem
    io.imsave(out / f"{stem}_valid.png", (~excluded * 255).astype(np.uint8), check_contrast=False)
    overview(rgb, tissue, excluded, regions, out / f"{stem}_exclusion_overlay.png")
    zoom_views(rgb, per, regions, out / f"{stem}_exclusion_zoom.png")
    for name, data in (("exclusion_lm.csv", rows), ("exclusion_regions.csv", region_rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print("wrote", out)


if __name__ == "__main__":
    main()
