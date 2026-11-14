"""Zoomed before/after figures: mask outlines at native pixel resolution, enlarged.

Usage:
    python -I src/zoom_figure.py real <image> <out_png>
    python -I src/zoom_figure.py synthetic <profile.json> <out_png>

Each native pixel becomes a ZOOMxZOOM block (nearest neighbour), so individual
pixels and the exact mask edge are visible. The outline is drawn on the
inside edge of the mask pixels.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage import io, segmentation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crops  # noqa: E402
import segment  # noqa: E402

ZOOM = 4
OUTLINE = np.array([0, 190, 0], np.uint8)  # green: automatic mask
TRUTH = np.array([20, 20, 20], np.uint8)   # near-black: ground truth


def zoom(img):
    return np.kron(img, np.ones((ZOOM, ZOOM, 1), img.dtype)) if img.ndim == 3 else \
        np.kron(img, np.ones((ZOOM, ZOOM), img.dtype))


def outlined(rgb_crop, mask_crop, colour, width=2):
    big = zoom(rgb_crop[..., :3].copy())
    b = segmentation.find_boundaries(zoom(mask_crop.astype(np.uint8)).astype(bool), mode="inner")
    for _ in range(width - 1):
        b |= segmentation.find_boundaries(zoom(mask_crop.astype(np.uint8)).astype(bool) & ~b, mode="inner")
    big[b] = colour
    return big


def grid_figure(rows, col_titles, row_titles, out_png, note):
    n_r, n_c = len(rows), len(rows[0])
    tile = rows[0][0].shape[0]
    pad_top, pad_left, gap = 40, 34, 8
    W = pad_left + n_c * tile + (n_c - 1) * gap
    Hh = pad_top + n_r * tile + (n_r - 1) * gap + 30
    fig = plt.figure(figsize=(W / 100, Hh / 100), dpi=100)
    fig.patch.set_facecolor("white")
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            x = (pad_left + j * (tile + gap)) / W
            y = 1 - (pad_top + (i + 1) * tile + i * gap) / Hh
            ax = fig.add_axes([x, y, tile / W, tile / Hh])
            ax.imshow(im, interpolation="nearest")
            ax.set_axis_off()
            if i == 0:
                ax.set_title(col_titles[j], fontsize=11, pad=6)
        fig.text(4 / W, 1 - (pad_top + i * (tile + gap) + tile / 2) / Hh, row_titles[i],
                 fontsize=13, rotation=90, va="center", weight="bold")
    fig.text(pad_left / W, 8 / Hh, note, fontsize=9, color="#444")
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def real_figure(img_path, out_png):
    rgb = io.imread(img_path)[..., :3]
    variants = [
        ("v1 global, offset 0.10 (approved)", dict(method="global", offset=0.10)),
        ("v2 half-max, frac 0.5 (new default)", dict(method="halfmax", frac=0.5)),
        ("v2 half-max, frac 0.7", dict(method="halfmax", frac=0.7)),
    ]
    masks = [segment.tissue_mask(rgb, **kw)[0] for _, kw in variants]
    rows = []
    for k in crops.CROPS:
        c = crops.cut(rgb, k)
        rows.append([zoom(c)] + [outlined(c, crops.cut(m, k), OUTLINE) for m in masks])
    grid_figure(rows, ["original"] + [t for t, _ in variants],
                [f"{k}  (x={crops.CROPS[k][0]}, y={crops.CROPS[k][1]})" for k in crops.CROPS],
                out_png, f"Emphysema_low_mag.jpg, 200x200 px crops shown at {ZOOM}x (1 native px = "
                         f"{ZOOM}x{ZOOM} screen px). Green = mask edge (inside). "
                         "Source: Nephron, CC BY-SA 3.0; this figure is an adaptation under the same license.")


def synthetic_figure(profile_path, out_png):
    import synthetic as syn

    profile = json.loads(Path(profile_path).read_text())["profile"]
    Wd = 512
    rng = np.random.default_rng(7)
    seeds = syn.poisson_seeds(Wd, Wd, radius=40, rng=rng, pad=120)
    segs = syn.voronoi_segments(seeds, Wd, Wd, margin=80)
    t = rng.uniform(3, 12, len(segs))
    truth = syn.rasterize(segs, Wd, Wd, t)
    rgb = syn.render_rgb_real(truth, rng, profile)
    variants = [("v1 global, offset 0.10", dict(method="global", offset=0.10)),
                ("v2 half-max, frac 0.5", dict(method="halfmax", frac=0.5))]
    masks = [segment.tissue_mask(rgb, **kw)[0] for _, kw in variants]
    rows = []
    for (x0, y0) in [(60, 60), (260, 240)]:
        sl = (slice(y0, y0 + 200), slice(x0, x0 + 200))
        c = rgb[sl]
        rows.append([zoom(c), outlined(c, truth[sl], TRUTH)] +
                    [outlined(c, m[sl], OUTLINE) for m in masks])
    grid_figure(rows, ["calibrated render", "ground truth (black)"] + [t for t, _ in variants],
                ["S1", "S2"], out_png,
                f"Synthetic render calibrated to the real image (walls 3-12 px), shown at {ZOOM}x.")


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("real", "synthetic"):
        sys.exit(__doc__)
    Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
    (real_figure if sys.argv[1] == "real" else synthetic_figure)(sys.argv[2], sys.argv[3])
    print("wrote", sys.argv[3])


if __name__ == "__main__":
    main()
