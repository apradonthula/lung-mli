"""One-page summary: original image, final mask with exclusions, Lm bar chart.

Usage:
    python -I src/summary_figure.py <image> <exclusion_dir> <out_png>

Reads the tissue mask from segment.tissue_mask (current defaults), the valid
mask and exclusion_lm.csv written by exclude.py.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage import io  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BEFORE, AFTER = "#2a78d6", "#eb6834"      # reference categorical slots 1, 2
EXCL = np.array([235, 104, 52])           # exclusions drawn in the "after" hue


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    img, d, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rgb = io.imread(img)[..., :3]
    tissue = segment.tissue_mask(rgb)[0]
    valid = io.imread(d / f"{img.stem}_valid.png") > 127
    rows = {r["exclusion"]: r for r in csv.DictReader(open(d / "exclusion_lm.csv"))}
    H, W = tissue.shape
    s = 3

    # mask panel: airspace = surface, measured tissue = dark grey, excluded = orange tint
    m = np.full((H, W, 3), 252, np.uint8)
    m[tissue] = (70, 70, 68)
    ex = ~valid
    m[ex] = (0.35 * m[ex] + 0.65 * EXCL).astype(np.uint8)

    fig = plt.figure(figsize=(16, 4.9), dpi=110, facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.66, 1.66, 1.05], wspace=0.13,
                          left=0.01, right=0.985, top=0.86, bottom=0.12)
    for i, (im, title) in enumerate([(rgb, "H&E, Emphysema_low_mag.jpg"),
                                     (m, "Final mask: tissue (grey), excluded (orange)")]):
        ax = fig.add_subplot(gs[i])
        ax.imshow(im[::s, ::s], extent=(0, W, H, 0), interpolation="antialiased")
        ax.axvline(W / 2, color=INK2, lw=1, ls=(0, (4, 3)))
        for x, lab in ((W * 0.25, "left half"), (W * 0.75, "right half")):
            ax.text(x, H * 0.04, lab, ha="center", va="top", fontsize=9, color=INK,
                    bbox=dict(fc=SURFACE, ec="none", alpha=0.85, pad=2))
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, color=INK, loc="left")

    ax = fig.add_subplot(gs[2])
    ax.set_facecolor(SURFACE)
    groups = ["whole", "left", "right"]
    x = np.arange(3)
    w = 0.36
    for k, (key, col, lab) in enumerate([("before", BEFORE, "no exclusions"),
                                          ("after", AFTER, "13 regions excluded")]):
        vals = [float(rows[key][f"Lm_{g}"]) for g in groups]
        xs = x + (k - 0.5) * (w + 0.02)
        ax.bar(xs, vals, width=w, color=col, label=lab, zorder=3)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 3, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, ["whole image", "left half", "right half"], fontsize=10, color=INK)
    ax.set_ylabel("Lm (px)", fontsize=10, color=INK2)
    ax.set_ylim(0, 240)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.tick_params(axis="y", labelsize=9, colors=INK2, length=0)
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2, labelcolor=INK,
              handlelength=1.2, columnspacing=1.2, borderaxespad=0.1)
    ax.set_title("Mean linear intercept", fontsize=11, color=INK, loc="left")

    fig.text(0.01, 0.035,
             "One diseased lung: left = severe emphysema, right = less affected. Lm in native pixels (no pixel size). "
             "Exclusions annotated by a non-pathologist.  Image: Nephron, CC BY-SA 3.0; this figure is shared under "
             "the same license.", fontsize=8.5, color=INK2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
