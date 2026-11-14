"""Close-up of one enclosed gap in an automatic mask, to judge whether it is real.

Usage:
    python -I src/hole_figure.py <image> <traces_dir> <crop> <out_png> [frac]

Finds the largest hole (unmasked region fully enclosed by tissue) in the crop at
the given edge fraction (default 0.5) and shows a 44x44 px window around it at
10x: the original, optical density with the local half-max level, the
automatic mask edge, the hand-trace edge, and an OD profile through the hole.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import io  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_traces as ct  # noqa: E402
import crops  # noqa: E402
import segment  # noqa: E402

W = 44


def main():
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    img, d, k, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    frac = float(sys.argv[5]) if len(sys.argv) == 6 else 0.5
    rgb = io.imread(img)[..., :3]
    m, _, od = segment.tissue_mask(rgb, method="halfmax", frac=frac)
    bg = segment.background_od(od)
    a, o, c = crops.cut(m, k), crops.cut(od, k), crops.cut(rgb, k)
    holes, n = ndi.label(ndi.binary_fill_holes(a) & ~a)
    if n == 0:
        sys.exit(f"no enclosed gap in crop {k} at frac {frac}")
    hole = holes == np.bincount(holes.ravel())[1:].argmax() + 1
    ys, xs = np.nonzero(hole)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0, x0 = int(np.clip(cy - W // 2, 0, 200 - W)), int(np.clip(cx - W // 2, 0, 200 - W))
    win = (slice(y0, y0 + W), slice(x0, x0 + W))
    p = d / f"trace_{k}.png"
    t = ct.read_trace(p, 200, ct._ref_for(rgb, k, p)) if p.exists() else None

    ring = ndi.binary_dilation(hole, iterations=4) & a
    air = ~ndi.binary_dilation(a, iterations=3)
    stats = (f"hole: {hole.sum()} px.  OD (sum over RGB, background {bg:.2f}): hole median "
             f"{np.median(o[hole]):.2f} (min {o[hole].min():.2f}); surrounding wall median "
             f"{np.median(o[ring]):.2f}; airspace median {np.median(o[air]):.2f}."
             + (f"  Trace covers {t[hole].mean():.0%} of the hole." if t is not None else ""))

    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.25])
    ext = (x0 - 0.5, x0 + W - 0.5, y0 + W - 0.5, y0 - 0.5)

    def panel(i, title, img_, cmap=None, outline=None, colour=None, **kw):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img_, cmap=cmap, interpolation="nearest", extent=ext, **kw)
        for msk, col in (outline or []):
            ax.contour(np.arange(x0, x0 + W), np.arange(y0, y0 + W), msk[win].astype(float),
                       levels=[0.5], colors=[col], linewidths=1.6)
        ax.axhline(cy, color="#333", lw=0.6, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    panel(0, f"crop {k}, rows {y0}-{y0 + W}, cols {x0}-{x0 + W}", c[win])
    ax = panel(1, "optical density", o[win], cmap="magma", vmin=0, vmax=max(1.2, np.percentile(o[win], 99)))
    panel(2, f"automatic, frac {frac} (green)", c[win], outline=[(a, "#00a000"), (hole, "#0050ff")])
    if t is not None:
        panel(3, "your trace (black)", c[win], outline=[(t, "black")])
    ax = fig.add_subplot(gs[4])
    xs_ = np.arange(x0, x0 + W)
    ax.plot(xs_, o[cy, x0:x0 + W], color="#7a1f5c", lw=1.8, label="OD along dotted row")
    ax.axhline(bg, color="#888", lw=0.8, ls="--", label="background (glass)")
    inside = hole[cy, x0:x0 + W]
    if inside.any():
        ax.axvspan(xs_[inside].min() - 0.5, xs_[inside].max() + 0.5, color="#0050ff", alpha=0.12,
                   label="hole")
    auto_row = a[cy, x0:x0 + W]
    ax.scatter(xs_[auto_row], np.full(auto_row.sum(), -0.06), s=8, color="#00a000", label="auto tissue")
    if t is not None:
        tr = t[cy, x0:x0 + W]
        ax.scatter(xs_[tr], np.full(tr.sum(), -0.12), s=8, color="black", label="traced tissue")
    ax.set_xlabel("crop column (px)")
    ax.set_ylabel("OD")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("profile through the hole", fontsize=10)
    fig.text(0.01, 0.02, stats, fontsize=9.5)
    fig.text(0.01, 0.96, "Source: Emphysema_low_mag.jpg, Nephron, CC BY-SA 3.0; adaptation under the same license.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    print(stats)
    print("wrote", out)


if __name__ == "__main__":
    main()
