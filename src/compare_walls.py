"""Walls-only comparison of automatic masks with hand traces (free cells removed).

Usage:
    python -I src/compare_walls.py <image> <traces_dir> <out_dir>

Rule: a piece (8-connected) smaller than segment.FREE_PIECE_PX that is not
attached to a wall is a free cell and is removed, from the traces and from the
automatic mask alike.
  automatic mask  applied to the full image (exactly what the pipeline does,
                  keeping pieces that touch the image edge),
                  then cut to the crop.
  trace           pieces not touching the crop edge: removed if smaller than the
                  cutoff. Pieces touching the crop edge may continue outside it,
                  where there is no trace, so they are judged from the full
                  image: kept if at least half their pixels lie on automatic
                  tissue that survives the full-image rule, otherwise removed.
The automatic masks are made with min_object_px=1, so that no other size filter
acts before this rule.

Writes to out_dir:
  walls_only.csv            per crop and pooled, for each edge fraction
  walls_only_cutoff.csv     pooled result vs cutoff (sensitivity to the rule)
  walls_only_pieces.csv     every traced piece with its size and whether it is kept
  walls_only.txt            the printed report
  small_pieces.png          every automatic piece of 100-1500 px in the full image
                            (frac 0.5), sorted by size; red = removed, green = kept
Metrics as in compare_traces.py: edge offset = (area_auto - area_trace) /
perimeter_trace (px/side, + = auto thicker); Lm from h+v test lines every 4 px,
min_tissue=3, pooled as 2*sum(L)/sum(T).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import io, measure

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_traces as ct  # noqa: E402
import crops  # noqa: E402
import mli  # noqa: E402
import segment  # noqa: E402

FRACS = (0.5, 0.55, 0.6)
CUTOFFS = (0, 100, 150, 200, 300, 400)
SPACING, MIN_TISSUE = 4, 3


def trace_walls(t, auto_kept, cutoff):
    """Apply the free-piece rule to a traced crop. auto_kept: the automatic crop
    after the full-image rule. Returns (walls mask, per-piece info list)."""
    lab, n = ndi.label(t, structure=np.ones((3, 3)))
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    keep = np.zeros(n + 1, bool)
    info = []
    for i in range(1, n + 1):
        p = lab == i
        a = int(p.sum())
        on_auto = float(auto_kept[p].mean())
        keep[i] = a >= cutoff or (i in edge and on_auto >= 0.5)
        info.append({"piece": i, "area_px": a, "touches_edge": i in edge,
                     "on_kept_auto": round(on_auto, 2), "kept": bool(keep[i])})
    return keep[lab], info


def compare(traces, autos_full, cutoff):
    """traces: {k: crop mask}; autos_full: full-image auto mask before the rule.
    Returns per-crop rows + pooled row."""
    full = segment.drop_free_pieces(autos_full, cutoff, keep_border=True) if cutoff else autos_full
    rows, pool = [], np.zeros(6)
    for k in traces:
        a = crops.cut(full, k)
        t = trace_walls(traces[k], a, cutoff)[0] if cutoff else traces[k]
        per = measure.perimeter(t)
        ma = mli.measure_lm(a, SPACING, min_tissue=MIN_TISSUE)
        mt = mli.measure_lm(t, SPACING, min_tissue=MIN_TISSUE)
        pool += [a.sum() - t.sum(), per, ma["L"], ma["T"], mt["L"], mt["T"]]
        rows.append({"crop": k, "edge_offset": (a.sum() - t.sum()) / per,
                     "dice": 2 * (a & t).sum() / (a.sum() + t.sum()),
                     "Lm_auto": ma["Lm"], "Lm_trace": mt["Lm"],
                     "Lm_err_pct": 100 * (ma["Lm"] / mt["Lm"] - 1)})
    dA, P, La, Ta, Lt, Tt = pool
    rows.append({"crop": "pooled", "edge_offset": dA / P, "dice": float("nan"),
                 "Lm_auto": 2 * La / Ta, "Lm_trace": 2 * Lt / Tt,
                 "Lm_err_pct": 100 * ((La / Ta) / (Lt / Tt) - 1)})
    return rows


def small_pieces_figure(rgb, full, cutoff, path, lo=100, hi=1500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import segmentation
    lab, n = ndi.label(full, structure=np.ones((3, 3)))
    size = np.bincount(lab.ravel())
    boxes = ndi.find_objects(lab)
    ids = sorted((i for i in range(1, n + 1) if lo <= size[i] < hi), key=lambda i: size[i])
    cols = 14
    rows = int(np.ceil(len(ids) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.62))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, i in zip(np.ravel(axs), ids):
        b = boxes[i - 1]
        cy, cx = (b[0].start + b[0].stop) // 2, (b[1].start + b[1].stop) // 2
        y0 = int(np.clip(cy - 40, 0, full.shape[0] - 80))
        x0 = int(np.clip(cx - 40, 0, full.shape[1] - 80))
        c = rgb[y0:y0 + 80, x0:x0 + 80].copy()
        c[segmentation.find_boundaries(lab[y0:y0 + 80, x0:x0 + 80] == i, mode="inner")] = \
            (210, 0, 0) if size[i] < cutoff else (0, 150, 0)
        ax.imshow(c, interpolation="nearest")
        ax.set_title(f"{size[i]}", fontsize=8, pad=1, color="#b00" if size[i] < cutoff else "#070")
    fig.suptitle(f"Automatic pieces {lo}-{hi} px (frac 0.5), 80x80 px views. Red outline: removed "
                 f"(< {cutoff} px); green: kept. Smaller pieces are not shown (all removed).", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98), pad=0.2)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(c for r in rows for c in r)))
        w.writeheader()
        w.writerows([{c: (round(v, 3) if isinstance(v, float) else v) for c, v in r.items()} for r in rows])


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    img, d, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    rgb = io.imread(img)[..., :3]
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    traces = {}
    for k, (_, _, size) in crops.CROPS.items():
        p = d / f"trace_{k}.png"
        if p.exists():
            traces[k] = ct.read_trace(p, size, ct._ref_for(rgb, k, p))
    if not traces:
        sys.exit(f"no trace_<K>.png in {d}")
    cut = segment.FREE_PIECE_PX

    full = {f: segment.tissue_mask(rgb, method="halfmax", frac=f, free_piece_px=None, min_object_px=1)[0]
            for f in FRACS}

    # traced pieces and what the rule does with them (edge pieces judged at frac 0.5)
    auto05 = segment.drop_free_pieces(full[0.5], cut, keep_border=True)
    piece_rows = []
    for k, t in traces.items():
        piece_rows += [{"crop": k, **r} for r in trace_walls(t, crops.cut(auto05, k), cut)[1]]
    flab, _ = ndi.label(full[0.5], structure=np.ones((3, 3)))
    fsize = np.bincount(flab.ravel())
    for r in piece_rows:
        t = traces[r["crop"]]
        lab = ndi.label(t, structure=np.ones((3, 3)))[0]
        ids = [int(o) for o in np.unique(crops.cut(flab, r["crop"])[lab == r["piece"]]) if o]
        r["auto_piece_px"] = ";".join(str(int(fsize[o])) for o in ids)
    small = [r for r in piece_rows if r["area_px"] < cut]
    say(f"cutoff {cut} px (segment.FREE_PIECE_PX); traced pieces, 8-connected:")
    say(f"  inside the crop, removed as free: {sorted(r['area_px'] for r in small if not r['touches_edge'])}")
    say("  touching the crop edge: " + ", ".join(
        f"{r['crop']} {r['area_px']} px ({'kept' if r['kept'] else 'removed'}, "
        f"{r['on_kept_auto']:.0%} on kept auto tissue)" for r in small if r["touches_edge"]))
    say(f"  walls: {sorted(r['area_px'] for r in piece_rows if r['area_px'] >= cut)}")
    say("  traced small piece -> size of the automatic full-image piece under it (frac 0.5): " + ", ".join(
        f"{r['crop']} {r['area_px']}->{r['auto_piece_px'] or 'none'}" for r in small))

    rows = []
    say("\nwalls only, same rule on trace and automatic crop:")
    say(f"{'frac':>5} {'crop':>7} {'edge_off':>9} {'dice':>6} {'Lm_auto':>8} {'Lm_trace':>8} {'err%':>6}")
    for f in FRACS:
        for r in compare(traces, full[f], cut):
            rows.append({"frac": f, "cutoff_px": cut, **r})
            say(f"{f:5} {r['crop']:>7} {r['edge_offset']:+9.3f} {r['dice']:6.3f} {r['Lm_auto']:8.1f} "
                f"{r['Lm_trace']:8.1f} {r['Lm_err_pct']:+6.1f}")

    sens = []
    say("\npooled result vs cutoff (0 = all pieces kept):")
    say(f"{'cutoff':>6} {'frac':>5} {'edge_off':>9} {'Lm_auto':>8} {'Lm_trace':>8} {'err%':>6}")
    for c in CUTOFFS:
        for f in FRACS:
            r = compare(traces, full[f], c)[-1]
            sens.append({"cutoff_px": c, "frac": f, **{x: r[x] for x in
                                                       ("edge_offset", "Lm_auto", "Lm_trace", "Lm_err_pct")}})
            say(f"{c:6d} {f:5} {r['edge_offset']:+9.3f} {r['Lm_auto']:8.1f} {r['Lm_trace']:8.1f} "
                f"{r['Lm_err_pct']:+6.1f}")

    small_pieces_figure(rgb, full[0.5], cut, out / "small_pieces.png")
    write_csv(out / "walls_only.csv", rows)
    write_csv(out / "walls_only_cutoff.csv", sens)
    write_csv(out / "walls_only_pieces.csv", piece_rows)
    (out / "walls_only.txt").write_text("\n".join(lines) + "\n")
    print("wrote", out / "walls_only.csv", "and walls_only_{cutoff,pieces}.csv, walls_only.txt")


if __name__ == "__main__":
    main()
