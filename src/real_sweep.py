"""Lm on a real image across segmentation settings.

Usage:
    python -I src/real_sweep.py <image> <out_csv> [valid_mask.png]

Reports Lm (px) for the whole image and for its left and right halves
(Emphysema_low_mag.jpg: left = severe emphysema, right = relatively preserved),
plus tissue fraction. With a valid mask (white = measured, from exclude.py),
each setting is also measured with the exclusions applied (columns *_excl).
The exclusion regions are the ones built on the default segmentation.
Counting: lines every 8 px, horizontal + vertical, min_tissue=3.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from skimage import io

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mli  # noqa: E402
import segment  # noqa: E402

SPACING, MIN_TISSUE = 8, 3


def settings():
    for off in (0.06, 0.10, 0.15, 0.20, 0.30, 0.50):
        yield "global", {"offset": off}, f"offset={off}"
    for low in (0.05, 0.10, 0.20):
        for frac in (0.4, 0.5, 0.6, 0.7):
            yield "halfmax", {"low": low, "frac": frac}, f"low={low} frac={frac}"


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    rgb = io.imread(sys.argv[1])
    H, W = rgb.shape[:2]
    left = np.zeros((H, W), bool)
    left[:, : W // 2] = True
    valid = io.imread(sys.argv[3]) > 127 if len(sys.argv) == 4 else None
    rows = []
    for method, kw, label in settings():
        mask, _, _ = segment.tissue_mask(rgb, method=method, **kw)
        whole = mli.measure_lm(mask, SPACING, min_tissue=MIN_TISSUE)
        lm_l = mli.measure_lm(mask, SPACING, valid=left, min_tissue=MIN_TISSUE)["Lm"]
        lm_r = mli.measure_lm(mask, SPACING, valid=~left, min_tissue=MIN_TISSUE)["Lm"]
        rows.append({"method": method, "setting": label, "tissue_frac": round(float(mask.mean()), 4),
                     "Lm_whole": round(whole["Lm"], 1), "Lm_left": round(lm_l, 1),
                     "Lm_right": round(lm_r, 1), "left_over_right": round(lm_l / lm_r, 3)})
        r = rows[-1]
        line = (f"{method:8s} {label:20s} tissue {r['tissue_frac']:.3f}  Lm whole {r['Lm_whole']:6.1f}  "
                f"left {r['Lm_left']:6.1f}  right {r['Lm_right']:6.1f}  L/R {r['left_over_right']:.3f}")
        if valid is not None:
            we = mli.measure_lm(mask, SPACING, valid=valid, min_tissue=MIN_TISSUE)["Lm"]
            le = mli.measure_lm(mask, SPACING, valid=valid & left, min_tissue=MIN_TISSUE)["Lm"]
            re_ = mli.measure_lm(mask, SPACING, valid=valid & ~left, min_tissue=MIN_TISSUE)["Lm"]
            r.update({"Lm_whole_excl": round(we, 1), "Lm_left_excl": round(le, 1),
                      "Lm_right_excl": round(re_, 1), "left_over_right_excl": round(le / re_, 3)})
            line += f"  | excl: whole {we:6.1f} left {le:6.1f} right {re_:6.1f} L/R {le / re_:.3f}"
        print(line)
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
