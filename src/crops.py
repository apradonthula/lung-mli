"""Crop boxes used for hand-tracing and zoomed before/after figures.

Boxes are (x0, y0, size) in original-image pixels of Emphysema_low_mag.jpg.
"""
CROPS = {
    "A": (2700, 1000, 200),  # preserved alveoli: three-way junction, cellular walls
    "B": (450, 680, 200),    # emphysematous left side: pale thin membrane + cellular wall
    "C": (2900, 1200, 200),  # preserved side: pale thin wall, junctions, free round cells
}


def cut(arr, key):
    x0, y0, s = CROPS[key]
    return arr[y0:y0 + s, x0:x0 + s]
