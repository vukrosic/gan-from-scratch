"""The two sets of real things: a ring of points and small pictures of digits."""
import math
import numpy as np
from sklearn.datasets import load_digits

RADIUS, SPREAD = 2.0, 0.05
SPOTS = np.array([[RADIUS * math.cos(2 * math.pi * k / 8), RADIUS * math.sin(2 * math.pi * k / 8)]
                  for k in range(8)], dtype=np.float32)          # eight spots on a circle


def ring(n, rng):
    """n real points: pick one of the eight spots, land very close to it."""
    k = rng.integers(0, 8, n)
    return (SPOTS[k] + rng.normal(0, SPREAD, (n, 2))).astype(np.float32)


def digits():
    """1,797 pictures of hand-written digits, 8 x 8 squares, each from -1 (paper) to 1 (ink)."""
    d = load_digits()
    return (d.data / 8.0 - 1.0).astype(np.float32), d.target


def draw(x):
    """One picture as text, for the terminal."""
    return '\n'.join(''.join(' .:*#'[min(4, int((v + 1) * 2.5))] * 2 for v in row) for row in np.asarray(x).reshape(8, 8))
