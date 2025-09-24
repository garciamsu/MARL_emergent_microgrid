import numpy as np


def digitize_clip(value, bins):
    idx = np.digitize([value], bins)[0] - 1
    return max(0, min(idx, len(bins) - 1))
