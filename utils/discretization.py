import numpy as np


def digitize_clip(value, bins):
    idx = np.digitize([value], bins)[0] - 1
    return max(0, min(idx, len(bins) - 1))


def discretize_ternary(value, threshold=0.01):
    """Discretize a normalized value to ternary state: -1, 0, or 1.
    
    Args:
        value (float): Normalized value (typically between -1 and 1).
        threshold (float): Threshold for zero region. Default: 0.01.
                          Values in [-threshold, threshold] map to 0.
                          Values > threshold map to 1.
                          Values < -threshold map to -1.
    
    Returns:
        int: -1 (deficit), 0 (balanced), or 1 (surplus).
    
    Examples:
        >>> discretize_ternary(0.5, 0.01)
        1
        >>> discretize_ternary(0.005, 0.01)
        0
        >>> discretize_ternary(-0.5, 0.01)
        -1
    """
    if value > threshold:
        return 1
    elif value < -threshold:
        return -1
    else:
        return 0
