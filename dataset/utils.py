import random
import numpy as np
import torch

# im_mean = (124, 116, 104)
im_mean = (0.5, 0.5, 0.5)

def reseed(seed):
    """Re-seed all RNGs (mirrors VOS pattern) for reproducible augmentations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for ni, l in enumerate(labels):
        Ms[ni] = (masks == l).astype(np.uint8)

    return Ms


def sort_by_number(filename: str) -> int:
    """
    Sort helper: extract the numeric portion of a filename.
    e.g. '3.png' -> 3, '10.png' -> 10. Returns -1 if no digits found.
    """
    # split on '.' and find the pure-digit segment
    # could also use a regex if filename layout varies
    for part in filename.split('.'):
        if part.isdigit():
            return int(part)
    return -1

def correct_dims(*images):
    """
    Expand (H, W) image to (H, W, 1) for Albumentations channel handling.
    Returns a single image if one is passed, else a list.
    """
    outputs = []
    for img in images:
        if len(img.shape) == 2:  # (H, W)
            img = np.expand_dims(img, axis=2)  # => (H, W, 1)
        outputs.append(img)
    if len(outputs) == 1:
        return outputs[0]
    return outputs
