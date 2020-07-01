import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import random

import math
import torch


def sample_mask(im_size: int, window_size: int):
    """
    Args:
    - im_size: size of image
    - window_size: size of window. if -1, return full size mask
    """
    assert im_size >= 2
    assert (1 <= window_size <= im_size) or (window_size == -1)

    # if window_size == -1, return all True mask.
    if window_size == -1:
        return torch.ones(im_size, im_size, dtype=torch.bool)

    mask = torch.zeros(im_size, im_size, dtype=torch.bool)  # all elements are False

    # sample window center. if window size is odd, sample from pixel position. if even, sample from grid position.
    window_center_h = random.randrange(0, im_size) if window_size % 2 == 1 else random.randrange(0, im_size + 1)
    window_center_w = random.randrange(0, im_size) if window_size % 2 == 1 else random.randrange(0, im_size + 1)

    for idx_h in range(window_size):
        for idx_w in range(window_size):
            h = window_center_h - math.floor(window_size / 2) + idx_h
            w = window_center_w - math.floor(window_size / 2) + idx_w

            if (0 <= h < im_size) and (0 <= w < im_size):
                mask[h, w] = True

    return mask


if __name__ == '__main__':
    print(sample_mask(6, 2))
    print(sample_mask(6, 3))
    print(sample_mask(5, 2))
    print(sample_mask(5, 3))
