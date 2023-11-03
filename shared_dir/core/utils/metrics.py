__author__ = 'yunbo'

import numpy as np

def batch_psnr(gen_frames, gt_frames):
    if gen_frames.ndim == 3:
        axis = (1, 2)
    elif gen_frames.ndim == 4:
        axis = (1, 2, 3)
    x = np.int32(gen_frames)
    # print(x)
    # print()
    y = np.int32(gt_frames)
    # print(y)
    # print()
    num_pixels = float(np.size(gen_frames[0]))                               # num_pixels = 16384.0
    #print(num_pixels)
    mse = np.sum((x - y) ** 2, axis=axis, dtype=np.float32) / num_pixels     # mse = [0. 0. 0. 0.]
    #print(np.sum((x - y) ** 2, axis=axis, dtype=np.float32))
    #print(mse)
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)