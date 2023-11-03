__author__ = 'yunbo'

import numpy as np

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    img_height = np.shape(img_tensor)[2]
    img_width = np.shape(img_tensor)[3]
    num_channels = np.shape(img_tensor)[4]
    # img_tensor を 第二引数に示された形状に変換　ー＞　５次元配列から７次元配列に変換
    a = np.reshape(img_tensor, [batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels])
    #print('a.shape :', a.shape, '----------------------------------------------------------------------') (4, 20, 32, 4, 32, 4, 1)
    b = np.transpose(a, [0,1,2,4,3,5,6])
    #print('b.shape :', b.shape, '----------------------------------------------------------------------') (4, 20, 32, 32, 4, 4, 1)
    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels])
    #print('patch_tensor.shape :', patch_tensor.shape, '------------------------------------------------') (4, 20, 32, 32, 16)
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    # print('patch_tensor :', patch_tensor.shape)          (4, 19, 32, 32, 48)
    img_channels = channels // (patch_size*patch_size)
    # print('img_channels :', img_channels)                3
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    # print(a.shape)                                        (4, 19, 32, 32, 4, 4, 3)
    b = np.transpose(a, [0,1,2,4,3,5,6])
    # print(b.shape)                                        (4, 19, 32, 4, 32, 4, 3)
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    # print(img_tensor.shape)                               (4, 19, 128, 128, 3)
    return img_tensor

