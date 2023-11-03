import cv2
import numpy as np
import argparse
import shutil
import os
import random
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from tqdm import tqdm
from numpy.core.shape_base import block
from numpy.lib.type_check import real
import skimage.util

import torch
import lpips
from IPython import embed

def main():

    use_gpu = True
    spatial = True
    loss_fn = lpips.LPIPS(net='alex', spatial = spatial)

    if(use_gpu):
        loss_fn.cuda()

    parser = argparse.ArgumentParser(description='img difference add color')
    parser.add_argument('--metric_img_dir', type=str, default='testA', help='real RGB img directory')
    parser.add_argument('--save_directory', type=str, default='save', help='save directory')
    parser.add_argument('--anomaly_value', type=float, default=0.6, help='anomaly value in picture(0~1)')
    # parser.add_argument('--deviation_value', type=int, default=55, help='deviation value')
    # parser.add_argument('--threshold', type=int, default=100, help='threshold(0~255)')

    # # 切り取る上のy座標
    high_y = 0
    # # 切り取る下のy座標
    low_y = 256

    args = parser.parse_args()

    metric_img_dir:str = args.metric_img_dir
    real_img_name:str = args.metric_img_dir + '/real'
    fake_img_name:str = args.metric_img_dir + '/fake'
    mask_img_name:str = args.metric_img_dir + '/gt_mask'
    depth_img_name:str = args.metric_img_dir + '/depth'
    save_directory:str = args.save_directory
    anomaly_value = args.anomaly_value

    F = [[], [], [], [], [], []]
    F_fin = [0, 0, 0, 0, 0, 0]

    step = ['1', '2', '3', '4', '5', '6']

    #画像のファイル名を読み込み
    # real_img_list = os.listdir(real_img_name)
    # fake_img_list = os.listdir(fake_img_name)

    for s in step:
        real_img_dir_step = os.path.join(real_img_name, s)
        fake_img_dir_step = os.path.join(fake_img_name, s)
        mask_img_dir_step = os.path.join(mask_img_name, s)
        depth_img_dir_step = os.path.join(depth_img_name, s)

        real_img_dir = os.listdir(real_img_dir_step)
        fake_img_dir = os.listdir(fake_img_dir_step)
        mask_img_dir = os.listdir(mask_img_dir_step)
        depth_img_dir = os.listdir(depth_img_dir_step)

        save_dir_name = os.path.join(save_directory, s)

        # if os.path.basename(save_dir_name)=="1":
        #     # 切り取る上のy座標
        #     high_y = 70
        #     # 切り取る下のy座標
        #     low_y = 190
        # elif os.path.basename(save_dir_name)=="2":
        #     high_y = 90
        #     low_y = 190
        # elif os.path.basename(save_dir_name)=="3":
        #     high_y = 90
        #     low_y = 190
        # elif os.path.basename(save_dir_name)=="4":
        #     high_y = 70
        #     low_y = 150
        # elif os.path.basename(save_dir_name)=="5":
        #     high_y = 60
        #     low_y = 160
        # elif os.path.basename(save_dir_name)=="6":
        #     high_y = 110
        #     low_y = 190
        # else:
        #     print('directory_name :', os.path.basename(save_dir_name))
        #     raise Exception("Unexpected directory")

        # 順番を０からにする
        real_img_dir.sort()
        fake_img_dir.sort()
        mask_img_dir.sort()
        depth_img_dir.sort()

        print('---------------------------------------------')
        print(real_img_dir)
        print(fake_img_dir)
        print(mask_img_dir)
        print(depth_img_dir)


        os.makedirs(save_dir_name, exist_ok=True)

        # together = save_dir_name + '/' + 'together/'
        # if os.path.exists(together):
        #     shutil.rmtree(together)
        # os.makedirs(together)

        heatmap = save_dir_name + '/' + 'heatmap/'
        if os.path.exists(heatmap):
            shutil.rmtree(heatmap)
        os.makedirs(heatmap)

        real_detection = save_dir_name + '/' + 'detection/'
        if os.path.exists(real_detection):
            shutil.rmtree(real_detection)
        os.makedirs(real_detection)

        mask_detection = save_dir_name + '/' + 'mask/'
        if os.path.exists(mask_detection):
            shutil.rmtree(mask_detection)
        os.makedirs(mask_detection)

        # real_img_dir に入っているものをひとつずつ取り出す
        for i in tqdm(range(len(real_img_dir))):
            # lpips算出用画像の読み込み
            real_img = lpips.im2tensor(lpips.load_image(os.path.join(real_img_dir_step, real_img_dir[i]))) # torch.Size[(1, 3, 256, 256)]
            fake_img = lpips.im2tensor(lpips.load_image(os.path.join(fake_img_dir_step, fake_img_dir[i])))
            # 通常用途画像の読み込み
            real = cv2.imread(os.path.join(real_img_dir_step, real_img_dir[i]), cv2.IMREAD_COLOR)
            gt_mask_img = cv2.imread(os.path.join(mask_img_dir_step, mask_img_dir[i]), cv2.IMREAD_UNCHANGED)
            depth_img = cv2.imread(os.path.join(depth_img_dir_step, depth_img_dir[i]), cv2.IMREAD_UNCHANGED)

            # 商品だけ切り取り
            real_cut = real_img[:,:,high_y:low_y, 0:real_img.shape[3]]
            fake_cut = fake_img[:,:,high_y:low_y, 0:real_img.shape[3]]
            real_real_cut = real.copy()[high_y:low_y, 0:gt_mask_img.shape[1]]
            gt_mask_cut = gt_mask_img.copy()[high_y:low_y, 0:gt_mask_img.shape[1]]
            depth_cut = depth_img.copy()[high_y:low_y, 0:gt_mask_img.shape[1]]
            _,_,h, w = real_cut.shape
            # print(real_real_cut.dtype)
            # print(real_real_cut)
            # exit()
            
            if(use_gpu):
                real_cut = real_cut.cuda()
                fake_cut = fake_cut.cuda()
            
            # lpips による異常度ヒートマップ算出＆作成
            comp = loss_fn.forward(real_cut, fake_cut)
            comp = comp[0,0,:,:].data.cpu().numpy()
            comp = depth_apply(depth_cut, comp)
            plt.imsave(save_dir_name + '/heatmap/' + (str)(i+1) + 'heatmap.png', comp)

            # anomaly_value 以上のところは少し赤くする
            det_mask = np.zeros((h, w))
            for a in range(h):
                for n in range(w):
                    if comp[a][n] >= anomaly_value:
                        real_real_cut[a, n, 2] = real_real_cut[a, n, 2] + 90
                        det_mask[a][n] = det_mask[a][n] + 255
            cv2.imwrite(save_dir_name + '/' + 'detection/' + (str)(i+1) + '_real_detection.png', (real_real_cut))
            cv2.imwrite(save_dir_name + '/' + 'mask/' + real_img_dir[i] + '_mask.png', (det_mask))
            F[int(s)-1] = f_value_cal(det_mask, gt_mask_cut, s, i, F[int(s)-1])

    # 最終的なF値を算出
    print()
    for an in range(len(step)):
        F_fin[an] = sum(F[an]) / len(F[an])
        F_fin[an] = round(F_fin[an], 3)
        print(f'{an + 1}段目の個別F値 : {F[an]}')
        print(f'{an + 1}段目の平均F値 : {F_fin[an]}')
        print('-----------------------------------')
    F1 = sum(F_fin) / len(F_fin)
    F1 = round(F1, 3)
    print(f'最終的なF値 : {F1}\n')




# F値算出プログラム
def f_value_cal(det_mask, gt_mask, step, i, F):

    height, width = det_mask.shape[:2]
    tp=0
    fp=0
    tn=0
    fn=0

    for m in range(height):
        for n in range(width):
            if gt_mask[m][n] == 255 and det_mask[m][n] == 255:
                tp = tp + 1
            elif gt_mask[m][n] == 0 and det_mask[m][n] == 255:
                fp = fp + 1
            elif gt_mask[m][n] == 0 and det_mask[m][n] == 0:
                tn = tn + 1
            elif gt_mask[m][n] == 255 and det_mask[m][n] == 0:
                fn = fn + 1
            else:
                print("何かミスしています")

    F_value = float(tp / (tp + 0.5*(fn + fp)))
    # 小数点以下 3桁 で丸める
    F_value = round(F_value, 3)
    # print(f'{step}段目の{i}枚目のF値 : {F_value}')
    F.append(F_value)
    print(f'F : {F}')

    return F


# depth で背景を黒塗りするプログラム
def depth_apply(depth, img):
    h, w = depth.shape[:2]

    for a in range(h):
        for b in range(w):
            if depth[a][b] >= 600 or depth[a][b] == 0:
                img[a][b] = 0.01
            else:
                img[a][b] = img[a][b]
    
    return img

    
if __name__ == '__main__':
    main()