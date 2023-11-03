import os.path
import datetime
import cv2
import numpy as np
# from skimage.measure import compare_ssim
# import skimage.metrics.structural_similarity
# from skimage.measure import structural_similarity
# from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
# from skimage import metrics
from core.utils import preprocess, metrics
from core.data_provider import products
#from utils import preprocess, metrics
import lpips
import torch

loss_fn_alex = lpips.LPIPS(net='alex')


def train(model, ims, real_input_flag, configs, itr):
    cost = model.train(ims, real_input_flag)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        cost += model.train(ims_rev, real_input_flag)
        cost = cost / 2

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))


def test(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'validation...')
    test_input_handle.begin(do_shuffle=True)############################################### シャッフルを False -> True に変更
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    lp = []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    # reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    while (test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        test_ims = test_ims[:, :, :, :, :configs.img_channel]
        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length 
        img_out = img_gen[:, -output_length:]

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            ################################################################################################################
            #print(gx)
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            ########################################################################################
            #print(gx)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            # cal lpips
            img_x = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 1]
                img_x[:, 2, :, :] = x[:, :, :, 2]
            else:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 0]
                img_x[:, 2, :, :] = x[:, :, :, 0]
            img_x = torch.FloatTensor(img_x)
            img_gx = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 1]
                img_gx[:, 2, :, :] = gx[:, :, :, 2]
            else:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 0]
                img_gx[:, 2, :, :] = gx[:, :, :, 0]
            img_gx = torch.FloatTensor(img_gx)
            lp_loss = loss_fn_alex(img_x, img_gx)
            lp[i] += torch.mean(lp_loss).item()

            real_frm = np.uint8(x * 255)
            ##########################################################################################################################################################333
            pred_frm = np.uint8(gx * 255)
            # print(x)
            # print(gx)


            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            for b in range(configs.batch_size):
                #score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, channel_axis=-1)
                ssim[i] += score

        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            print('path :',path)
            os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(output_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        test_input_handle.next()

    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])

    lp = np.asarray(lp, dtype=np.float32) / batch_id
    print('lpips per frame: ' + str(np.mean(lp)))
    for i in range(configs.total_length - configs.input_length):
        print(lp[i])

# 自作 test 専用のプログラム
def test_test(model, test_input_handle, frames_products_mark, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    # print(frames_products_mark)   [1,1,1,1,1,1...1,2,2,2,2,2,2,...,.....6,6,6]
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    dir_id = 0##############################33
    img_mse, ssim, psnr = [], [], []
    lp = []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    # reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0
    while (test_input_handle.no_batch_left_test() == False):
        dir_id = dir_id + 1########################################
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch_test()
        # test_ims は (4, 20, 128, 128, 3)
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        # test_dat は (4, 20, 32, 32, 16)
        test_ims = test_ims[:, :, :, :, :configs.img_channel]
        # print('test_ims :', test_ims)
        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length 
        img_out = img_gen[:, -output_length:]

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            ################################################################################################################
            #print(gx)
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            ########################################################################################
            #print(gx)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            # cal lpips
            img_x = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 1]
                img_x[:, 2, :, :] = x[:, :, :, 2]
            else:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 0]
                img_x[:, 2, :, :] = x[:, :, :, 0]
            img_x = torch.FloatTensor(img_x)
            img_gx = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 1]
                img_gx[:, 2, :, :] = gx[:, :, :, 2]
            else:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 0]
                img_gx[:, 2, :, :] = gx[:, :, :, 0]
            img_gx = torch.FloatTensor(img_gx)
            lp_loss = loss_fn_alex(img_x, img_gx)
            lp[i] += torch.mean(lp_loss).item()

            real_frm = np.uint8(x * 255)
            ##########################################################################################################################################################333
            pred_frm = np.uint8(gx * 255)
            # print(x)
            # print(gx)


            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            for b in range(configs.batch_size):
                #score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, channel_axis=-1)
                ssim[i] += score

        # save prediction examples
        if batch_id <= configs.num_save_samples:
            # path = os.path.join(res_path, str(configs.num_save_samples + 1 - batch_id))
            path = os.path.join(res_path, str(batch_id))
            print('path :',path)
            os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(output_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)

        # if batch_id <= configs.num_save_samples:
        #     end = len(frames_products_mark) - 1
        #     start = 0
        #     cnt = [0,0,0,0,0,0]
        #     cnt1 = 0
        #     cnt2 = 0
        #     cnt3 = 0
        #     cnt4 = 0
        #     cnt5 = 0
        #     cnt6 = 0
        #     count = [0,0,0,0,0,0]
        #     count1 = 0
        #     count2 = 0
        #     count3 = 0
        #     count4 = 0
        #     count5 = 0
        #     count6 = 0
        #     while start!=end:
        #         if frames_products_mark[start] == 1:
        #             cnt[0] += 1
        #             if cnt[0]%20==0:
        #                 count[0] += 1
        #             start += 1
        #         elif frames_products_mark[start] == 2:
        #             cnt[1] += 1
        #             if cnt[1]%20==0:
        #                 count[1] += 1
        #             start += 1
        #         elif frames_products_mark[start] == 3:
        #             cnt[2] += 1
        #             if cnt[2]%20==0:
        #                 count[2] += 1
        #             start += 1
        #         elif frames_products_mark[start] == 4:
        #             cnt[3] += 1
        #             if cnt[3]%20==0:
        #                 count[3] += 1
        #             start += 1
        #         elif frames_products_mark[start] == 5:
        #             cnt[4] += 1
        #             if cnt[4]%20==0:
        #                 count[4] += 1
        #             start += 1
        #         elif frames_products_mark[start] == 6:
        #             cnt[5] += 1
        #             if cnt[5]%20==0:
        #                 count[5] += 1
        #             start += 1
        #         else:
        #             print('framse_products_mark に存在してはいけない値が含まれています')
 

        #     path = os.path.join(res_path, str(batch_id))
        #     print('path :',path)
        #     os.mkdir(path)
        #     for i in range(configs.total_length):
        #         name = str(num+1) + 'gt' + str(i + 1) + '.png'
        #         file_name = os.path.join(path, name)
        #         img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
        #         cv2.imwrite(file_name, img_gt)
        #     for i in range(output_length):
        #         name = str(num+1) + 'pd' + str(i + 1 + configs.input_length) + '.png'
        #         file_name = os.path.join(path, name)
        #         img_pd = img_out[0, i, :, :, :]
        #         img_pd = np.maximum(img_pd, 0)
        #         img_pd = np.minimum(img_pd, 1)
        #         img_pd = np.uint8(img_pd * 255)
        #         cv2.imwrite(file_name, img_pd)

        test_input_handle.next_test()

    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])

    lp = np.asarray(lp, dtype=np.float32) / batch_id
    print('lpips per frame: ' + str(np.mean(lp)))
    for i in range(configs.total_length - configs.input_length):
        print(lp[i])