__author__ = 'taniguchi'
import numpy as np
import os
import cv2
from PIL import Image
import logging
import random
from typing import Iterable, List
from dataclasses import dataclass

from rich.progress import track

logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, datas, indices, frames_products_mark, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.batch_size = input_param['batch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.curent_batch_indices = []
        self.current_input_length = input_param['seq_length']
        self.frames_products_mark = frames_products_mark

    # 3
    def total(self):
        return len(self.indices)
    
    # 1
    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.batch_size]

    def next(self):
        self.current_position += self.batch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.batch_size]

    def next_test(self):
        self.current_position += self.batch_size
        if self.no_batch_left_test():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.batch_size]

    # 2
    def no_batch_left(self):
        if self.current_position + self.batch_size >= self.total():
            return True
        else:
            return False
    
    # 2
    def no_batch_left_test(self):
        if self.current_position + self.batch_size > self.total(): ########## >= を > に変更
            return True
        else:
            return False
        
    # 4
    def get_batch(self):
        if self.no_batch_left(): # no_batch_left が True なら エラーを出す
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        # Create batch of N videos of length L, i.e. shape = (N, L, w, h, c)
        # where w x h is the resolution and c the number of color channels
        # 要素が０の５次元配列を作成
        input_batch = np.zeros(
            (self.batch_size, self.current_input_length, self.image_width, self.image_width, 3)).astype(
            self.input_data_type)
        for i in range(self.batch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length # seq_len 20
            data_slice = self.datas[begin:end, :, :, :] # self.datas[画像枚数, wid?, hei?, ch?]
            # print('data_slice :', data_slice.shape)  (20, 128, 128, 3)
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            # print('input_batch :', input_batch.shape)  (4, 20, 128, 128, 3)
            
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch
    
    def get_batch_test(self):
        if self.no_batch_left_test(): # no_batch_left が True なら エラーを出す
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        # Create batch of N videos of length L, i.e. shape = (N, L, w, h, c)
        # where w x h is the resolution and c the number of color channels
        # 要素が０の５次元配列を作成
        input_batch = np.zeros(
            (self.batch_size, self.current_input_length, self.image_width, self.image_width, 3)).astype(
            self.input_data_type)
        for i in range(self.batch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length # seq_len 20
            data_slice = self.datas[begin:end, :, :, :] # self.datas[画像枚数, wid?, hei?, ch?]
            # print('data_slice :', data_slice.shape)  (20, 128, 128, 3)
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            # print('input_batch :', input_batch.shape)  (4, 20, 128, 128, 3)
            
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    batch Size: " + str(self.batch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))


@dataclass
class FrameInfo:
    file_name: str
    file_path: str
    products_mark: int


class DataProcess:
    def __init__(self, configs):
        self.configs = configs
        self.train_data_path = configs['train_data_paths']
        self.valid_data_path = configs['valid_data_paths']
        self.test_data_path = configs['test_data_paths']
        self.image_width = configs['image_width']
        self.seq_len = configs['seq_length']
        self.train_val_pro = ['train_valid']
        self.test_pro = [configs['test_dir_name']]
        self.step = ['1', '2', '3', '4', '5', '6']
        self.train_products = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                               '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70']
        self.valid_products = ['10', '60']
        self.test_products = ['99', '98', '97', '96', '95', '94', '93', '92', '91', '90', '89', '88', '87', '86', '85', '84', '83', '82', '81', '80']############################################################

    def generate_frames(self, root_path, product_ids, mode: List[int]) -> Iterable[FrameInfo]:
        products_mark = 0
        if mode == 'train' or mode == 'valid':
            train_val_dir = os.path.join(root_path, self.train_val_pro[0])###############################33
            for step_dir in self.step:
                step_dir_path = os.path.join(train_val_dir, step_dir)
                step_videos = os.listdir(step_dir_path)
                step_videos = sorted(step_videos)
                products_mark += 1

                for products_direction_video in step_videos:
                    product_id = products_direction_video[0:2]
                    # print('product_id :', product_id, '---------------')
                    if product_id not in product_ids:
                        continue
                    # products_mark += 1
                    yield FrameInfo(
                        file_name = products_direction_video,
                        file_path = os.path.join(step_dir_path, products_direction_video),
                        products_mark = products_mark
                    )
        elif mode == 'test':################################################################
            test_dir_path = os.path.join(root_path, self.test_pro[0])
            for step_dir in self.step:
                step_dir_path = os.path.join(test_dir_path, step_dir)
                step_videos = os.listdir(step_dir_path)
                step_videos = sorted(step_videos)
                products_mark += 1

                for products_direction_video in step_videos:
                    product_id = products_direction_video[0:2]
                    # print('product_id :', product_id, '---------------')
                    if product_id not in product_ids:
                        continue
                    # products_mark += 1
                    yield FrameInfo(
                        file_name = products_direction_video,
                        file_path = os.path.join(step_dir_path, products_direction_video),
                        products_mark = products_mark
                    )
        else:
            raise Exception("Unexpected mode(def generate_frames) :" + mode)

    def load_data(self, paths, mode='train'):
        path = paths[0]
        if mode == 'train':
            mode_products_ids = self.train_products
        elif mode == 'valid':
            mode_products_ids = self.valid_products
        elif mode == 'test':
            mode_products_ids = self.test_products###########################################
        else:
            raise Exception("Unexpected mode(def load_data) : " + mode)
        print('begin load data' + str(path))

        frames_file_name = []
        frames_products_mark = []

        tot_num_frames = sum((1 for _ in self.generate_frames(path, mode_products_ids, mode)))
        print(f"Preparing to load {tot_num_frames} video frames.")

        data = np.empty((tot_num_frames, self.image_width, self.image_width, 3),
                        dtype = np.float32)

        for i, frame in enumerate(track(self.generate_frames(path, mode_products_ids, mode))):
            # print('i :', i, 'frame :', frame, '-------------------')　　　i : 112  frame : FrameInfo(file_name='9913.png', file_path='/data/products_change1/start/6/9913.png', products_mark=6)
            frame_im = Image.open(frame.file_path)
            frame_np = np.array(frame_im, dtype=np.float32)
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)#################################################追加
            data[i,:,:,:] = (cv2.resize(frame_np, (self.image_width, self.image_width))/255)
            # print('data.shape', data.shape, '--------------')  (587, 128, 128, 3)

            frames_file_name.append(frame.file_name)
            frames_products_mark.append(frame.products_mark)
            # print(frames_products_mark)
        
        indices = []
        seq_end_idx = len(frames_products_mark) - 1
        # print('seq_end_idx :', seq_end_idx, '------------------------------')
        while seq_end_idx >= self.seq_len - 1:
            seq_start_idx = seq_end_idx - self.seq_len + 1
            if frames_products_mark[seq_end_idx] == frames_products_mark[seq_start_idx]:
                end = int(frames_file_name[seq_end_idx][2:4])
                start = int(frames_file_name[seq_start_idx][2:4])
                # print('frames_products_mark[seq_end_idx] :', frames_products_mark[seq_end_idx], '----------------------')
                # print('frames_products_mark[seq_start_idx] :', frames_products_mark[seq_start_idx], '-----------------------------')
                # print('end :', end, '----------------')
                # print('start :', start, '-------------------------')
                # print()

                if end - start == self.seq_len - 1:
                    indices.append(seq_start_idx)
            
            seq_end_idx -= 2 # 次の20枚を取得するのに 3枚 ずらして取得する

        print("there are " + str(data.shape[0]) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return data, indices, frames_products_mark
    
    def load_test_data(self, paths, mode='train'):
        path = paths[0]
        if mode == 'train':
            mode_products_ids = self.train_products
        elif mode == 'valid':
            mode_products_ids = self.valid_products
        elif mode == 'test':
            mode_products_ids = self.test_products###########################################
        else:
            raise Exception("Unexpected mode(def load_data) : " + mode)
        print('begin load data' + str(path))

        frames_file_name = []
        frames_products_mark = []

        tot_num_frames = sum((1 for _ in self.generate_frames(path, mode_products_ids, mode)))
        print(f"Preparing to load {tot_num_frames} video frames.")

        data = np.empty((tot_num_frames, self.image_width, self.image_width, 3),
                        dtype = np.float32)

        for i, frame in enumerate(track(self.generate_frames(path, mode_products_ids, mode))):
            # print('i :', i, 'frame :', frame, '-------------------')    i : 112  frame : FrameInfo(file_name='9913.png', file_path='/data/products_change1/start/6/9913.png', products_mark=6)
            # print('i :', i, 'frame :', frame, '-------------------')
            # print(frame.products_mark)

            frame_im = Image.open(frame.file_path)
            frame_np = np.array(frame_im, dtype=np.float32)
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)#################################################追加
            data[i,:,:,:] = (cv2.resize(frame_np, (self.image_width, self.image_width))/255)
            # print('data.shape', data.shape, '--------------')  (587, 128, 128, 3)

            frames_file_name.append(frame.file_name)
            frames_products_mark.append(frame.products_mark)
            # print(frames_products_mark)    [1,1,1,1,1,1,.....1,2,2,2,2,2,...2,......6,6,6,6,6,6,6,6,6]
        
        indices = []
        seq_end_idx = len(frames_products_mark) - 1
        # print('seq_end_idx :', seq_end_idx, '------------------------------')    119
        seq_start_idx = 0

        # while seq_end_idx >= self.seq_len - 1:
        #     seq_start_idx = seq_end_idx - self.seq_len + 1
        #     if frames_products_mark[seq_end_idx] == frames_products_mark[seq_start_idx]:
        #         end = int(frames_file_name[seq_end_idx][2:4])
        #         start = int(frames_file_name[seq_start_idx][2:4])
        #         if end - start == self.seq_len - 1:
        #             indices.append(seq_start_idx)
        #             print(indices)
        #     seq_end_idx -= self.seq_len
        # print('seq_end_idx : ', seq_end_idx)
        while seq_end_idx > seq_start_idx:
            # print('seq_start_idx : ' , seq_start_idx)
            # print(f'frames_products_mark[seq_start_idx+self.seq_len-1]:{frames_products_mark[seq_start_idx+self.seq_len-1]}  frames_products_mark[seq_start_idx]:{frames_products_mark[seq_start_idx]}')
            if frames_products_mark[seq_start_idx+self.seq_len-1] == frames_products_mark[seq_start_idx]:
                end = int(frames_file_name[seq_start_idx+self.seq_len-1][2:4])
                start = int(frames_file_name[seq_start_idx][2:4])
                # print(f'end:{end}   start:{start}')
                if end - start == self.seq_len - 1:
                    indices.append(seq_start_idx)
                    print(indices)
            seq_start_idx += self.seq_len

        print("there are " + str(data.shape[0]) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return data, indices, frames_products_mark
    
    def get_train_input_handle(self):
        train_data, train_indices, frames_products_mark = self.load_data(self.train_data_path, mode='train')
        return InputHandle(train_data, train_indices, frames_products_mark, self.configs)
    
    def get_test_input_handle(self):
        test_data, test_indices, test_frames_products_mark = self.load_data(self.valid_data_path, mode='valid')
        return InputHandle(test_data, test_indices, test_frames_products_mark , self.configs)
    
    def get_test_test_input_handle(self):
        test_test_data, test_test_indices, test_test_frames_products_mark = self.load_test_data(self.test_data_path, mode='test')##########################333
        return InputHandle(test_test_data, test_test_indices, test_test_frames_products_mark, self.configs)
    
    def get_frames_products_mark(self):
        test_test_data, test_test_indices, test_test_frames_products_mark = self.load_test_data(self.test_data_path, mode='test')##########################333
        return test_test_frames_products_mark