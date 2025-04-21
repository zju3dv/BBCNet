import os
import torch
import jpeg4py as jpeg

from dataset.base_dataset import BaseDataset
import numpy as np
import h5py
import cv2 

class lmdb_dataset(BaseDataset):
    def __init__(self, filenames_path='./dataset/filenames/',
                 is_train=True, crop_size=(384, 512), scale_size=None):
        super().__init__(crop_size)

        self.is_train = is_train

        self.image_path_list = []
        self.depth_path_list = []

        txt_path = filenames_path
        if is_train:
            # txt_path += '/train.txt'
            txt_path += '/filelists.txt'
            # txt_path += '/filelists-debug.txt'
        else:
            # txt_path += '/valid.txt'
            txt_path += '/nyu_val.txt'
            # txt_path += '/test_list.txt'

        self.is_cache = False


        self.filenames_list = self.readTXT(txt_path)
        phase = 'train' if is_train else 'test'
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))


    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.filenames_list[idx]
        with h5py.File(img_path,'r') as f:
            image = jpeg.JPEG(f['rgb'][:]).decode()
            depth = cv2.imdecode(f['depth'][:],-1)
            focal_length_orig = f['fx'][()]
        depth = depth / 1000
        depth[depth > 20] = 0

        image, depth, focal_length = self.preprocess(image, depth, is_training=self.is_train,focal_length_orig=focal_length_orig)

        focal_length = torch.Tensor([focal_length])
        return {'image': image, 'depth': depth, 'filename': img_path, 'focal_length' : focal_length}


    def readTXT(self, txt_path):
        with open(txt_path, 'r') as f:
            listInTXT = [line.strip() for line in f]

        return listInTXT