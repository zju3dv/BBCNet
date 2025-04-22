import os
import cv2
import torch
import jpeg4py as jpeg

from dataset.base_dataset import BaseDataset


class nyudepthv2(BaseDataset):
    def __init__(self, args, filenames_path='./dataset/filenames/nyudepthv2/',
                 is_train=True, crop_size=(384, 512), scale_size=None):
        super().__init__(args, crop_size)

        self.is_train = is_train

        self.image_path_list = []
        self.depth_path_list = []
        self.args = args

        txt_path = filenames_path
        if is_train:
            txt_path += '/filelists.txt'
        else:
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
        gt_path = self.filenames_list[idx].replace('jpg','png').replace('color','depth')
        filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]
        image = jpeg.JPEG(img_path).decode()
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32') / 1000.0
        focal_length_orig = 519 if 'nyudepthv2_image' in filename else 578
        depth[depth > self.args.max_depth] = 0

        image, depth, focal_length = self.preprocess(image, depth, is_training=self.is_train,focal_length_orig=focal_length_orig)

        focal_length = torch.Tensor([focal_length])
        return {'image': image, 'depth': depth, 'filename': filename, 'focal_length' : focal_length}


    def readTXT(self, txt_path):
        with open(txt_path, 'r') as f:
            listInTXT = [line.strip() for line in f]

        return listInTXT
