import os
import cv2
import torch
import jpeg4py as jpeg
import torch.distributed as dist

from dataset.base_dataset import BaseDataset


class nyudepthv2(BaseDataset):
    def __init__(self, data_path, filenames_path='./dataset/filenames/',
                 is_train=True, crop_size=(384, 512), scale_size=None):
        super().__init__(crop_size)

        self.is_train = is_train

        self.image_path_list = []
        self.depth_path_list = []

        txt_path = os.path.join(filenames_path, 'scannet')
        if is_train:
            txt_path += '/train.txt'
        else:
            txt_path += '/valid.txt'
            # txt_path += '/test_list.txt'

        self.is_cache = False

        self.filenames_list = self.readTXT(txt_path)
        phase = 'train' if is_train else 'test'
        if dist.get_rank() == 0:
            print("Dataset: NYU Depth V2")
            print("# of %s images: %d" % (phase, len(self.filenames_list)))


    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.filenames_list[idx]
        gt_path = self.filenames_list[idx].replace('jpg','png').replace('color','depth')
        filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]
        focal_length_orig = None
        image = jpeg.JPEG(img_path).decode()
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32') / 1000.0
        focal_length_orig = 519 
        depth[depth > 20] = 0

        if self.is_train:
            image, depth, focal_length = self.preprocess(image, depth,is_training=True,focal_length_orig=focal_length_orig)
        else:
            # depth = depth[45:472,43:608]
            # image = image[45:472,43:608]
            # depth = depth[8:232,48:272]
            # image = image[8:232,48:272]
            image, depth, focal_length = self.preprocess(image, depth,focal_length_orig=focal_length_orig)

        focal_length = torch.Tensor([focal_length])

        return {'image': image, 'depth': depth, 'filename': filename, 'focal_length' : focal_length}


    def readTXT(self, txt_path):
        with open(txt_path, 'r') as f:
            listInTXT = [line.strip() for line in f]

        return listInTXT