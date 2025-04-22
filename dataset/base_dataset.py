import random
import importlib
import albumentations as A
from torch.utils.data import Dataset
from albumentations.core.transforms_interface import ImageOnlyTransform
import torchvision.transforms as transforms
import cv2
import numpy as np

def get_dataset(dataset_name, **kwargs):
    dataset_name = dataset_name.lower()
    dataset_lib = importlib.import_module(
        '.' + dataset_name, package='dataset')
    dataset_abs = getattr(dataset_lib, dataset_name)

    return dataset_abs(**kwargs)

fast_detector = cv2.FastFeatureDetector.create(5)

class ToGrayCustom(ImageOnlyTransform):

    def apply(self, img, **params):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def get_transform_init_args_names(self):
        return ()

class BaseDataset(Dataset):
    def __init__(self, args, crop_size):

        self.args = args 
        self.count = 0
        if args.with_confidence:
            linear_transform = [
                A.RandomScale(scale_limit=(-0.05,0.15), interpolation=cv2.INTER_AREA, always_apply=True),
                A.Rotate(limit=25,interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_CONSTANT,value=0),
                # A.ShiftScaleRotate
            ]
        else:
            linear_transform = [
                # A.RandomScale(scale_limit=(-0.05,0.15), interpolation=cv2.INTER_AREA, always_apply=True),
                A.Rotate(limit=25,interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_CONSTANT,value=0),
                # A.ShiftScaleRotate
            ]
        self.linear_transform = linear_transform    
        if not self.args.close_aug:
            if self.args.with_confidence:
                test_transform = [
                    # A.RandomCrop(crop_size[0], crop_size[1]),
                    A.CenterCrop(crop_size[0], crop_size[1]),
                    A.HorizontalFlip(),
                ]
            else:
                basic_transform = [
                    A.RandomCrop(crop_size[0], crop_size[1]),
                    # A.CenterCrop(crop_size[0], crop_size[1]),
                    A.HorizontalFlip(),
                ]
        else:
            basic_transform = [
                # A.RandomCrop(crop_size[0], crop_size[1]),
                A.CenterCrop(crop_size[0], crop_size[1]),
                A.HorizontalFlip(),
            ]
        brightness_transform = [
            A.HueSaturationValue(),
            ToGrayCustom(always_apply=True),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.GaussNoise()
        ]
        self.basic_transform = basic_transform
        self.test_transform = test_transform
        self.brightness_transform = brightness_transform    
        self.to_tensor = transforms.ToTensor()
        self.epoch = 0

    def getSparseDepth(self,image,depth,n_sample_kpt = None):
        all_kpts = list(fast_detector.detect(image, ((depth > 0) & (depth < self.args.max_depth)).astype(np.uint8)))
        n_kpt = len(all_kpts)
        if(n_kpt < 18):
            return np.zeros_like(depth),np.zeros_like(depth)
        all_kpts.sort(key=lambda kp:kp.response,reverse=True)
        # min_n = max(50,50 + 50*(10-self.epoch))#int(1440 * (0.9648 ** self.epoch))
        min_n = 50#max(50,60 + 6*(10-self.epoch))#int(1440 * (0.9648 ** self.epoch))
        max_n = 200#max(120,120 + 12*(10-self.epoch))#int(1600 * (0.97265 ** self.epoch))
        # max_n = max(200,200 + 50*(10-self.epoch))#int(1600 * (0.97265 ** self.epoch))
        
        if(n_kpt > max_n * 2):
            sparse_kpts = all_kpts[0:max_n * 2]
        else:
            sparse_kpts = all_kpts

        if not self.is_training:
            n_sample_kpt = min(n_kpt,n_sample_kpt)
        else:
            n_sample_kpt = n_kpt if n_kpt <= min_n else random.randint(min_n,min(max_n,n_kpt))

        sample_kpts = random.sample(sparse_kpts,n_sample_kpt)

        sample_pts = np.array([item.pt for item in sample_kpts])
        sample_pts_noise = (np.random.randn(sample_pts.shape[0],sample_pts.shape[1]) * 2. + sample_pts).astype(np.int32)
        sample_pts = sample_pts.astype(np.int32)
        sample_pts_noise[sample_pts_noise < 0] = 0
        sample_pts_noise[sample_pts_noise[:,0] >= depth.shape[1],0] = depth.shape[1] - 1
        sample_pts_noise[sample_pts_noise[:,1] >= depth.shape[0],1] = depth.shape[0] - 1

        sparse_depth = np.zeros_like(depth)

        # ADD_NOISE = True #True if self.is_training else False
        noise_level = self.args.depth_noise_level
        if self.args.add_noise:
            # mask = depth[sample_pts_noise[:,1],sample_pts_noise[:,0]] > 0
            # sparse_depth[sample_pts[:,1],sample_pts[:,0]] = depth[sample_pts[:,1],sample_pts[:,0]] * (~mask).astype(np.uint8) \
            #     + depth[sample_pts_noise[:,1],sample_pts_noise[:,0]] * mask.astype(np.uint8)
            
            sparse_depth[sample_pts_noise[:,1],sample_pts_noise[:,0]] = depth[sample_pts[:,1],sample_pts[:,0]] 
            sd_val = sparse_depth[sparse_depth>0]
            sparse_depth[sparse_depth > 0] += sd_val * np.random.randn(sd_val.shape[0]) * noise_level
            # sparse_depth = sparse_depth + sparse_depth * 
        else:
            sparse_depth[sample_pts[:,1],sample_pts[:,0]] = depth[sample_pts[:,1],sample_pts[:,0]] 
        
        # USE_DIFF_KPTS = True 
        if self.is_training and self.args.use_diff_kpts:
            sample_kpts = random.sample(all_kpts,n_sample_kpt)
            sample_pts = np.array([item.pt for item in sample_kpts]).astype(np.int32)
            gt_sd = np.zeros_like(depth)
            gt_sd[sample_pts[:,1],sample_pts[:,0]] = depth[sample_pts[:,1],sample_pts[:,0]] 
        else:
            gt_sd = sparse_depth
        return sparse_depth,gt_sd

    def getSparseDepthRandom(self,depth,n_sample_kpt = None):
        mask = depth.reshape(-1) > 0
        rdepth = depth.reshape(-1)
        all_points_idx = mask.nonzero()[0]
        p_len = len(all_points_idx)
        n_sample_kpt = min(n_sample_kpt,p_len)
        sample_inx = np.random.permutation(all_points_idx)[0:n_sample_kpt]
        mask_new = np.zeros_like(rdepth)
        mask_new[sample_inx] = 1
        mask_new = mask_new.reshape(depth.shape[0],depth.shape[1])
        sparse_depth = depth * mask_new.astype(np.uint8)
        sparse_depth = sparse_depth
        if self.args.add_noise:
            sparse_depth += sparse_depth * np.random.randn(sparse_depth.shape[0],sparse_depth.shape[1]) * self.args.depth_noise_level

        return sparse_depth,sparse_depth


    def preprocess(self, image, depth,is_training = False,focal_length_orig = None):
        scale_factor = 1
        self.is_training = is_training
        deterministic_rand = self.args.points_num 
        if is_training:
            if not self.args.close_aug:
                aug2 = A.Compose(transforms=self.brightness_transform)
                augmented = aug2(image=image)
                image = augmented['image']
            self.count += 1

            imask = (depth > 0).astype(np.float32)  #(depth < 1e-3).astype('uint8') * 127
            origin_h = image.shape[1]
            # additional_targets = {'depth': 'mask'} # for mask type, neareast interpolation is used
            additional_targets = {'depth': 'image','imask' :'image'} # for mask type, neareast interpolation is used
            if not self.args.close_aug:
                aug0 = A.Compose(transforms=self.linear_transform,
                                additional_targets=additional_targets)
                augmented = aug0(image=image, depth=depth, imask = imask)
                image = augmented['image']
                imask = augmented['imask']
                depth = augmented['depth']
            mask = imask  > 0
            depth[mask] /= imask[mask]
            # depth[imask > 0] = 0
            resized_h = image.shape[1]
            scale_factor = resized_h / origin_h

            additional_targets = {'depth': 'mask'}
            aug1 = A.Compose(transforms=self.basic_transform,
                            additional_targets=additional_targets)
            augmented = aug1(image=image, depth=depth)
            image = augmented['image']
            depth = augmented['depth']
        else:
            self.count += 1
            additional_targets = {'depth': 'mask'}
            aug1 = A.Compose(transforms=self.test_transform,
                            additional_targets=additional_targets)
            augmented = aug1(image=image, depth=depth)
            image = augmented['image']
            depth = augmented['depth']

        focal_length = focal_length_orig * scale_factor
        # cv2.imwrite(f'test_image_aug-{self.count}.png', image)
        # gray = image if self.is_training else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # if image is rgb, then convert to gray
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # if self.is_training:
        #     cv2.imwrite(f'train_image_aug-{self.count}.png', gray)
        # else:
        #     cv2.imwrite(f'test_image_aug-{self.count}.png', gray)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.args.use_fast_corner:
            sparse_depth, gt_depth = self.getSparseDepth(gray, depth, deterministic_rand)
            if((sparse_depth>0).sum() < 18):
                sparse_depth, gt_depth = self.getSparseDepthRandom(depth, random.randint(15,40))
        else:
            sparse_depth, gt_depth = self.getSparseDepthRandom(depth, deterministic_rand)

        H = image.shape[0]
        W = image.shape[1]
        if is_training:
            if self.count % 4 == 0:
                alpha = random.random()
                beta = random.random()
                p = 0.75
                l = int(alpha * W)
                w = int(max((W - alpha * W) * beta * p, 1))
                gray[:, l:l+w] = depth[:, l:l+w]/10 * int((random.random() * 0.2 + 0.8) * 255)
            
        depth_factor = (focal_length / 519) * 10
        image = np.zeros((H,W,3),np.float32)
        image[:,:,0] = gray.astype(np.float32) / 255
        image[:,:,1] = sparse_depth / depth_factor
        image[:,:,2] = gt_depth / depth_factor

        image = self.to_tensor(image)
        depth = self.to_tensor(depth)
        
        return image, depth, focal_length # depth is depth/focal_length in fact

