import random
import importlib
import albumentations as A
from torch.utils.data import Dataset
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

class BaseDataset(Dataset):
    def __init__(self, crop_size):
        
        self.count = 0
        linear_transform = [
            A.geometric.RandomScale(scale_limit=(-0.1,0.2), interpolation=cv2.INTER_AREA, always_apply=True),
            A.geometric.Rotate(limit=25,interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_CONSTANT,value=0),
        ]
        self.linear_transform = linear_transform    
        basic_transform = [
            A.HorizontalFlip(),
            A.RandomCrop(crop_size[0], crop_size[1])
        ]
        brightness_transform = [
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue(),
            A.GaussNoise()
        ]
        self.basic_transform = basic_transform    
        self.brightness_transform = brightness_transform    
        self.to_tensor = transforms.ToTensor()
        self.epoch = 0

    def getSparseDepth(self,image,depth,n_sample_kpt = None):
        all_kpts = list(fast_detector.detect(image, (depth > 0).astype(np.uint8)))
        all_kpts.sort(key=lambda kp:kp.response,reverse=True)
        n_kpt = len(all_kpts)
        min_n = 100#max(60,60 + 6*(10-self.epoch))#int(1440 * (0.9648 ** self.epoch))
        max_n = 150#max(120,120 + 12*(10-self.epoch))#int(1600 * (0.97265 ** self.epoch))
        if(len(all_kpts) > max_n * 2):
            sparse_kpts = all_kpts[0:max_n * 2]
        else:
            sparse_kpts = all_kpts
        if(n_kpt == 0):
            return np.zeros_like(depth),np.zeros_like(depth)
        if n_sample_kpt is None:
            n_sample_kpt = n_kpt if n_kpt <= min_n else random.randint(min_n,min(max_n,n_kpt))
        else:
            n_sample_kpt = min(n_kpt,n_sample_kpt)

        sample_kpts = random.sample(sparse_kpts,n_sample_kpt)

        sample_pts = np.array([item.pt for item in sample_kpts])
        sample_pts_noise = (np.random.randn(sample_pts.shape[0],sample_pts.shape[1]) * 3. + sample_pts).astype(np.int32)
        sample_pts = sample_pts.astype(np.int32)
        sample_pts_noise[sample_pts_noise < 0] = 0
        sample_pts_noise[sample_pts_noise[:,0] >= depth.shape[1],0] = depth.shape[1] - 1
        sample_pts_noise[sample_pts_noise[:,1] >= depth.shape[0],1] = depth.shape[0] - 1

        sparse_depth = np.zeros_like(depth)

        ADD_NOISE = True#True if self.is_training else False
        noise_level = 0.1
        if ADD_NOISE:
            sparse_depth[sample_pts[:,1],sample_pts[:,0]] = depth[sample_pts_noise[:,1],sample_pts_noise[:,0]] 
            gtd = np.zeros_like(depth)
            gtd[sample_pts[:,1],sample_pts[:,0]] = depth[sample_pts[:,1],sample_pts[:,0]]
            sparse_depth[sparse_depth==0] = gtd[sparse_depth==0]
        else:
            sparse_depth[sample_pts[:,1],sample_pts[:,0]] = depth[sample_pts[:,1],sample_pts[:,0]] 
            noise_level = 0
        sparse_depth = sparse_depth + sparse_depth * np.random.randn(sparse_depth.shape[0],sparse_depth.shape[1]) * noise_level
        
        USE_DIFF_KPTS = True 
        if USE_DIFF_KPTS:
            sample_kpts = random.sample(all_kpts,n_sample_kpt)
            sample_pts = np.array([item.pt for item in sample_kpts]).astype(np.int32)
        gt_sd = np.zeros_like(depth)
        gt_sd[sample_pts[:,1],sample_pts[:,0]] = depth[sample_pts[:,1],sample_pts[:,0]] 
        return sparse_depth,gt_sd

    # def getSparseDepth(self,image,depth,n_sample_kpt = None):
    #     mask = depth.reshape(-1) > 0
    #     rdepth = depth.reshape(-1)
    #     n_sample_kpt = 500
    #     all_points_idx = mask.nonzero()[0]
    #     p_len = len(all_points_idx)
    #     n_sample_kpt = min(n_sample_kpt,p_len)
    #     sample_inx = np.random.permutation(all_points_idx)[0:n_sample_kpt]
    #     mask_new = np.zeros_like(rdepth)
    #     mask_new[sample_inx] = 1
    #     mask_new = mask_new.reshape(depth.shape[0],depth.shape[1])
    #     sparse_depth = depth * mask_new.astype(np.uint8)

    #     return sparse_depth,sparse_depth


    def preprocess(self, image, depth,is_training = False,focal_length_orig = None):
        scale_factor = 1
        self.is_training = is_training
        deterministic_rand = None
        if is_training:
            imask = (depth < 1e-3).astype('uint8') * 127
            origin_h = image.shape[1]
            additional_targets = {'depth': 'mask', 'imask':'mask'}
            aug0 = A.Compose(transforms=self.linear_transform,
                            additional_targets=additional_targets)
            augmented = aug0(image=image, depth=depth,imask = imask)
            image = augmented['image']
            imask = augmented['imask']
            depth = augmented['depth']
            imask = imask > 0
            depth[imask] = 0
            resized_h = image.shape[1]
            scale_factor = resized_h / origin_h
            depth = depth
            augmented=None

            additional_targets = {'depth': 'mask'}
            aug1 = A.Compose(transforms=self.basic_transform,
                            additional_targets=additional_targets)
            augmented = aug1(image=image, depth=depth)
            image = augmented['image']
            depth = augmented['depth']
            augmented=None
            
            aug2 = A.Compose(transforms=self.brightness_transform)
            augmented = aug2(image=image)
            image = augmented['image']
            self.count += 1
        else:
            deterministic_rand = 125
        focal_length = focal_length_orig * scale_factor

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sparse_depth,gt_depth = self.getSparseDepth(gray,depth,deterministic_rand)
        if((sparse_depth>0).sum() < 18):
            sparse_depth[::15,::15] = depth[::15,::15] 
            gt_depth[::15,::15] = depth[::15,::15] 
        if is_training:
            H, W, C = image.shape

            if self.count % 4 == 0:
                alpha = random.random()
                beta = random.random()
                p = 0.75
                l = int(alpha * W)
                w = int(max((W - alpha * W) * beta * p, 1))
                gray[:, l:l+w] = depth[:, l:l+w]/10 * int((random.random() * 0.2 + 0.8) * 255)
        else:
            USE_DIFF_KPTS_FOR_TEST = False
            if not USE_DIFF_KPTS_FOR_TEST:
                gt_depth = sparse_depth
            

        image[:,:,0] = gray
        image[:,:,1] = 0#np.random.randn(*gray.shape)
        image[:,:,2] = 0

        image = self.to_tensor(image)
        depth = self.to_tensor(depth).squeeze()
        if sparse_depth is not None:
            sparse_depth = self.to_tensor(sparse_depth).squeeze()
            sparse_depth = sparse_depth  /(focal_length / 519) / 10
            image[1,:,:] = sparse_depth

            gt_depth = self.to_tensor(gt_depth).squeeze()
            gt_depth = gt_depth  /(focal_length / 519) / 10
            image[2,:,:] = gt_depth

        return image, depth, focal_length # depth is depth/focal_length in fact
