import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np

class imagepath(Dataset):
    # for test only
    def __init__(self, data_path):
        super().__init__()

        self.data_path = data_path
        self.to_tensor = transforms.ToTensor()
        self.filenames_list = [os.path.join(data_path+'/camera/images/', i) for i in os.listdir(data_path+'/camera/images/')
                               if i.split('.')[-1] in ['png']]
        self.filenames_list.sort()
        self.filenames_list2 = [os.path.join(data_path+'/depth/images/', i) for i in os.listdir(data_path+'/depth/images/')
                               if i.split('.')[-1] in ['png']]
        self.filenames_list2.sort()

        print("Dataset : Image Path")
        print("# of images: %d" % (len(self.filenames_list)))
        self.fast = cv2.FastFeatureDetector.create(5)

    def __len__(self):
        return len(self.filenames_list)

    def getSparseDepth(self,image,depth,n_sample_kpt = None):
        sparse_kpts = list(self.fast.detect(image, (depth > 0).astype(np.uint8)))
        sparse_kpts.sort(key=lambda kp:kp.response,reverse=True)
        n_kpt = len(sparse_kpts)
        min_n = 125#int(1440 * (0.9648 ** self.epoch))
        max_n = 125#max(int(1600 * (0.97265 ** self.epoch)),n_sample_kpt)
        if(len(sparse_kpts) > max_n * 2):
            sparse_kpts = sparse_kpts[0:max_n * 2]
        if(n_kpt == 0):
            return None
        if n_sample_kpt is None:
            n_sample_kpt = n_kpt if n_kpt <= min_n else random.randint(min_n,min(max_n,n_kpt))
        else:
            n_sample_kpt = min(n_kpt,n_sample_kpt)

        sample_kpts = random.sample(sparse_kpts,n_sample_kpt)

        sample_pts = np.array([item.pt for item in sample_kpts])
        add_noise = True
        sample_pts_noise = (np.random.randn(sample_pts.shape[0],sample_pts.shape[1]) * 2. + sample_pts).astype(np.int32)
        sample_pts = sample_pts.astype(np.int32)
        sample_pts_noise[sample_pts_noise < 0] = 0
        sample_pts_noise[sample_pts_noise[:,0] >= depth.shape[1],0] = depth.shape[1] - 1
        sample_pts_noise[sample_pts_noise[:,1] >= depth.shape[0],1] = depth.shape[0] - 1

        sparse_depth = np.zeros_like(depth)
        if add_noise:
            sparse_depth[sample_pts[:,1],sample_pts[:,0]] = depth[sample_pts_noise[:,1],sample_pts_noise[:,0]] 
            sparse_depth[sample_pts[:,1],sample_pts[:,0]][sparse_depth[sample_pts[:,1],sample_pts[:,0]]==0] = depth[sample_pts[:,1],sample_pts[:,0]][sparse_depth[sample_pts[:,1],sample_pts[:,0]]==0]
            sparse_depth = sparse_depth + sparse_depth * np.random.randn(sparse_depth.shape[0],sparse_depth.shape[1]) * 0.05
        else:
            sparse_depth[sample_pts[:,1],sample_pts[:,0]] = depth[sample_pts[:,1],sample_pts[:,0]] 
            
        return sparse_depth

    def __getitem__(self, idx):
        batch = {}
        focal_length = 554
        file = self.filenames_list[idx]
        file1 = self.filenames_list2[idx]
        filename = file.split('/')[-1]
        
        image = cv2.imread(file)  # [H x W x C] and C: BGR
        # image = image.reshape(image.shape[0],image.shape[1],1)
        # print(file.replace('rgb','depth').replace('jpg','png'))
        sd = cv2.imread(file1,-1)  # [H x W x C] and C: BGR
        sd[sd==65535] = 0
        # sd = cv2.imread(file.replace('jpg','png').replace('rgb','depth'),-1)  # [H x W x C] and C: BGR
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        is_dense = (sd > 0).sum() > 3000
        
        # input size should be multiple of 32
        h, w, c = image.shape
        new_h, new_w = h // 32 * 32, w // 32 * 32
        # image = cv2.resize(image, (new_w, new_h))
        # sd = cv2.resize(sd, (new_w, new_h),cv2.INTER_NEAREST)
        sd = sd.astype(float) / 1000 /(focal_length / 519) / 10
        # image = image[::2,::2,:]
        # sd = sd[::2,::2]


        # sd = sd[8:232,12:364]
        depth = sd.copy()  * (focal_length / 519) * 10
        min_n = 40#int(1440 * (0.9648 ** self.epoch))
        max_n = 100#int(1600 * (0.97265 ** self.epoch))
        deterministic_rand = (sd>0).sum()%(max_n - min_n) + min_n
        if not is_dense:
            s1d = sd
        else:
            s1d = self.getSparseDepth(image,sd,n_sample_kpt = deterministic_rand)
            cv2.imwrite('logs/sd/' + file.split('/')[-1].replace('jpg','png'),(s1d >0) * 255)
        sd = s1d
        s2d = s1d#self.getSparseDepth(image,sd,n_sample_kpt = 1500)

        
        image = self.to_tensor(image)
        sd = self.to_tensor(sd) 
        s2d = self.to_tensor(s2d) 
        image[1,:,:] = sd
        image[2,:,:] = s2d

        batch['image'] = image
        batch['depth2solve'] = s2d
        batch['depth'] = self.to_tensor(depth)
        batch['filename'] = filename
        batch['focal_length'] = focal_length

        return batch
