import h5py
import numpy as np
import os
import cv2
import random
from imagededup.methods import PHash
import sys
phasher = PHash()

# root_path = '/mnt/afs0/chuguanyi/scannet_train/'
# out_path = '/mnt/afs1/Group/chuguanyi1/scannetDB/'
root_path = sys.argv[1] 
out_path = sys.argv[2] 

file_lists = root_path + '/filelist.txt'


all_file_images = []

with open(file_lists, 'r') as f:
    for line in f:
        all_file_images.append(line.strip())

# sort by filename which is the video id then frame id
all_file_images.sort(key=lambda x: (x.split('/')[-3], int(x.split('/')[-1].split('.')[0])))
sliding_window_img_vec = []
sliding_window_img_dict = {}
cnt = 0
for line in all_file_images:
    video_id = line.strip().split('/')[-3]
    filename = line.strip().split('/')[-1]
    filename = video_id + filename.replace('.jpg', '.h5')
    filename = out_path + filename
    if os.path.exists(filename):
        print('already exists', filename)
        continue

    img = cv2.imread(line.strip())
    depth = cv2.imread(line.strip().replace('color', 'depth').replace('jpg','png'), -1)
    # center_crop to 640 * 480 from 730 * 530
    if img is None or depth is None :
        print('invalid image', line.strip())
        continue

    valid_ratio = (depth < 1).sum() / (depth.shape[0] * depth.shape[1])
    if valid_ratio > 0.3:
        print('invalid depth', line.strip())
        continue
    lowdepth_ratio = (depth[depth>0] < 2000).sum() / (depth > 0).sum()
    if lowdepth_ratio > 0.75 and random.random() > 0.5:
        print('random throw', line.strip(), lowdepth_ratio)
        continue
    
    sliding_window_img_dict[line] = phasher.encode_image(image_array=img)
    image_vec = phasher.encode_image(image_array=img)
    if len(sliding_window_img_vec) > 30:
        sliding_window_img_vec.pop(0)
    skip = False
    for i in range(len(sliding_window_img_vec)):
        dup = phasher.hamming_distance(sliding_window_img_vec[i], image_vec)
        if dup < 8:
            skip = True
            break
    if skip:
        print('duplicate', line.strip())
        continue
    sliding_window_img_vec.append(image_vec)

    fx = 578.05

    # create h5 file
    f = h5py.File(filename, 'w')
    jpg = cv2.imencode('.jpg', img)[1]
    png = cv2.imencode('.png', depth)[1]
    f.create_dataset('rgb', data=jpg)
    f.create_dataset('depth', data=png)
    f.create_dataset('fx', data = fx)
    # done
    f.close()

    cnt += 1
    if cnt % 100 == 1:
        print(cnt)
        
