import cv2
import numpy as np

with open('/mnt/cache/chuguanyi1/Workspace/dense_depth/dataset/filenames/nyudepthv2/train_list.txt') as f:
    cnt = 0
    for line in f:
        line = line.strip()
        img = cv2.imread(line)
        path = line.replace('color','depth').replace('jpg','png')
        depth = cv2.imread(path,-1)
        img = cv2.resize(img,(320,240),cv2.INTER_LINEAR)
        # depth = cv2.resize(depth,(320,240),cv2.INTER_NEAREST)
        depth[depth > 16000] = 0
        a = depth[::2,::2]
        b = depth[1::2,::2]
        c = depth[1::2,1::2]
        d = depth[::2,1::2]
        small_d = [a,b,c,d]
        mask_d = [a>0,b>0,c>0,d>0]
        final_d = small_d[0] + small_d[1] + small_d[2] + small_d[3]
        mask = final_d > 0
        mask_d = mask_d[0].astype(np.uint16) + mask_d[1].astype(np.uint16) + mask_d[2].astype(np.uint16) + mask_d[3].astype(np.uint16) 
        final_d[mask] = final_d[mask] / mask_d[mask]



        line = line.replace('jpg','resize.jpg')
        path = path.replace('png','resize.png')
        cv2.imwrite(line,img)
        cv2.imwrite(path,final_d)
        cnt += 1
        if(cnt %200 == 0):
            print(cnt)

