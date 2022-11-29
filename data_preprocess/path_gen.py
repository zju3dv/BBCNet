import os


def get_all_abs_path(source_dir):
    path_list = []
    for fpathe, dirs, fs in os.walk(source_dir):
        for f in fs:
            p = os.path.join(fpathe, f)
            path_list.append(p)
    return path_list


prefix = '/home/chuguanyi1/WorkSpace/depth_est/LERES_B2G/data/nyu_depth_v2/official_splits/train'
result = get_all_abs_path(prefix)

jpgs=[]
pngs=[]

for s in result:
    s = s.replace(prefix,'')
    if 'jpg' in s:
        jpgs.append(s)
    elif 'png' in s:
        pngs.append(s)

jpgs.sort()
pngs.sort()

for i in range(len(jpgs)):
    print(jpgs[i], pngs[i])
