## training

1. 准备数据。图像用jpg格式保存，深度用同名的png格式保存。深度单位为mm，数据类型为uint16。无深度处赋值0
2. 将图像名搜集到train.txt和valid.txt两个文件里面，将这两个文件放进./dataset/filenames/scannet文件夹下
3. bash train.sh

## evaluation
同上，运行bash eval.sh

