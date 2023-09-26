## training

1. 准备数据。图像用jpg格式保存，深度用同名的png格式保存。深度单位为mm，数据类型为uint16。无深度处赋值0
2. 将图像的绝对路径写入train.txt和valid.txt两个文件里面，将这两个文件放进./dataset/filenames/scannet文件夹下
3. 
```
CUDA_VISIBLE_DEVICES=$GPU_ID python train224_single_gpu.py --dataset lmdb_dataset --max_depth 10.0 --max_depth_eval 10.0 --save_result --epochs 25 --batch_size 64 --save_model --workers 8 
```
如果需要加入confidence，则加入 --with_confidence

## evaluation

1. 按照上面的方式生成valid.txt文件
2. 
```
python eval.py --dataset nyudepthv2 --max_depth 10.0 --max_depth_eval 10.0 --ckpt /PATH/TO/CHECKPOINT #--save_result
```
如需保存结果，则加入--save_result 。其中ckpt 为用于推理的模型文件名。结果保存在 logs 文件夹下
