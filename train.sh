export HOME=/mnt/cache/$USER
python train224.py --dataset nyudepthv2 --data_path /mnt/lustre/share_data/chuguanyi/scannet/public_datalist_185/ --max_depth 10.0 --max_depth_eval 10.0 --save_result --epochs 25 --save_model --batch_size 64 --workers 8
