source activate /nas/users/xieweijian/miniconda/envs/bbcnet

CUDA_VISIBLE_DEVICES=0 python train224_single_gpu.py --dataset lmdb_dataset --max_depth 20.0 \
    --max_depth_eval 20.0 --save_result \
    --epochs 25 --batch_size 32 --workers 8 --base_num 4 --exp_name res_34-rs \
    --cond_loss_weight 0.0 --crit_loss_weight 1.1 --dep_grad_loss_weight 0.1 \
    --log_dir /nas/users/xieweijian/project/train_log/ \
    --use_fast_corner --points_num 125 --resume --ckpt /nas/users/xieweijian/project/densedepth2/logs/lmdb_dataset/0207153055_test/epoch_24_model_big.ckpt \
    --with_confidence \


# CUDA_VISIBLE_DEVICES=0c python train224_single_gpu.py --dataset lmdb_dataset --max_depth 10.0 --max_depth_eval 10.0 --save_result --epochs 25 --batch_size 64 --save_model --workers 8 #--with_confidence
# accelerate launch train_multi.py --dataset nyudepthv2 --max_depth 10.0 --max_depth_eval 10.0 --save_result --epochs 25 --batch_size 64 --save_model --workers 5 #--with_confidence