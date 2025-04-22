# python eval.py --dataset nyudepthv2 --max_depth 10.0 --max_depth_eval 10.0 --ckpt ./logs/nyudepthv2/0814204158_test/epoch_25_model_big.ckpt #--save_result

source activate /nas/users/xieweijian/miniconda/envs/bbcnet

CUDA_VISIBLE_DEVICES=0 python eval.py --dataset nyudepthv2 --max_depth 20.0 \
    --max_depth_eval 20.0 --save_result \
    --epochs 25 --batch_size 32 --workers 8 --base_num 4 --exp_name res_34-rs \
    --cond_loss_weight 0.0 --crit_loss_weight 1.1 --dep_grad_loss_weight 0.1 \
    --log_dir /nas/users/xieweijian/project/train_log/ \
    --use_fast_corner --points_num 125 --resume --ckpt /nas/users/xieweijian/project/densedepth2/logs/lmdb_dataset/0222145502_test/epoch_48_model_big.ckpt  \
    --with_confidence \

