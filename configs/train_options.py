import argparse


class BaseOptions():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # base configs
        parser.add_argument('--gpu_or_cpu',   type=str, default='gpu')
        parser.add_argument('--dataset',      type=str, default='nyudepthv2',
                            choices=['nyudepthv2', 'imagepath', 'lmdb_dataset'])
        parser.add_argument('--exp_name',     type=str, default='test')
        parser.add_argument('--batch_size',   type=int, default=64)
        parser.add_argument('--workers',      type=int, default=8)
        parser.add_argument('--ckpt',      type=str, default='',help='model to load')
        
        # depth configs
        parser.add_argument('--max_depth',      type=float, default=20.0)
        parser.add_argument('--max_depth_eval', type=float, default=10.0)
        parser.add_argument('--min_depth_eval', type=float, default=1e-3)        
        parser.add_argument('--backbone', type=str, default='resnet34')        
        parser.add_argument('--slam_infer', action='store_true')        
        parser.add_argument('--with_confidence', action='store_true')        
        parser.add_argument('--port', default='23466', type=str,
                            help='node rank for distributed training')
        parser.add_argument('--use_sigmoid', action='store_true')
        parser.add_argument('--base_num', type=int, default = 4)
        parser.add_argument('--add_noise', action='store_true')
        parser.add_argument('--depth_noise_level', type=float, default=0.07)
        parser.add_argument('--use_diff_kpts', action='store_true')
        parser.add_argument('--use_vnl_loss', action='store_true')
        parser.add_argument('--use_fast_corner', action='store_true')
        parser.add_argument('--points_num', type=int, default = 125)
        parser.add_argument('--close_aug', action='store_true')
        parser.add_argument('--use_huber_weight', action='store_true')
        parser.add_argument('--use_last_bn_layer', action='store_true')
        parser.add_argument('--use_rgb', action='store_true')
        parser.add_argument('--crop_h',  type=int, default=448)
        parser.add_argument('--crop_w',  type=int, default=576)
        return parser

class TrainOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)

        # experiment configs
        parser.add_argument('--epochs',      type=int,   default=40)
        parser.add_argument('--lr',          type=float, default=1e-3)
        parser.add_argument('--export_onnx',   action='store_true')        
        parser.add_argument('--resume',   action='store_true')        
        parser.add_argument('--resume_path',   type=str, default='/nas/users/xieweijian/project/densedepth2/logs/lmdb_dataset/0207153055_test/epoch_24_model_big.ckpt')        
        parser.add_argument('--begin_lr',    type=float, default=1e-3)
        parser.add_argument('--end_lr',    type=float, default=1e-5)
        
        # parser.add_argument('--crop_h',  type=int, default=384)
        # parser.add_argument('--crop_w',  type=int, default=512)        
        parser.add_argument('--log_dir', type=str, default='./logs') 

        # logging options
        parser.add_argument('--val_freq', type=int, default=1)
        parser.add_argument('--save_freq', type=int, default=10)
        parser.add_argument('--save_model', action='store_true')        
        parser.add_argument('--save_result', action='store_true')
        parser.add_argument('--dep_grad_loss_weight', type=float, default=0.1)
        parser.add_argument('--cond_loss_weight', type=float, default=0.1)
        parser.add_argument('--crit_loss_weight', type=float, default=1.1)
        parser.add_argument('--grad_max', type=float, default=-1.)
        parser.add_argument('--confidence_type', type=int, default='0')
        return parser
