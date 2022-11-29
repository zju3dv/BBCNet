import argparse


class BaseOptions():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # base configs
        parser.add_argument('--gpu_or_cpu',   type=str, default='gpu')
        parser.add_argument('--data_path',    type=str, default='/data/ssd1/')
        parser.add_argument('--dataset',      type=str, default='nyudepthv2',
                            choices=['nyudepthv2', 'imagepath', 'nyudepthh5'])
        parser.add_argument('--exp_name',     type=str, default='test')
        parser.add_argument('--batch_size',   type=int, default=8)
        parser.add_argument('--workers',      type=int, default=8)
        
        # depth configs
        parser.add_argument('--max_depth',      type=float, default=10.0)
        parser.add_argument('--max_depth_eval', type=float, default=10.0)
        parser.add_argument('--min_depth_eval', type=float, default=1e-3)        
        parser.add_argument('--backbone', type=str, default='resnet34')        
        parser.add_argument('--slam_infer', action='store_true')        
        parser.add_argument('--with_confidence', action='store_true')        
        parser.add_argument('--port', default='23466', type=str,
                            help='node rank for distributed training')
        return parser

class TrainOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)

        # experiment configs
        parser.add_argument('--epochs',      type=int,   default=150)
        parser.add_argument('--lr',          type=float, default=1e-4)
        
        parser.add_argument('--crop_h',  type=int, default=384)
        parser.add_argument('--crop_w',  type=int, default=512)        
        parser.add_argument('--log_dir', type=str, default='./logs') # TODO logs1 for late fusion

        # logging options
        parser.add_argument('--val_freq', type=int, default=1)
        parser.add_argument('--save_freq', type=int, default=10)
        parser.add_argument('--save_model', action='store_true')        
        parser.add_argument('--save_result', action='store_true')

        return parser
