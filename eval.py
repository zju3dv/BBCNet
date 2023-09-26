import os
import cv2
import numpy as np
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from model.MBNet import DepthModel

import utils.metrics as metrics
from utils.criterion import FinalLoss
import utils.logging as logging

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions

def initilize_workspace(args):
    exp_name = '%s_%s' % (datetime.now().strftime('%m%d%H%M%S'), args.exp_name)
    log_dir = os.path.join(args.log_dir, args.dataset, exp_name)
    log_txt = os.path.join(log_dir, 'logs.txt')  
    logging.check_and_make_dirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    logging.log_args_to_txt(log_txt, args)
    global result_dir
    result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return writer,log_dir

def main_worker(args, model):
    model = torch.nn.DataParallel(model, device_ids=[0])

    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset}

    val_dataset = get_dataset(**dataset_kwargs, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size , shuffle=False,num_workers=args.workers,
                                            pin_memory=True)

    # Training settings
    criterion_d = FinalLoss()

    writer,log_dir = initilize_workspace(args)
    log_txt = os.path.join(log_dir, 'logs.txt')  

    # Perform experiment
    results_dict, loss_val = validate(val_loader, model, criterion_d, 
                                    epoch=0, args=args)
    result_lines = logging.display_result(results_dict)
    print(result_lines)
    with open(log_txt, 'a') as txtfile:
        txtfile.write(result_lines)                


def validate(val_loader, model, criterion_d, epoch, args):
    depth_loss = logging.AverageMeter()
    model.eval()
    val_loader.dataset.epoch = 100

    metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
                'log10', 'silog']
    result_metrics = {metric:0 for metric in metric_name}

    total_cnt = 0
    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].cuda(non_blocking=True)
        depth_gt_ori = batch['depth'].cuda(non_blocking=True)
        focal_length = batch['focal_length'].cuda(non_blocking=True)
        total_cnt += input_RGB.shape[0]
        with torch.no_grad():
            preds,confidence,_ = model(input_RGB)
            preds = preds * (focal_length.view(preds.shape[0], 1, 1, 1) / 519)
        loss_d = criterion_d(preds, depth_gt_ori,confidence,False).mean()

        depth_loss.update(loss_d.item(), input_RGB.size(0))

        for i in range(preds.shape[0]):
            pred_d = preds[i].squeeze()
            depth_gt = depth_gt_ori[i].squeeze()
            filename = batch['filename'][i]

            pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
            computed_result = metrics.eval_depth(pred_crop, gt_crop)

            if args.save_result:
                global result_dir
                save_path = os.path.join(result_dir, filename)
                pred_d_numpy = pred_d.cpu().numpy()
                pred_d_numpy = (pred_d_numpy  * 25).astype(np.uint8)
                cv2.imwrite(save_path.replace('npy','jpg'), pred_d_numpy.astype(np.uint8))
            for key in result_metrics.keys():
                result_metrics[key] += computed_result[key]
        logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

    loss_d = depth_loss.avg
    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (total_cnt + 1e-8)

    return result_metrics, loss_d

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # 设置随机数种子
    # setup_seed(7240)
    opt = TrainOptions()
    eval_args = opt.initialize().parse_args()
    model = DepthModel(eval_args)

    checkpoint = torch.load(eval_args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('.module','').replace('._orig_mod.',''):v for k,v in checkpoint.items()},strict=True)

    main_worker(eval_args,model)
