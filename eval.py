from math import nan,inf
import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from model.MBNet import RelDepthModel

from torch.utils.tensorboard import SummaryWriter
import random

from model.MBNet import RelDepthModel
import utils.metrics as metrics
from utils.criterion import FinalLoss,RankingLoss,VNLoss
import utils.logging as logging

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions
import utils.manager_torch_gpu as manager_torch_gpu

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']
device_ids = manager_torch_gpu.GPUManager(target_number=1).auto_choice()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(7240)
def main():
    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)

    # Logging
    exp_name = '%s_%s' % (datetime.now().strftime('%m%d'), args.exp_name)
    log_dir = os.path.join(args.log_dir, args.dataset, exp_name)
    logging.check_and_make_dirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    log_txt = os.path.join(log_dir, 'logs.txt')  
    logging.log_args_to_txt(log_txt, args)

    global result_dir
    result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    model = RelDepthModel()


    device = torch.device('cuda')
    cudnn.benchmark = True
    model = torch.nn.DataParallel(model.depth_model, device_ids=[i for i in range(len(device_ids))])
    # model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
    if args.dataset == 'nyudepthv2':
        dataset_kwargs['crop_size'] = (384, 512)
    else:
        dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

    train_dataset = get_dataset(**dataset_kwargs)
    val_dataset = get_dataset(**dataset_kwargs, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, 
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=4,
                                             pin_memory=True)

    # Training settings
    criterion_d = FinalLoss()

    global global_step
    global_step = 0

    # if True:
        # pretrained = torch.load('/mnt/cache/chuguanyi1/dense_depth/logs/nyudepthv2/1106_test/epoch_21_model_big.ckpt')
        # model.load_state_dict(pretrained,strict=False)

    # Perform experiment
    for epoch in range(24, 25):
        # pretrained = torch.load('/mnt/cache/chuguanyi1/dense_depth/logs4/nyudepthv2/1118_test/epoch_24_model_big.ckpt')
        pretrained = torch.load('/mnt/cache/chuguanyi1/dense_depth/logs432/nyudepthv2/1119_test/epoch_25_model_big.ckpt')
        model.load_state_dict(pretrained,strict=False)
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))

        if epoch % args.val_freq == 0:
            results_dict, loss_val = validate(val_loader, model, criterion_d, 
                                              device=device, epoch=epoch, args=args,
                                              log_dir=log_dir)
            writer.add_scalar('Val loss', loss_val, epoch)

            result_lines = logging.display_result(results_dict)
            print(result_lines)

            with open(log_txt, 'a') as txtfile:
                txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
                txtfile.write(result_lines)                

            for each_metric, each_results in results_dict.items():
                writer.add_scalar(each_metric, each_results, epoch*3)


def validate(val_loader, model, criterion_d, device, epoch, args, log_dir):
    depth_loss = logging.AverageMeter()
    model.eval()

    val_loader.dataset.epoch = min(epoch * 6,100)

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    total_cnt = 0
    abs_rel = 0
    step = 0
    total_mask = 0
    valid_mask = 0
    invalid = 0

    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device)
        depth_gt_ori = batch['depth'].to(device)
        focal_length = batch['focal_length'].to(device)
        step+=1
        with torch.no_grad():
            preds,conf,out_base = model(input_RGB)
            preds = preds[:,0,:,:]
            # print(preds.shape,focal_length.shape)
            focal_length = focal_length.view(preds.shape[0],1).expand(
                preds.shape[0],preds.shape[1]).view(preds.shape[0],preds.shape[1],1).expand(preds.shape[0],preds.shape[1],preds.shape[2])
            preds = preds * ( focal_length/ 519)

        mask = input_RGB[:,1] > 0
        input_RGB[:,1] = input_RGB[:,1] * focal_length / 519 * 10
        # print(input_RGB[:,1][mask].max())
        # if(input_RGB[:,1][mask].max() > 10):
        #     print('input error')
        #     continue
        abs_rel += ((preds[mask] - input_RGB[:,1][mask]).abs()/input_RGB[:,1][mask]).mean()
        # loss_d = criterion_d(preds.squeeze(), depth_gt_ori.squeeze())
        # depth_loss.update(loss_d.item(), input_RGB.size(0))

        for i in range(preds.shape[0]):
            pred_d = preds[i].squeeze()
            if(pred_d.sum(dim=(-1,-2)) == 0):
                invalid+=1
                continue
            total_cnt+=1
            # pred_d[pred_d==nan] = 0
            depth_gt = depth_gt_ori[i].squeeze()
            depth_gt[conf[i] < 0] = 0
            valid_mask += (conf[i] > 0.545).sum()
            total_mask += (conf[i] > 0).sum()
            # print((conf[i] < 0.5).sum(),(conf[i] < 0.7).sum())

            mask = depth_gt > 0

            filename = batch['filename'][i]

            pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
            computed_result = metrics.eval_depth(pred_crop, gt_crop)
            save_path = os.path.join(result_dir, filename)
            # print(save_path)
            if args.save_result and i == 0:
                pred_d_numpy = pred_d.cpu().numpy()
                d_max = pred_d_numpy.max()
                #cv2.imwrite(save_path.replace('rgb','predict').replace('jpg','gray.jpg'), (input_RGB[:,0].squeeze().cpu().numpy() * 255).astype(np.uint8))
                #cv2.imwrite(save_path.replace('rgb','predict').replace('jpg','png'), (pred_d_numpy * 1000).astype(np.uint16))
                pred_d_numpy = ((pred_d_numpy ) / 10 * 256).astype(np.uint8)
                #cv2.imwrite(save_path, pred_d_numpy.astype(np.uint8))

                for j,base in enumerate(out_base):
                    pred_d_numpy = base.squeeze().abs().cpu().numpy()
                    d_min = pred_d_numpy.min()
                    d_max = pred_d_numpy.max()
                    pred_d_numpy = ((pred_d_numpy - d_min) / (d_max - d_min) * 256).astype(np.uint8)
                    #cv2.imwrite(save_path.replace('.jpg','.%d'%j)+ ".jpg", pred_d_numpy)
                # else:
                #     pred_d_numpy = pred_d.cpu().numpy()
                #     d_min = pred_d_numpy.min()
                #     d_max = pred_d_numpy.max()
                #     pred_d_numpy = ((pred_d_numpy - d_min) / (d_max - d_min) * 256).astype(np.uint8)
                #     cv2.imwrite(save_path, pred_d_numpy.astype(np.uint8))
            for key in result_metrics.keys():
                result_metrics[key] += computed_result[key]

        logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

    print('sparse abs rel:',abs_rel/step,"valid mask:",valid_mask/total_mask, "invalid number:", invalid)

            
    loss_d = depth_loss.avg
    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (total_cnt + 1e-8)

    return result_metrics, loss_d


if __name__ == '__main__':
    main()
