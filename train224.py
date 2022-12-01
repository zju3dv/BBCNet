from math import nan,inf
import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from model.MBNet import DepthModel

from torch.utils.tensorboard import SummaryWriter
import random

import utils.metrics as metrics
from utils.criterion import FinalLoss
import utils.logging as logging

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False



def main_worker(gpu, nprocs, args, model):
    if gpu == 0:
        exp_name = '%s_%s' % (datetime.now().strftime('%m%d%H%M%S'), args.exp_name)
        log_dir = os.path.join(args.log_dir, args.dataset, exp_name)
        log_txt = os.path.join(log_dir, 'logs.txt')  
        logging.check_and_make_dirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        logging.log_args_to_txt(log_txt, args)
        global result_dir
        result_dir = os.path.join(log_dir, 'results')
        code_dir = os.path.join(log_dir, 'code')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists(code_dir):
            os.makedirs(code_dir)
        logging.backup_source_code(code_dir)

    dist.init_process_group(backend='nccl', world_size=nprocs, rank=gpu)
    torch.cuda.set_device(gpu)

    # model = torch.nn.DataParallel(model, device_ids=[i for i in range(len(device_ids))])
    # model = torch.nn.DataParallel(model)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(dist.get_rank())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])#,find_unused_parameters=True)

    # model = model.cuda()

    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
    dataset_kwargs['crop_size'] = (224, 224)

    train_dataset = get_dataset(**dataset_kwargs)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size // nprocs,
                                               shuffle=False, num_workers=args.workers, 
                                               pin_memory=True, drop_last=True, sampler = train_sampler)
                                               
    # # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # val_dataset = get_dataset(**dataset_kwargs, is_train=False)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size , shuffle=False,num_workers=args.workers,
    #                                          pin_memory=True)

    # Training settings
    criterion_d = FinalLoss()
    # optimizer = optim.Adam(model.parameters()) #, args.lr,weight_decay=1e-8)
    optimizer = optim.AdamW(model.parameters()) #, args.lr,weight_decay=1e-8)

    # Perform experiment
    for epoch in range(1, args.epochs + 1):
        if dist.get_rank() == 0:
            print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        loss_train = train(train_loader, model, criterion_d, optimizer=optimizer, 
                           epoch=epoch, args=args)

        # if dist.get_rank() == 0:
        #     writer.add_scalar('Training loss', loss_train, epoch)
        #     if epoch % args.val_freq == 0:
        #         results_dict, loss_val = validate(val_loader, model.module, criterion_d, 
        #                                         epoch=epoch, args=args,
        #                                         log_dir=log_dir)
        #         writer.add_scalar('Val loss', loss_val, epoch)

        #         result_lines = logging.display_result(results_dict)
        #         print(result_lines)
        #         with open(log_txt, 'a') as txtfile:
        #             txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        #             txtfile.write(result_lines)                

        #         for each_metric, each_results in results_dict.items():
        #             writer.add_scalar(each_metric, each_results, epoch)

class LRFactor:
    def __init__(self, decay, gamma):
        assert len(decay) == len(gamma)

        self.decay = decay
        self.gamma = gamma

    def get_factor(self, epoch):
        for (d, g) in zip(self.decay, self.gamma):
            if epoch < d:
                return g
        return self.gamma[-1]

def train(train_loader, model, criterion_d, optimizer, epoch, args):    
    model.train()
    if dist.get_rank() == 0:
        depth_loss = logging.AverageMeter()

    lr_schedule = [1.00000000e-03, 9.96061293e-04, 9.84307289e-04, 9.64923355e-04,
            9.38215187e-04, 9.04603989e-04, 8.64619829e-04, 8.18893283e-04,
            7.68145484e-04, 7.13176756e-04, 6.54853989e-04, 5.94096967e-04,
            5.31863865e-04, 4.69136135e-04, 4.06903033e-04, 3.46146011e-04,
            2.87823244e-04, 2.32854516e-04, 1.82106717e-04, 1.36380171e-04,
            9.63960113e-05, 6.27848133e-05, 3.60766453e-05, 1.66927110e-05,
            4.93870669e-06]

    # train_loader.dataset.epoch = 100
    train_loader.sampler.set_epoch(epoch)
    warm_up_step = 200

    for batch_idx, batch in enumerate(train_loader):      

        # for param_group in optimizer.param_groups:
        #     if batch_idx < warm_up_step and epoch == 1:
        #         current_lr = 1e-5 + (1e-3 - 1e-5) * ( batch_idx/ warm_up_step) 
        #     else:
        #         current_lr = 1e-3#lr_schedule[epoch - 1]

        #     param_group['lr'] = current_lr

        input_RGB = batch['image'].cuda(non_blocking=True)
        depth_gt = batch['depth'].cuda(non_blocking=True)
        focal_length = batch['focal_length'].cuda(non_blocking=True)
        preds,pred_straight,confidence,sum_w = model(input_RGB)
        preds = preds[:,0,:,:]
        preds = preds * (focal_length.expand(preds.shape[0],preds.shape[1]).view(preds.shape[0],preds.shape[1],1).expand(preds.shape[0],preds.shape[1],preds.shape[2]) / 519)
        pred_straight = pred_straight[:,0,:,:]
        pred_straight = pred_straight * (focal_length.expand(preds.shape[0],preds.shape[1]).view(preds.shape[0],preds.shape[1],1).expand(preds.shape[0],preds.shape[1],preds.shape[2]) / 519)

        optimizer.zero_grad()
        loss_d = criterion_d(preds, depth_gt, confidence, args.with_confidence)  + criterion_d(pred_straight, depth_gt, confidence,args.with_confidence) * 0.1
        if loss_d is nan or loss_d is inf:
            loss_d = 0
            print('loss_d is nan')
            continue

        loss_d = loss_d +  sum_w  * 0.1
        loss_d = loss_d.mean()
        loss_d.backward()
        if dist.get_rank() == 0:
            real_loss = loss_d.item()
            depth_loss.update(real_loss, input_RGB.size(0))
            logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,
                                ('Depth Loss: %.4f,%.4f (%.4f)' %
                                (depth_loss.val, sum_w.mean().item()  * 0.1,depth_loss.avg)))
        optimizer.step()

    return loss_d


def validate(val_loader, model, criterion_d, epoch, args, log_dir):
    depth_loss = logging.AverageMeter()
    model.eval()

    if args.save_model and dist.get_rank() == 0:
        idx = epoch %50
        torch.save(model.state_dict(), os.path.join(
            log_dir, 'epoch_%02d_model_big.ckpt' % idx))
    val_loader.dataset.epoch = 100

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    total_cnt = 0
    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].cuda()
        depth_gt_ori = batch['depth'].cuda()
        focal_length = batch['focal_length'].cuda()
        total_cnt += input_RGB.shape[0]
        with torch.no_grad():
            preds,confidence,_ = model(input_RGB)
            preds = preds[:,0,:,:]
            preds = preds * (focal_length.expand(preds.shape[0],preds.shape[1]).view(preds.shape[0],preds.shape[1],1).expand(preds.shape[0],preds.shape[1],preds.shape[2]) / 519)
        loss_d = criterion_d(preds.squeeze(), depth_gt_ori.squeeze(),confidence,False).mean()

        depth_loss.update(loss_d.item(), input_RGB.size(0))

        for i in range(preds.shape[0]):
            pred_d = preds[i].squeeze()
            depth_gt = depth_gt_ori[i].squeeze()
            # mask = depth_gt > 0
            # s =  depth_gt[mask].mul(pred_d[mask]).abs().sum() / pred_d[mask].square().sum()
            # pred_d = pred_d * s
            filename = batch['filename'][i]

            pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
            computed_result = metrics.eval_depth(pred_crop, gt_crop)

            if dist.get_rank() == 0:
                if args.save_result:
                    global result_dir
                    save_path = os.path.join(result_dir, filename)
                    pred_d_numpy = pred_d.cpu().numpy()
                    d_max = pred_d_numpy.max()
                    pred_d_numpy = ((pred_d_numpy ) / (d_max) * 256).astype(np.uint8)
                    if i == 0:
                        cv2.imwrite(save_path.replace('npy','jpg'), pred_d_numpy.astype(np.uint8))
            for key in result_metrics.keys():
                result_metrics[key] += computed_result[key]
        if dist.get_rank() == 0:
            logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)


            
    loss_d = depth_loss.avg
    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (total_cnt + 1e-8)

    return result_metrics, loss_d


if __name__ == '__main__':
    # main()
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    cudnn.benchmark = True
    # 设置随机数种子
    setup_seed(7240)
    opt = TrainOptions()
    train_args = opt.initialize().parse_args()
    # Logging
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = train_args.port
    model = DepthModel(train_args)

    if train_args.backbone == 'resnet34':
        checkpoint = torch.load('./pretrain/resnet34-333f7ec4.pth', map_location=torch.device('cpu'))
        model.encoder_modules.load_state_dict(checkpoint,strict=False)
        print("resnet loaded")
    elif train_args.backbone == 'mobilenet3':
        checkpoint = torch.load('./pretrain/resnet34-333f7ec4.pth', map_location=torch.device('cpu'))
        model.encoder_modules.load_state_dict(checkpoint,strict=False)
        print("mobilenet3 loaded")

    main_worker(proc_id,ntasks,train_args,model)
