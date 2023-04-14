import torch
import torch.nn as nn

import torch
from torch import nn
from kornia.utils import create_meshgrid
import torch.nn.functional as F


def randomSampling(inputs, targets, masks, threshold, sample_num):

    # find A-B point pairs from predictions
    inputs_index = torch.masked_select(inputs, targets.gt(threshold))
    num_effect_pixels = len(inputs_index)
    shuffle_effect_pixels = torch.randperm(num_effect_pixels).cuda()
    inputs_A = inputs_index[shuffle_effect_pixels[0:sample_num*2:2]]
    inputs_B = inputs_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # find corresponding pairs from GT
    target_index = torch.masked_select(targets, targets.gt(threshold))
    targets_A = target_index[shuffle_effect_pixels[0:sample_num*2:2]]
    targets_B = target_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # only compute the losses of point pairs with valid GT
    consistent_masks_index = torch.masked_select(masks, targets.gt(threshold))
    consistent_masks_A = consistent_masks_index[shuffle_effect_pixels[0:sample_num*2:2]]
    consistent_masks_B = consistent_masks_index[shuffle_effect_pixels[1:sample_num*2:2]]

    # The amount of A and B should be the same!!
    if len(targets_A) > len(targets_B):
        targets_A = targets_A[:-1]
        inputs_A = inputs_A[:-1]
        consistent_masks_A = consistent_masks_A[:-1]

    return inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B


def randomSampling2(inputs, targets, threshold, sample_num):
    m = targets[:,2].gt(threshold)
    # find A-B point pairs from predictions
    inputs_index = inputs[m,:]
    num_effect_pixels = inputs_index.shape[0]
    shuffle_effect_pixels = torch.randperm(num_effect_pixels).cuda()
    shuffle_effect_pixels1 = torch.randperm(num_effect_pixels).cuda()
    shuffle_effect_pixels2 = torch.randperm(num_effect_pixels).cuda()
    sample_num = min(num_effect_pixels - 1, sample_num)
    inputs_A = inputs_index[shuffle_effect_pixels[0:sample_num]]
    inputs_B = inputs_index[shuffle_effect_pixels1[0:sample_num]]
    inputs_C = inputs_index[shuffle_effect_pixels2[0:sample_num]]
    # find corresponding pairs from GT
    target_index = targets[m,:]
    targets_A = target_index[shuffle_effect_pixels[0:sample_num]]
    targets_B = target_index[shuffle_effect_pixels1[0:sample_num]]
    targets_C = target_index[shuffle_effect_pixels2[0:sample_num]]
    # only compute the losses of point pairs with valid GT
    # consistent_masks_index = torch.masked_select(masks, m)
    # consistent_masks_A = consistent_masks_index[shuffle_effect_pixels[0:sample_num]]
    # consistent_masks_B = consistent_masks_index[shuffle_effect_pixels1[0:sample_num]]
    # consistent_masks_C = consistent_masks_index[shuffle_effect_pixels2[0:sample_num]]

    # # The amount of A and B should be the same!!
    # if len(targets_A) > len(targets_B):
    #     targets_A = targets_A[:-1]
    #     inputs_A = inputs_A[:-1]
    #     # consistent_masks_A = consistent_masks_A[:-1]

    return inputs_A, inputs_B,inputs_C, targets_A, targets_B, targets_C

def ind2sub(idx, cols):
    r = idx / cols
    c = idx - r * cols
    return r, c

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx

######################################################
# Ranking loss (Random sampling)
#####################################################
class RankingLoss(nn.Module):
    def __init__(self, point_pairs=5000, sigma=0.03, alpha=1.0, mask_value=1e-8):
        super(RankingLoss, self).__init__()
        self.point_pairs = point_pairs # number of point pairs
        self.sigma = sigma # used for determining the ordinal relationship between a selected pair
        self.alpha = alpha  # used for balancing the effect of = and (<,>)
        self.mask_value = mask_value

    def forward(self, inputs, targets, masks=None):
        n,h,w = targets.size()
        if masks == None:
            masks = targets > self.mask_value
        if n != 1:
            inputs = inputs.view(n, -1).double()
            targets = targets.view(n, -1).double()
            masks = masks.view(n, -1).double()
        else:
            inputs = inputs.contiguous().view(1, -1).double()
            targets = targets.contiguous().view(1, -1).double()
            masks = masks.contiguous().view(1, -1).double()

        loss = torch.DoubleTensor([0.]).cuda()
        for i in range(n):
            # find A-B point pairs
            inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B = randomSampling(inputs[i,:], targets[i, :], masks[i, :], self.mask_value, self.point_pairs)

            #GT ordinal relationship
            target_ratio = torch.div(targets_A, targets_B+1e-8)
            mask_eq = target_ratio.lt(1.0 + self.sigma) * target_ratio.gt(1.0/(1.0+self.sigma))
            labels = torch.zeros_like(target_ratio)
            labels[target_ratio.ge(1.0 + self.sigma)] = 1
            labels[target_ratio.le(1.0/(1.0+self.sigma))] = -1

            # consider forward-backward consistency checking, only compute the losses of point pairs with valid GT
            consistency_mask = consistent_masks_A * consistent_masks_B

            # compute loss
            equal_loss = (inputs_A - inputs_B).pow(2) * mask_eq.double() * consistency_mask
            unequal_loss = torch.log(1 + torch.exp((-inputs_A + inputs_B) * labels)) * (~mask_eq).double() * consistency_mask

            loss = loss + self.alpha * torch.sum(equal_loss)/(equal_loss.numel() + 1e-8) + torch.sum(unequal_loss) / (unequal_loss.numel() + 1e-8)

        return loss[0].float()/(n + 1e-8)

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss

# Ranking loss (Random sampling)
#####################################################
class VNLoss(nn.Module):
    def __init__(self, point_pairs=12000, mask_value=1e-5):
        super(VNLoss, self).__init__()
        self.point_pairs = point_pairs # number of point pairs
        self.mask_value = mask_value

    def forward(self, inputs, targets, focal_length = None):
        n,h,w = targets.size()
        inputs = inputs.view(n, -1)
        targets = targets.view(n, -1)

        proto_grid = create_meshgrid(h,w,normalized_coordinates=False,device=inputs.device) # (1, H, W, 2)
        proto_grid = proto_grid.reshape(1,-1,2) # (1, H* W, 2)
        proto_grid[:,:,0] = proto_grid[:,:,0] - w // 2
        proto_grid[:,:,1] = proto_grid[:,:,1] - h // 2
        proto_grid = proto_grid.view(-1,2)

        # print(focal_length.shape)
        # print(proto_grid.shape,inputs.shape,targets.shape)
        loss = 0
        for i in range(n):
            input_grid = torch.cat((proto_grid/focal_length[i],inputs[i,:].view(inputs.shape[1],1)),1)# (H* W, 3)
            target_grid = torch.cat((proto_grid/focal_length[i],targets[i,:].view(inputs.shape[1],1)),1)# (H* W, 3)
            # find A-B point pairs
            inputs_A, inputs_B, inputs_C, targets_A, targets_B,targets_C = randomSampling2(input_grid, target_grid, self.mask_value, self.point_pairs)
            gt_p12 = inputs_A - inputs_B
            gt_p13 = inputs_A - inputs_C
            dt_p12 = targets_A - targets_B
            dt_p13 = targets_A - targets_C

            gt_normal = torch.cross(gt_p12, gt_p13, dim=1)
            dt_normal = torch.cross(dt_p12, dt_p13, dim=1)
            dt_norm = torch.norm(dt_normal, 2, dim=1, keepdim=True)
            gt_norm = torch.norm(gt_normal, 2, dim=1, keepdim=True)
            distance_mask = (dt_norm > 0.04)
            gt_normal = gt_normal[distance_mask[:,0],:]
            dt_normal = dt_normal[distance_mask[:,0],:]
            gt_norm = gt_norm[distance_mask[:,0],:]
            dt_norm = dt_norm[distance_mask[:,0],:]
            # dt_mask = dt_norm == 0.0
            # gt_mask = gt_norm == 0.0
            # dt_mask = dt_mask.to(torch.float32)
            # gt_mask = gt_mask.to(torch.float32)
            # dt_mask *= 0.01
            # gt_mask *= 0.01
            # gt_norm = gt_norm + gt_mask
            # dt_norm = dt_norm + dt_mask
            gt_normal = gt_normal / gt_norm
            dt_normal = dt_normal / dt_norm
            loss_t = torch.abs(gt_normal - dt_normal)
            # loss, indices = torch.sort(loss, dim=0, descending=False)
            # loss = loss[int(loss.size(0) * 0.1):]
            loss_t = torch.mean(torch.sum(loss_t, dim=1))
            if loss_t > 0:
                loss += loss_t

            # consider forward-backward consistency checking, only compute the losses of point pairs with valid GT
            # consistency_mask = consistent_masks_A * consistent_masks_B * consistent_masks_C

        loss = loss / n

        return loss

def gradient_1order(x,h_x=None,w_x=None):
    x = x.view(x.shape[0],1,x.shape[1],x.shape[2])
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = (torch.pow((r - l), 2) + torch.pow((t - b), 2)) / 4
    return xgrad[:,0]

class FinalLoss(nn.Module):
    def __init__(self, max_depth=10):
        super(FinalLoss, self).__init__()

        self.max_depth = max_depth
        self.t_valid = 0.0001
        self.pool = torch.nn.MaxPool2d(kernel_size=3,stride=1,padding=1)

    def forward(self, pred, gt, theta, eval_confidence = True):
        mask = (gt > self.t_valid).detach()
        gt = torch.clamp(gt, max=self.max_depth)
        pred = torch.clamp(pred, min=0, max=self.max_depth)

        diff = torch.abs(pred[mask] - gt[mask])
        loss = diff.mean() 

        if eval_confidence:
            ddiff = None
            mask_2 = (self.pool(-gt) < 0).detach()
            gt_tv = gradient_1order(gt)
            pred_tv = gradient_1order(pred)
            ddiff = (gt_tv[mask_2] - pred_tv[mask_2]).abs()
            loss += ddiff.mean() * 0.3

        if eval_confidence:
            abs_rel = diff / gt[mask] 
            abs_rel = torch.clamp(abs_rel * 5,0,1)
            theta = torch.clamp(theta,1e-3,0.999)
            theta_mask = theta[mask]
            diff1 = theta_mask * diff
            loss += -0.1*((1 - abs_rel) * torch.log(theta_mask) + \
                              abs_rel * torch.log(1 - theta_mask)).mean()  + diff1.sum() / (theta_mask.sum()) * 0.1 
            
        return loss
        