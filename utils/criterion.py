import torch
import torch.nn as nn

import torch
from torch import nn
from kornia.utils import create_meshgrid
import torch.nn.functional as F
from kornia.filters import spatial_gradient

def randomSampling2(inputs, targets, m, sample_num):
    # find A-B point pairs from predictions
    inputs_index = inputs[m,:]
    target_index = targets[m,:]
    num_effect_pixels = inputs_index.shape[0]
    shuffle_effect_pixels = torch.randperm(num_effect_pixels,device=inputs.device)
    
    sample_num = min(num_effect_pixels // 3 - 1, sample_num)
    inputs_A = inputs_index[shuffle_effect_pixels[0:sample_num]]
    inputs_B = inputs_index[shuffle_effect_pixels[sample_num:sample_num*2]]
    inputs_C = inputs_index[shuffle_effect_pixels[sample_num*2:sample_num*3]]

    # find corresponding pairs from GT
    targets_A = target_index[shuffle_effect_pixels[0:sample_num]]
    targets_B = target_index[shuffle_effect_pixels[sample_num:sample_num*2]]
    targets_C = target_index[shuffle_effect_pixels[sample_num*2:sample_num*3]]

    return inputs_A, inputs_B,inputs_C, targets_A, targets_B, targets_C

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
    def __init__(self, point_pairs=8192, mask_value=1e-5):
        super(VNLoss, self).__init__()
        self.point_pairs = point_pairs # number of point pairs
        self.mask_value = mask_value

    def forward(self, inputs, targets, imask, focal_length = None):
        n,_,h,w = targets.size()
        inputs = inputs.view(n, -1)
        targets = targets.view(n, -1)
        imask = imask.view(n, -1)

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
            inputs_A, inputs_B, inputs_C, targets_A, targets_B,targets_C = randomSampling2(input_grid, target_grid, imask[i], self.point_pairs)
            p12 = inputs_A - inputs_B
            p13 = inputs_A - inputs_C
            g_p12 = targets_A - targets_B
            g_p13 = targets_A - targets_C
            #linear test
            g_p12_norm = torch.norm(g_p12, 2, dim=1, keepdim=True) + 1e-5
            g_p13_norm = torch.norm(g_p13, 2, dim=1, keepdim=True) + 1e-5
            g_p12 = g_p12 / g_p12_norm
            g_p13 = g_p13 / g_p13_norm

            # dot product
            gp_dot = torch.sum(g_p12 * g_p13, dim=1, keepdim=True).abs() # cose theta
            
            p_normal = torch.cross(p12, p13, dim=1)
            gt_normal = torch.cross(g_p12, g_p13, dim=1)
            d_norm_mask = (g_p12_norm > 0.1) & (g_p13_norm > 0.1) & (gp_dot < 0.96)
            
            
            final_mask = d_norm_mask.squeeze().detach()
            p_normal = torch.nn.functional.normalize(p_normal,p=2,dim=-1)
            gt_normal = torch.nn.functional.normalize(gt_normal,p=2,dim=-1)
            loss_t = torch.abs(p_normal - gt_normal)

            loss_t = loss_t.sum(dim=1) * final_mask
            # loss, indices = torch.sort(loss, dim=0, descending=False)
            # loss = loss[int(loss.size(0) * 0.1):]
            # print(loss_t,distance_mask.sum())
            loss_t = torch.sum(loss_t) / (final_mask.sum() + 1)
            loss += loss_t

        loss = loss / n

        return loss

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
            mask_2 = (self.pool(-gt) < 0).detach()
            grad = spatial_gradient(torch.cat([gt,pred],dim=1), mode = 'diff',normalized=False)
            ddiff = (grad[:,0:1] - grad[:,1:2]).abs().sum(dim=2)
            ddiff = ddiff[mask_2]
            loss += ddiff.mean() * 0.1

        if eval_confidence:
            error = diff  #/ gt[mask] 
            error = torch.clamp(error * 7,0,1)
            theta = torch.clamp(theta,1e-3,0.999)
            theta_mask = theta[mask]
            diff1 = theta_mask * diff
            loss += -0.1*((1 - error) * torch.log(theta_mask) + \
                              error * torch.log(1 - theta_mask)).mean()  + diff1.sum() / (theta_mask.sum()) * 0.1 
            
        return loss
        