import torch
import torch.nn as nn

import torch
from torch import nn
from kornia.utils import create_meshgrid
from kornia.filters import spatial_gradient

def randomSampling(inputs, targets, sample_num):
    # find A-B point pairs from predictions
    num_effect_pixels = inputs.shape[1]
    shuffle_effect_pixels = torch.randperm(num_effect_pixels,device=inputs.device)
    
    inputs_A = inputs[:,shuffle_effect_pixels[0:sample_num]]
    inputs_B = inputs[:,shuffle_effect_pixels[sample_num:sample_num*2]]
    inputs_C = inputs[:,shuffle_effect_pixels[sample_num*2:sample_num*3]]

    # find corresponding pairs from GT
    targets_A = targets[:,shuffle_effect_pixels[0:sample_num]]
    targets_B = targets[:,shuffle_effect_pixels[sample_num:sample_num*2]]
    targets_C = targets[:,shuffle_effect_pixels[sample_num*2:sample_num*3]]

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
    def __init__(self, point_pairs=16384, mask_value=1e-5):
        super(VNLoss, self).__init__()
        self.point_pairs = point_pairs # number of point pairs
        self.mask_value = mask_value
        h = 480
        w = 640
        proto_grid = create_meshgrid(h,w,normalized_coordinates=False).cuda() # (1, H, W, 2)
        proto_grid = proto_grid.reshape(1,-1,2) # (1, H* W, 2)
        proto_grid[:,:,0] = proto_grid[:,:,0] - w // 2
        proto_grid[:,:,1] = proto_grid[:,:,1] - h // 2
        self.proto_grid = proto_grid


    def forward(self, inputs, targets, focal_length = None):
        B,_,h,w = targets.size()
        inputs = inputs.view(B, -1)
        targets = targets.view(B, -1)

        proto_grid = self.proto_grid.view(1,-1,2).repeat(B,1,1).contiguous() # (N, H* W, 2)
        # print(proto_grid.shape,focal_length.shape)
        proto_grid = proto_grid / focal_length.view(B,1,1)
        ones = torch.ones(B,h*w,1,device=inputs.device)
        input_grid_all = torch.cat((proto_grid,ones),-1)# (N, H* W, 3)
        target_grid_all = torch.cat((proto_grid,ones),-1)# (N, H* W, 3)
        input_grid_all = input_grid_all * inputs.unsqueeze(-1) # (N, H* W, 3)
        target_grid_all = target_grid_all * targets.unsqueeze(-1) # (N, H* W, 3)
        # print(focal_length.shape)
        # print(proto_grid.shape,inputs.shape,targets.shape)
        inputs_A, inputs_B, inputs_C, targets_A, targets_B,targets_C = randomSampling(input_grid_all, target_grid_all, self.point_pairs)# B N 3
        imask = (targets_A[:,:,2] > self.mask_value) & (targets_B[:,:,2] > self.mask_value) & (targets_C[:,:,2] > self.mask_value) # B N

        p12 = inputs_A - inputs_B # B N 3
        p13 = inputs_A - inputs_C # B N 3
        g_p12 = targets_A - targets_B # B N 3
        g_p13 = targets_A - targets_C # B N 3

        p_normal = torch.cross(p12, p13, dim=2) # B N 3
        gt_normal = torch.cross(g_p12, g_p13, dim=2) # B N 3

        g_p12_norm = torch.norm(g_p12, 2, dim=2, keepdim=True) + 1e-5 # B N 1
        g_p13_norm = torch.norm(g_p13, 2, dim=2, keepdim=True) + 1e-5 # B N 1
        g_p12 = g_p12 / g_p12_norm # B N 3
        g_p13 = g_p13 / g_p13_norm # B N 3
        gp_dot = torch.sum(g_p12 * g_p13, dim=2, keepdim=True).abs() # cose theta # B N 1

        p_normal = torch.nn.functional.normalize(p_normal,p=2,dim=-1) # B N 3
        gt_normal = torch.nn.functional.normalize(gt_normal,p=2,dim=-1) # B N 3
        loss_t = torch.abs(p_normal - gt_normal).sum(dim=2) # B N

        d_norm_mask = (g_p12_norm > 0.1) & (g_p13_norm > 0.1) & (gp_dot < 0.96)
        final_mask = (d_norm_mask.squeeze().detach() & imask.squeeze().detach()).float()
        loss_t = loss_t * final_mask
        loss_t = loss_t.sum(dim=1) / (final_mask.sum(dim=1) + 1)
        loss = loss_t.mean()


        return loss

class FinalLoss(nn.Module):
    def __init__(self, max_depth=10):
        super(FinalLoss, self).__init__()

        self.max_depth = max_depth
        self.t_valid = 0.0001
        self.pool = torch.nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.vnl_loss = VNLoss()

    def forward(self, pred, gt, theta, eval_confidence = True,focal_length = None):
        mask = (gt > self.t_valid).detach()
        gt = torch.clamp(gt, max=self.max_depth)
        pred = torch.clamp(pred, min=0, max=self.max_depth)
        diff = torch.abs(pred[mask] - gt[mask])
        loss = diff.mean()# + self.vnl(pred,gt,mask)
        # TODO:
        # 1. distance(to center) loss
        # 2. vnl loss
        if focal_length is not None:
            loss += self.vnl_loss(pred,gt,focal_length) * 0.2
        if eval_confidence:
            mask_2 = (self.pool(-gt) < 0).detach()
            grad = spatial_gradient(torch.cat([gt,pred],dim=1), mode = 'diff',normalized=False)
            ddiff = (grad[:,0:1] - grad[:,1:2]).abs().sum(dim=2)
            ddiff = ddiff[mask_2]
            loss += ddiff.mean() * 0.1

        if eval_confidence:
            # error = diff  #/ gt[mask] 
            # error = torch.clamp(error * 5,0,1)
            theta = torch.clamp(theta,1e-3,0.999) * 10
            theta_mask = theta[mask]
            # diff1 = theta_mask * diff
            # loss += -0.1*((1 - error) * torch.log(theta_mask) + \
            #                   error * torch.log(1 - theta_mask)).mean()  + diff1.sum() / (theta_mask.sum()) * 0.1 
            loss += 0.1 * (diff - theta[mask]).abs().mean()
            # loss += (torch.log(theta_mask.square())+ (diff / theta_mask).square()).mean() * 0.1

        return loss
        