from . import Decoder as network
from .Resnet import resnet34
import torch
import torch.nn as nn
from .MobileNetv3 import mobilenetv3_large
from .DepthSolver import DepthSolver
from .Decoder import positionalencoding2d
from .convnext import convnext_tiny

def positionalencodingxy(input):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    d_model, height, width = input.shape[1::]
    pe = torch.zeros(d_model, height, width,device = input.device)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:1, :, :] = (pos_w).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:2, :, :] = (pos_h).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class DepthModel(nn.Module):
    def __init__(self,args):
        super(DepthModel, self).__init__()
        if args.backbone == 'resnet34':
            self.encoder_modules = resnet34()
        elif args.backbone == 'mobilenetv3':
            self.encoder_modules = mobilenetv3_large()
        else:
            self.encoder_modules = convnext_tiny()
            # raise "Not valid backbone"
        
        self.decoder_modules = network.Decoder(args)
        if args.slam_infer:
            self.down1 = nn.AvgPool2d(2,2,0)
            self.down2 = nn.MaxPool2d(2,2)
        else:
            self.down1 = nn.Identity()
            self.down2 = nn.Identity()
        self.softmax  = nn.Softmax(dim=-2)    
        self.solver = DepthSolver()


    def forward(self, x):
        out_base = []
        if(x.shape[1] == 3):
            sd = x[:,2:3,:,:].clone()
        else:
            sd = x[:,1:2,:,:].clone()



        rgb = self.down1(x[:,0:1,:,:])
        sd_map = self.down2(x[:,1:2,:,:])
        
        x = torch.cat([rgb,sd_map,sd_map],dim=1)

        lateral_out = self.encoder_modules(x)
        basis,confidence = self.decoder_modules(lateral_out)

        if False:
            out = self.upsample(torch.cat([x,confidence],dim=1))
            return out
        # return torch.cat([basis,confidence],dim=1)
        confidence = confidence[:,0,:,:]
        out_prior = basis.sum(dim=1).view(x.shape[0],1,x.shape[2],x.shape[3])
        out_prior = torch.clamp(out_prior,1e-6) * 10
        
        out,w_sum = self.solver(sd,basis)
        
        out = torch.clamp(out,1e-6)  * 10 # n,10,H,W

        if self.training:
            w_sum = w_sum / x.shape[0]
            return out,out_prior,confidence,w_sum
        return out,confidence,out_base

