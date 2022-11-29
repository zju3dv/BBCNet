from . import Decoder as network
from .Resnet import resnet34
import torch
import torch.nn as nn
from .MobileNetv3 import mobilenetv3_large
from .DepthSolver import solve

class DepthModel(nn.Module):
    def __init__(self,args):
        super(DepthModel, self).__init__()
        if args.backbone == 'resnet34':
            self.encoder_modules = resnet34()
        elif args.backbone == 'mobilenetv3':
            self.encoder_modules = mobilenetv3_large()
        else:
            raise "Not valid backbone"
        
        self.decoder_modules = network.Decoder(args)
        if args.slam_infer:
            self.down1 = nn.AvgPool2d(2,2,0)
            self.down2 = nn.MaxPool2d(2,2)
        else:
            self.down1 = nn.Identity()
            self.down2 = nn.Identity()
        

    def forward(self, x):
        out_base = []
        if(x.shape[1] == 3):
            sd = x[:,2:3,:,:].clone()
        else:
            sd = x[:,1:2,:,:].clone()

        rgb = self.down1(x[:,0:1,:,:])
        sd_map = self.down2(x[:,1:2,:,:])
        # mask = sd_map  * 0.5 * 10
        # coeff1 = sd_map * (mask<2.5)
        # coeff2 = sd_map * (mask>=2.5)

        x = torch.cat([rgb,sd_map,sd_map],dim=1)

        lateral_out = self.encoder_modules(x)
        basis,confidence = self.decoder_modules(lateral_out)

        if False:
            out = self.upsample(torch.cat([x,confidence],dim=1))
            return out
        confidence = confidence[:,0,:,:]
        out_prior = basis.sum(dim=1).view(x.shape[0],1,x.shape[2],x.shape[3])
        out_prior = torch.clamp(out_prior,1e-6) * 10
        
        out,w_sum = solve(sd,basis)
        
        out = torch.clamp(out,1e-6)  * 10 # n,10,H,W

        if self.training:
            w_sum = w_sum / x.shape[0]
            return out,out_prior,confidence,w_sum
        return out,confidence,out_base

