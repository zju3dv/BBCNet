from . import Decoder as network
from .Resnet import resnet34
import torch
import torch.nn as nn
from .MobileNetv3 import mobilenetv3_large
from .DepthSolver import DepthSolver
from .convnext import convnext_tiny
from torch.cuda.amp import autocast, GradScaler

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
        self.solver = DepthSolver()

        self.encoder_infer = False
        if args.with_confidence:
            self.encoder_infer = True
            self.encoder_modules.requires_grad_(False)

    def forward(self, x):
        out_base = []
        sd = x[:,2:3,:,:].clone()

        if self.training:        
            rgb = self.down1(x[:,0:1,:,:])
            sd_map = self.down2(x[:,1:2,:,:])
            x = torch.cat([rgb,sd_map,sd_map],dim=1)
        if self.encoder_infer:
            with torch.no_grad():
                lateral_out = self.encoder_modules(x)
        else:
            lateral_out = self.encoder_modules(x)
        basis,confidence = self.decoder_modules(lateral_out)

        return torch.cat([basis,confidence],dim=1)
        with autocast(dtype=torch.float32):
            out,w_sum = self.solver(sd,basis)
        
        out = torch.clamp(out,1e-6)  * 10 # n,10,H,W

        if self.training:
            return out,confidence,w_sum
        return out,confidence,out_base

