from . import Decoder as network
from .Resnet import resnet34
import torch
import torch.nn as nn
from .MobileNetv3 import mobilenetv3_large
from .DepthSolver import DepthSolver
from .convnext import convnext_tiny

class DepthModel(nn.Module):
    def __init__(self,args):
        super(DepthModel, self).__init__()
        if args.backbone == 'resnet34':
            self.encoder_modules = resnet34()
        elif args.backbone == 'mobilenetv3':
            self.encoder_modules = mobilenetv3_large()
        else:
            self.encoder_modules = convnext_tiny()
        
        self.args = args
        self.decoder_modules = network.Decoder(args)
        self.solver = DepthSolver(args)

        self.export_onnx = args.export_onnx
        self.encoder_infer = False

    def forward(self, x):
        out_base = []
        if not self.export_onnx:
            sd = x[:,2:3,:,:].clone()

        if self.training:        
            x[:,2:3] = x[:,1:2,:,:]

        if self.encoder_infer: # encoder module should be set to eval mode
            with torch.no_grad():
                lateral_out = self.encoder_modules(x)
        else:
            lateral_out = self.encoder_modules(x)
        basis,confidence = self.decoder_modules(lateral_out)

        if self.export_onnx:
            return torch.cat([basis,confidence],dim=1)
        out,w_sum = self.solver(sd,basis)
        
        out = torch.clamp(out,1e-6)  * 10 # n,10,H,W

        if self.training:
            return out,confidence,w_sum
        return out,confidence,out_base

