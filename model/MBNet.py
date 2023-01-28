from . import Decoder as network
from .Resnet import resnet34
import torch
import torch.nn as nn
from .MobileNetv3 import mobilenetv3_large
from .DepthSolver import solve
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

        # base_sd = sd_map.clone().expand(sd_map.shape[0],2,sd_map.shape[2],sd_map.shape[3])
        # pe = positionalencodingxy(base_sd) # 2 X H X W
        # base_sd1 = []
        # for i in range(base_sd.shape[0]):
        #     sd0 = base_sd[i,0] # H X W
        #     mask = sd0 > 0 # H X W
        #     sd0_mask = sd0[mask] # 125
        #     pe_s = pe.permute(2,1,0)[mask] # 125 X 2
        #     pe_d = pe.view(pe.shape[0],-1).T.unsqueeze(1).expand(224*224,pe_s.shape[0],2) # HW X 125 X 2
        #     kv =  (pe_d - pe_s).pow(2).sum(-1).sqrt().T
        #     kv = -kv + 1/(kv + 1e-1) * 10
        #     kv = self.softmax(kv)  # 125 X HW
        #     inter = sd0_mask.view(1,sd0_mask.shape[0]) @ kv # 1 X HW
        #     # out.append(inter.view(1,1,sd_map.shape[2],sd_map.shape[3]))
        #     base_sd1.append(inter.view(1,1,sd_map.shape[2],sd_map.shape[3]))


        # base_sd1 = torch.cat(base_sd1,dim=0)
        # out = torch.clamp(out,1e-6)  * 10 # n,10,H,W
        # return out,None,None


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

