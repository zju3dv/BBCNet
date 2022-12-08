import torch
import torch.nn as nn
import torch.nn.init as init
from .MobileNetv3 import InvertedResidual
from .Resnet import BasicBlock

def positionalencoding2d(input):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    d_model, height, width = input.shape[1::]
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width,device = input.device)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(9.21034 / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def make_block(inchannels,midchannels,backbone='resnet34'):
    if backbone == 'resnet34':
        downsample = None
        if inchannels !=midchannels:
            downsample = nn.Sequential(
                nn.Conv2d(inchannels, midchannels,
                            kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(midchannels), #NN.BatchNorm2d
            )
        block = BasicBlock(inchannels,midchannels,downsample = downsample)
        return block
    else:
        block = InvertedResidual(inchannels,inchannels * 2,midchannels,3,1,False,True)
        return block


class Decoder(nn.Module):
    def __init__(self,args):
        super(Decoder, self).__init__()
        if(args.backbone == 'resnet34'):
            self.inchannels =  [64, 128, 256, 512] # resnet
            self.midchannels = [16, 128, 256, 512]
            self.initial_fusion_out = 64
        elif args.backbone == 'mobilenetv3':
            self.inchannels =  [24, 40, 112, 160] #mobilenet
            self.midchannels = [24, 40, 112, 160]
            self.initial_fusion_out = 16
        else:
            raise "no implementation"

        self.upfactors = [2,2,2,2]
        self.outchannels = 24
        self.args = args

        self.conv = make_block(inchannels=self.inchannels[3], midchannels=self.midchannels[3])
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=1, stride=1, bias=True)
        # self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=2, stride=1, dilation = 2, bias=True)
        self.upsample = nn.Upsample(scale_factor=self.upfactors[3], mode='bilinear', align_corners=False)

        self.ffmp2 = FFM(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels = self.midchannels[1], upfactor=self.upfactors[2])
        self.ffmp1 = FFM(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels = self.midchannels[0], upfactor=self.upfactors[1])
        self.ffmp0 = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0], upfactor=self.upfactors[0])
        self.ffmp = FFM(self.midchannels[0], midchannels=self.midchannels[0], outchannels = self.outchannels, upfactor=1)
        # self.ffmnp1 = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0], upfactor=self.upfactors[0])
        
        # self.outconv = nn.Sequential(
        #     make_block(self.initial_fusion_out + self.midchannels[0], midchannels=self.outchannels,backbone=args.backbone),
        #     make_block(self.outchannels, midchannels=self.outchannels,backbone=args.backbone)
        # )
        self.adapt_conv_short = nn.Sequential(
            nn.BatchNorm2d(num_features=self.outchannels ),
            nn.Conv2d(self.outchannels , 24 + 18, kernel_size=1, stride=1, padding=0,bias=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(24 + 18,32 , kernel_size=1, stride=1, padding=0,bias=False),
            # nn.ReLU(inplace=True),
        )
        if self.args.with_confidence:
            self.adapt_conv1 = nn.Sequential(
                make_block(self.outchannels, midchannels=self.outchannels,backbone=args.backbone),
                nn.BatchNorm2d(num_features=self.outchannels),
                nn.Conv2d(self.outchannels, 24, kernel_size=1, stride=1, padding=0,bias=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(24,1 , kernel_size=1, stride=1, padding=0,bias=False),
                nn.Sigmoid(),
            )

        self.n_infer = 0

        self._init_params()
        
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): #NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, features):
        x_32x = self.conv(features[3])  # 1/32
        x_32 = self.conv1(x_32x)

        x_16 = x_32#self.upsample(x_32)  # 1/16
        x_8 = self.ffmp2(features[2], x_16)  # 1/8

        x_4 = self.ffmp1(features[1], x_8)  # 1/4
        x_4 = positionalencoding2d(x_4) + x_4 # down

        x_2 = self.ffmp0(features[0], x_4)  # 1/2
        # x_2 = self.outconv(torch.cat((x_2,features[0]),dim=1))  # 1/2
        x_2_1 = self.ffmp(x_2,x_2)  # 1/2

        x = self.adapt_conv_short(x_2_1)
        confidence = torch.zeros_like(x[:,0:1])
        if self.args.with_confidence:
            confidence = self.adapt_conv1(x_2_1)
        return x,confidence

class FFM(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, upfactor=2):
        super(FFM, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.ftb1 = make_block(inchannels=self.inchannels , midchannels=self.midchannels)
        self.ftb2 = make_block(inchannels=self.midchannels, midchannels=self.outchannels)

        self.upsample = nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=False)

        self.init_params()

    def forward(self, low_x, high_x):
        x = self.ftb1(low_x)
        x = x + high_x
        x = self.ftb2(x)
        x = self.upsample(x)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)