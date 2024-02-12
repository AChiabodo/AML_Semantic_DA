#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Function
from .stdcnet import STDCNet813


BatchNorm2d = nn.BatchNorm2d

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        
        # Creating a convolutional layer
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        
        # Batch normalization layer to normalize the output of the convolutional layer
        self.bn = BatchNorm2d(out_chan)
        
        # ReLU activation function to introduce non-linearity
        self.relu = nn.ReLU()
        
        # Initialize the weights of the layers
        self.init_weight()

    def forward(self, x):
        # Forward pass through layers: convolution -> batch normalization -> ReLU activation
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        # Initializing weights using Kaiming Normal initialization for convolutional layers
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

            
class BiSeNetDiscriminator(nn.Module):
    class DiscriminatorConv(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            # Creating a convolutional block with leaky ReLU activation
            self.conv = nn.Sequential(
                nn.Conv2d(*args, **kwargs),  # Setting up a 2D convolutional layer
                nn.LeakyReLU(0.2, inplace=True)  # Applying Leaky ReLU activation for non-linearity
            )
            
        def forward(self, x):
            return self.conv(x)  # Perform convolutional operation on input x

    def __init__(self, num_classes, alpha=0.1, *args, **kwargs):
        super(BiSeNetDiscriminator, self).__init__()
        self.alpha = alpha
        
        # Sequential layers for the discriminator architecture
        self.discriminator = nn.Sequential(
            # Series of convolutional layers with leaky ReLU activations
            nn.Conv2d(num_classes, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final convolutional layer for output generation
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),
            nn.Upsample(scale_factor=32, mode='bilinear')  # Upsampling the output
        )
        
    def forward(self, x):
        #x = ReverseLayer.apply(x, self.alpha)
        x = self.discriminator(x)  # Pass input through the discriminator layers
        return x
    
    def train_params(self, requires_grad=True):
        # Function to set requires_grad attribute of parameters for training
        for param in self.parameters():
            param.requires_grad = requires_grad



class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()

        # Convolutional block for feature transformation
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)

        # 1x1 convolution for attention computation
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)

        # Batch normalization for attention map
        # self.bn_atten = BatchNorm2d(out_chan)
        self.bn_atten = BatchNorm2d(out_chan)

        # Sigmoid activation to compute attention weights
        self.sigmoid_atten = nn.Sigmoid()

        # Initializing weights for layers
        self.init_weight()

    def forward(self, x):
        # Feature transformation through convolutional block
        feat = self.conv(x)

        # Calculating attention map by performing average pooling and applying a 1x1 convolution
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)

        # Applying attention to the features
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        # Initializing weights of convolutional layers using Kaiming Normal initialization
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


    
    
class ContextPath(nn.Module):
    def __init__(self, backbone='STDCNet813', pretrain_model='', use_conv_last=False, *args, **kwargs):
        super(ContextPath, self).__init__()

        # Initialize the ContextPath module with a chosen backbone
        self.backbone = STDCNet813(pretrain_model=pretrain_model, use_conv_last=use_conv_last)

        # Attention Refinement Modules (ARM) for different feature maps
        self.arm16 = AttentionRefinementModule(512, 128)  # ARM for feature map at 1/16 scale
        inplanes = 1024
        if use_conv_last:
            inplanes = 1024
        self.arm32 = AttentionRefinementModule(inplanes, 128)  # ARM for feature map at 1/32 scale
        
        # Convolutional heads for merging features from ARM modules
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)  # Average pooling and convolution

        # Initialize weights
        self.init_weight()

    def forward(self, x):
        # Obtain the sizes of input feature map
        H0, W0 = x.size()[2:]

        # Extract feature maps from the backbone network
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        # Average pooling on the feature map at 1/32 scale
        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)  # Apply convolution to the averaged feature map
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')  # Upsample the averaged feature map

        # Process feature maps with Attention Refinement Modules (ARMs)
        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up  # Combine ARM output with the upsampled average
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat2, feat4, feat8, feat16, feat16_up, feat32_up  # Return selected feature maps at different scales

    def init_weight(self):
        # Initialize weights of convolutional layers using Kaiming Normal initialization
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        # Separate weights and biases for weight decay and no weight decay respectively
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params  # Return parameters for weight decay and no weight decay



class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                               out_chan // 4,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4,
                               out_chan,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, backbone, n_classes, pretrain_model='', use_boundary_2=False, use_boundary_4=False,
                 use_boundary_8=False, use_boundary_16=False, use_conv_last=False, heat_map=False, *args, **kwargs):
        super(BiSeNet, self).__init__()

        # self.heat_map = heat_map
        self.cp = ContextPath(backbone, pretrain_model, use_conv_last=use_conv_last)

        conv_out_inplanes = 128
        sp2_inplanes = 32
        sp4_inplanes = 64
        sp8_inplanes = 256
        sp16_inplanes = 512
        inplane = sp8_inplanes + conv_out_inplanes


        self.ffm = FeatureFusionModule(inplane, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)
        #self.discr_out = BiSeNetDiscriminator(n_classes, 256, 2)
        #self.discr_out16 = BiSeNetDiscriminator(n_classes, 64, 2)
        #self.discr_out32 = BiSeNetDiscriminator(n_classes, 64, 2)

        self.init_weight()

    def forward(self, x):

        print("x.size(): ", x.size())
        
        H, W = x.size()[2:]

        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)

        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)
        # if self.training:
        #     domain_out = self.discr_out(feat_out)
        #     domain_out16 = self.discr_out16(feat_out16)
        #     domain_out32 = self.discr_out32(feat_out32)
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True) 
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)

        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params



class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None