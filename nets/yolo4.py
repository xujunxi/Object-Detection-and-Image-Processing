from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.CSPdarknet import darknet53
from nets.mobilenet_v3 import mobilenet_v3

#Extract 3 featured layers from MobileNet
class MobileNetV3(nn.Module):
    def __init__(self, pretrained = False):
        super(MobileNetV3, self).__init__()
        self.model = mobilenet_v3(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:13](out3)
        out5 = self.model.features[13:16](out4)
        return out3, out4, out5

#CSPDarkNet Backbone Conv Blocks    
def conv2d_CSP(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
        ("SAM", SAM(filter_out)), ##########################
        ("DropBlock", DropBlock2D())])) ##########################

#MobileNetv3 Backbone Conv Blocks
def conv2d_MOB(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True))]))

#Depthwise Separable Conv Structure
def conv_dw(filter_in, filter_out, stride = 1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),
        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True))

#SPP Structure
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        return features
    
#SAM Structure
class SAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1)
        
    def forward(self, x):
        spatial_features = self.conv(x)
        attention = torch.sigmoid(spatial_features)
        return attention.expand_as(x) * x

#DropBlock Structure    
class DropBlock2D(nn.Module):
    def __init__(self, keep_prob=0.9, block_size=7):
        super(DropBlock2D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input):
        # print("Before: ", torch.isnan(input).sum())
        if not self.training or self.keep_prob == 1:
            return input
        gamma = (1. - self.keep_prob) / self.block_size ** 2
        for sh in input.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(input) * gamma).to(device=input.device)
        Msum = F.conv2d(M,
                        torch.ones((input.shape[1], 1, self.block_size, self.block_size)).to(device=input.device,
                                                                                             dtype=input.dtype),
                        padding=self.block_size // 2,
                        groups=input.shape[1])
        mask = (Msum < 1).to(device=input.device, dtype=input.dtype)
        # print("After: ", torch.isnan(input * mask * mask.numel() /mask.sum()).sum())
        return input * mask * mask.numel() / mask.sum() 
    
#PANet Conv + Upsampling
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, lite=True):
        super(Upsample, self).__init__()
        if lite:
            self.upsample = nn.Sequential(
                conv2d_MOB(in_channels, out_channels, 1),
                nn.Upsample(scale_factor=2, mode='nearest'))
        else:
            self.upsample = nn.Sequential(
                conv2d_CSP(in_channels, out_channels, 1),
                nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, x,):
        x = self.upsample(x)
        return x
    
#Three Conv Block
def make_three_conv(filters_list, in_filters, lite=True):
    if lite:
        m = nn.Sequential(
            conv2d_MOB(in_filters, filters_list[0], 1),
            conv_dw(filters_list[0], filters_list[1]),
            conv2d_MOB(filters_list[1], filters_list[0], 1))
    else:
        m = nn.Sequential(
            conv2d_CSP(in_filters, filters_list[0], 1),
            conv2d_CSP(filters_list[0], filters_list[1], 3),
            conv2d_CSP(filters_list[1], filters_list[0], 1))
    return m

#Five Conv Block
def make_five_conv(filters_list, in_filters, lite=True):
    if lite:
        m = nn.Sequential(
            conv2d_MOB(in_filters, filters_list[0], 1),
            conv_dw(filters_list[0], filters_list[1]),
            conv2d_MOB(filters_list[1], filters_list[0], 1),
            conv_dw(filters_list[0], filters_list[1]),
            conv2d_MOB(filters_list[1], filters_list[0], 1))
    else:
        m = nn.Sequential(
            conv2d_CSP(in_filters, filters_list[0], 1),
            conv2d_CSP(filters_list[0], filters_list[1], 3),
            conv2d_CSP(filters_list[1], filters_list[0], 1),
            conv2d_CSP(filters_list[0], filters_list[1], 3),
            conv2d_CSP(filters_list[1], filters_list[0], 1))
    return m

#YOLO Head
def yolo_head(filters_list, in_filters, lite=True):
    if lite:
        m = nn.Sequential(
            conv_dw(in_filters, filters_list[0]),
            nn.Conv2d(filters_list[0], filters_list[1], 1))
    else:
        m = nn.Sequential(
            conv2d_CSP(in_filters, filters_list[0], 3),
            nn.Conv2d(filters_list[0], filters_list[1], 1))
    return m

#YOLOv4 Body
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes, backbone="mobilenetv3", lite=True, pretrained=False):
        super(YoloBody, self).__init__()
        if backbone == "mobilenetv3":
            self.backbone = MobileNetV3(pretrained=pretrained)
            in_filters = [40,112,160]                    
            self.conv1 = make_three_conv([512,1024],in_filters[2],lite)
            self.SPP = SpatialPyramidPooling()
            self.conv2 = make_three_conv([512,1024],2048,lite)     
            self.upsample1 = Upsample(512,256,lite)
            self.conv_for_P4 = conv2d_MOB(in_filters[1],256,1)
            self.make_five_conv1 = make_five_conv([256, 512],512,lite)   
            self.upsample2 = Upsample(256,128,lite)
            self.conv_for_P3 = conv2d_MOB(in_filters[0],128,1)
            self.make_five_conv2 = make_five_conv([128, 256],256,lite)     
        elif backbone == "CSPdarknet":
            self.backbone = darknet53(None)
            self.conv1 = make_three_conv([512,1024],1024,lite)
            self.SPP = SpatialPyramidPooling()
            self.conv2 = make_three_conv([512,1024],2048,lite)
            self.upsample1 = Upsample(512,256,lite)
            self.conv_for_P4 = conv2d_CSP(512,256,1)
            self.make_five_conv1 = make_five_conv([256, 512],512,lite)
            self.upsample2 = Upsample(256,128,lite)
            self.conv_for_P3 = conv2d_CSP(256,128,1)
            self.make_five_conv2 = make_five_conv([128, 256],256,lite)

        final_out_filter2 = num_anchors * (5 + num_classes)
        self.yolo_head3 = yolo_head([256, final_out_filter2],128,lite) 
        if lite:
            self.down_sample1 = conv_dw(128,256,stride=2)
        else:
            self.down_sample1 = conv2d_CSP(128,256,3,stride=2)
        self.make_five_conv3 = make_five_conv([256, 512],512,lite)
        final_out_filter1 =  num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head([512, final_out_filter1],256,lite)
        if lite:
            self.down_sample2 = conv_dw(256,512,stride=2)
        else:
            self.down_sample2 = conv2d_CSP(256,512,3,stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024],1024,lite)
        final_out_filter0 =  num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head([1024, final_out_filter0],512,lite)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)
        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4,P5_upsample],axis=1)
        P4 = self.make_five_conv1(P4)
        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3,P4_upsample],axis=1)
        P3 = self.make_five_conv2(P3)
        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample,P4],axis=1)
        P4 = self.make_five_conv3(P4)
        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample,P5],axis=1)
        P5 = self.make_five_conv4(P5)
        #Feature Layer (Big) 52x52
        out2 = self.yolo_head3(P3)
        #Feature Layer (Middle) 26x26
        out1 = self.yolo_head2(P4)
        #Feature Layer (Small) 13x13
        out0 = self.yolo_head1(P5)
        return out0, out1, out2