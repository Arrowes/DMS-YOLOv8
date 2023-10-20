import torch 
import cv2
import numpy as np
import torchvision
import os
import sys
from torch import nn
import math
sys.path.append('/home/jliu8')
sys.path.append('/home/jliu8/proj6037_seed')
from proj6037_seed.YOLOV8.block import *
from proj6037_seed.YOLOV8.block import L0
from proj6037_seed.YOLOV8.block import C2f
from proj6037_seed.YOLOV8.block import Conv
'''
YOLOv8模型网络结构图
https://blog.csdn.net/u014297502/article/details/128614946

'''




class YOLOv8(nn.Module):
    def __init__(self,backbone=None,head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOv8PAFPN()
        if head is None:
            head  = YOLOv8Head(num_classes=1)
        self.backbone = backbone
        self.head = head

    def forward(self, image):

        fpn_outs = self.backbone(image)

        outputs = self.head(fpn_outs)

        return outputs




class YOLOv8PAFPN(nn.Module):
    def __init__(
            self,
            params,
            width=1.0,
            depth=1.0,
            ratio=1.0,
            in_features = ("p4", "p6", "p9"),
            in_channels=512
    ):
        super().__init__()
        self.is_deploy = params.is_BNC
        self.in_channels = in_channels
        self.in_features = in_features
        self.backbone = L0(width=width,depth=depth,ratio=ratio)

        if params.upsample_type == 'bilinear': 
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        else: # default
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest") 
        
        self.c2f12 = C2f(in_channels*width*(1+ratio),in_channels*width,shortcut=False,n=int(3*depth))
        self.upconv = Conv(c1=in_channels*width,c2=in_channels//2 * width,k=3,s=1,p=1)
        self.c2f15 = C2f(c1=in_channels*width,c2=in_channels*width//2,shortcut=False,n=int(3*depth))
        self.conv16 = Conv(in_channels*width//2,in_channels*width//2,k=3,s=2,p=1)
        self.c2f18 = C2f(c1=(in_channels+in_channels//2)*width,c2=in_channels*width,shortcut=False,n=int(3*depth))
        self.conv19 = Conv(in_channels*width,in_channels*width,k=3,s=2,p=1)
        self.c2f21 = C2f(c1=in_channels*width*(1+ratio),c2=in_channels*width)


    def forward(self,input):
        '''
        input:输入的图片
        return: FPN feature.
        '''
        if self.is_deploy :
            input = input / 255

        # backbone
        out_features = self.backbone(input) 
        features = [out_features[f] for f in self.in_features]
        [x2,x1,x0] = features
        f_out0 = x0 #p20 concat输入
        # fpn_out0 = x0 #p9 输出
        p10 = self.upsample(x0) #p10 输出
        p11 = torch.concat([p10,x1],1)
        p12 = self.c2f12(p11)
        f_out1 = p12
        p13 = self.upsample(p12)
        p13 = self.upconv(p13)
        p14 = torch.concat([p13,x2],1)
        p15 = self.c2f15(p14)
        f_out2 = p15
        p16 = self.conv16(p15)
        p17 = torch.concat([p16,f_out1],1)
        p18 = self.c2f18(p17)
        p19 = self.conv19(p18)
        p20 = torch.concat([p19,f_out0],1)
        p21 = self.c2f21(p20)
        return p21
    

class YOLOv8Head(nn.Module):
    # 借用YOLOx的检测头
    def __init__(
            self,
            params,
            num_classes,
            head_width=1.0,
            yolo_width=1.0,
            in_channels=[256, 512, 1024],
            act="silu"
    ):
        super().__init__()
        self.params = params
        self.n_anchors = 1
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.cls_convs = nn.Sequential(*[
            Conv(
                int(256 * head_width),
                int(256 * head_width),
                k=3,
                s=1,
                
            ),
            Conv(
                int(256 * head_width),
                int(256 * head_width),
                k=3,
                s=1,
                
            ),
        ])
        self.reg_convs = nn.Sequential(*[
            Conv(
                int(256 * head_width),
                int(256 * head_width),
                k=3,
                s=1,
                
            ),
            Conv(
                int(256 * head_width),
                int(256 * head_width),
                k=3,
                s=1,
                
            ),
        ])
        self.entryline_preds = nn.Conv2d(
            in_channels=int(256 * head_width),
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sepline_preds = nn.Conv2d(
            in_channels=int(256 * head_width),
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.reg_preds = nn.Conv2d(
            in_channels=int(256 * head_width),
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.obj_preds = nn.Conv2d(
            in_channels=int(256 * head_width),
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.occ_preds = nn.Conv2d(
            in_channels=int(256 * head_width),
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.stems = Conv(
            int(self.in_channels[1] * yolo_width),
            int(256 * head_width),
            k=1,
            s=1,
            
        )

    def initialize_biases(self, prior_prob):

        b = self.reg_preds.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.reg_preds.bias = torch.nn.Parameter(b.view(-1),
                                                 requires_grad=True)

        b = self.entryline_preds.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.entryline_preds.bias = torch.nn.Parameter(b.view(-1),
                                                 requires_grad=True)

        b = self.sepline_preds.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.sepline_preds.bias = torch.nn.Parameter(b.view(-1),
                                                 requires_grad=True)

        b = self.obj_preds.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.obj_preds.bias = torch.nn.Parameter(b.view(-1),
                                                 requires_grad=True)

        b = self.occ_preds.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.occ_preds.bias = torch.nn.Parameter(b.view(-1),
                                                 requires_grad=True)
        
    def forward(self, x):

        x = self.stems(x)
        cls_x = x
        reg_x = x

        cls_feat = self.cls_convs(cls_x)
        entryline_output = self.entryline_preds(cls_feat)  # 2 channel
        sepline_output = self.sepline_preds(cls_feat)  # 2 channel

        reg_feat = self.reg_convs(reg_x)
        position_output = self.reg_preds(reg_feat)  # 2 channel
        confidence_output = self.obj_preds(reg_feat)# 1 channel
        occupied_output = self.occ_preds(reg_feat)  # 1 channel

        if self.params.without_exp:
            output = torch.cat([
                confidence_output,  # (0, 1)  confidence
                position_output,    # (0, 1)  offset x, y
                sepline_output,     # (-1, 1) sepline_x和y
                entryline_output,   # (-1, 1) entryline_x和y
                occupied_output,    # (0, 1)  occupied
            ], 1)
        else:

            output = torch.cat([
                confidence_output.sigmoid(),# (0, 1)  confidence
                position_output.sigmoid(),  # (0, 1)  offset x, y
                # sepline_output.tanh(),      # (-1, 1) sepline_x和y
                sepline_output.tanh(),      # (-1, 1) sepline_x和y
                # entryline_output.tanh(),    # (-1, 1) entryline_x和y
                entryline_output.tanh(),    # (-1, 1) entryline_x和y
                occupied_output.sigmoid(),  # (0, 1)  occupied
            ], 1)


        if self.params.is_BNC:
            b, c, h, w = output.shape
            output = output.reshape(b, c, -1).permute(0, 2, 1)

        return output

def get_model(params):
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    backbone = YOLOv8PAFPN(params,
                         params.yolo_width,
                         params.yolo_depth,
                         ratio=1
                         )
    head = YOLOv8Head(params,
                     num_classes=1,
                     in_channels=params.in_channels,
                     head_width=params.head_width,
                     yolo_width=params.yolo_width,
                     )
    model = YOLOv8(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    return model








        


        