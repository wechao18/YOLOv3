from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib.pyplot as plt
from util import *


class LPConv(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride=1, padding=0, bias=True):
        super(LPConv, self).__init__()
        # self.add_module('conv1', nn.Sequential(
        #     nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding, bias),
        #     nn.BatchNorm2d(num_features=16),
        #     nn.LeakyReLU(0.1, inplace=True),
        # ))
        self.lp_conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(num_features=outchannel),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        x1 = self.lp_conv(x)
        return x1


class ResBlock(nn.Module):
    def __init__(self, channel1, channel2, bias=True):
        super(ResBlock, self).__init__()

        self.res_conv1 = nn.Sequential(
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(num_features=channel1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(num_features=channel2),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        x1 = self.res_conv1(x)
        x1 = self.res_conv2(x1)
        x1 = x1 + x
        return x1


class UpsampleBlock(nn.Module):
    def __init__(self, channel1, channel2, bias=True):
        super(UpsampleBlock, self).__init__()

        self.Upsample_conv = nn.Sequential(
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(num_features=channel1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x, feature):
        x = self.Upsample_conv(x)
        x = torch.cat((x, feature), 1)
        return x

class DetectionBlock(nn.Module):
    def __init__(self, channel1, channel2, channel3=0, bias=True):
        super(DetectionBlock, self).__init__()
        self.detection_conv1 = nn.Sequential(
            nn.Conv2d(channel2+channel3, channel1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_features=channel1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=channel2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_features=channel1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=channel2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_features=channel1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.detection_conv2 = nn.Sequential(
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=channel2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel2, 255, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        x1 = self.detection_conv1(x)
        x2 = self.detection_conv2(x1)
        return x1, x2

class YOLOV3(nn.Module):
    def __init__(self):
        super(YOLOV3, self).__init__()
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.conv_1 = LPConv(3, 32, 3, 1, 1, bias=False)

        self.conv_2 = LPConv(32, 64, 3, 2, 1, bias=False)
        self.res1 = ResBlock(32, 64, bias=False)

        self.conv_3 = LPConv(64, 128, 3, 2, 1, bias=False)
        self.res2_1 = ResBlock(64, 128, bias=False)
        self.res2_2 = ResBlock(64, 128, bias=False)

        self.conv_4 = LPConv(128, 256, 3, 2, 1, bias=False)
        self.res3_1 = ResBlock(128, 256, bias=False)
        self.res3_2 = ResBlock(128, 256, bias=False)
        self.res3_3 = ResBlock(128, 256, bias=False)
        self.res3_4 = ResBlock(128, 256, bias=False)
        self.res3_5 = ResBlock(128, 256, bias=False)
        self.res3_6 = ResBlock(128, 256, bias=False)
        self.res3_7 = ResBlock(128, 256, bias=False)
        self.res3_8 = ResBlock(128, 256, bias=False)

        self.conv_5 = LPConv(256, 512, 3, 2, 1, bias=False)
        self.res4_1 = ResBlock(256, 512, bias=False)
        self.res4_2 = ResBlock(256, 512, bias=False)
        self.res4_3 = ResBlock(256, 512, bias=False)
        self.res4_4 = ResBlock(256, 512, bias=False)
        self.res4_5 = ResBlock(256, 512, bias=False)
        self.res4_6 = ResBlock(256, 512, bias=False)
        self.res4_7 = ResBlock(256, 512, bias=False)
        self.res4_8 = ResBlock(256, 512, bias=False)


        self.conv_6 = LPConv(512, 1024, 3, 2, 1, bias=False)
        self.res5_1 = ResBlock(512, 1024, bias=False)
        self.res5_2 = ResBlock(512, 1024, bias=False)
        self.res5_3 = ResBlock(512, 1024, bias=False)
        self.res5_4 = ResBlock(512, 1024, bias=False)

        self.detection3 = DetectionBlock(512, 1024, bias=False)
        self.concat3 = UpsampleBlock(256, 512, bias=False)
        self.detection2 = DetectionBlock(256, 512, 256, bias=False)
        self.concat2 = UpsampleBlock(128, 256, bias=False)
        self.detection1 = DetectionBlock(128, 256, 128, bias=False)

        self.feature1 = nn.Sequential(
            self.conv_1, self.conv_2, self.res1,
            self.conv_3, self.res2_1, self.res2_2,
            self.conv_4, self.res3_1, self.res3_2, self.res3_3, self.res3_4, self.res3_5, self.res3_6, self.res3_7, self.res3_8,
        )

        self.feature2 = nn.Sequential(
            self.conv_5, self.res4_1, self.res4_2, self.res4_3, self.res4_4, self.res4_5, self.res4_6, self.res4_7, self.res4_8,
        )
        self.feature3 = nn.Sequential(
            self.conv_6, self.res5_1, self.res5_2, self.res5_3, self.res5_4,
        )

        self.module_list = nn.ModuleList()
        self.module_list.append(self.feature1)
        self.module_list.append(self.feature2)
        self.module_list.append(self.feature3)
        self.module_list.append(self.detection3)
        self.module_list.append(self.concat3)
        self.module_list.append(self.detection2)
        self.module_list.append(self.concat2)
        self.module_list.append(self.detection1)

    def forward(self, x, CUDA, labels=None, lp_labels=None):

        feature1 = self.feature1(x)
        feature2 = self.feature2(feature1)
        feature3 = self.feature3(feature2)
        feature3, det3 = self.detection3(feature3)
        feature2 = self.concat3(feature3, feature2)
        feature2, det2 = self.detection2(feature2)
        feature1 = self.concat2(feature2, feature1)
        feature1, det1 = self.detection1(feature1)

        x3 = det3.data
        x3 = predict_transform(x3, 416, [(116,90),  (156,198),  (373,326)], 80, CUDA)
        x2 = det2.data
        x2 = predict_transform(x2, 416, [(30,61),  (62,45),  (59,119)], 80, CUDA)
        x1 = det1.data
        x1 = predict_transform(x1, 416, [(10,13),  (16,30),  (33,23)], 80, CUDA)
        detections = torch.cat((x3, x2, x1), 1)

        return detections


    def load_weights(self, weightfile):

        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. IMages seen
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype=np.float32)
        num = 1
        ptr = 0
        for i in range(len(self.module_list)):
            model_unit = self.module_list[i]
            if i < 3:
                for model_class in model_unit:
                    class_name = str(model_class.__class__).split('.')[-1].split("'")[0]
                    # print(class_name)
                    #if class_name == 'LPConv' or class_name == 'ResBlock' or class_name== 'UpsampleBlock':
                    for model in model_class.children():
                        #print(len(model))
                        if (len(model) > 1):
                            #print('load_bn')
                            ptr = self.load_weights_bn(model, ptr, weights)
                            print(num,ptr)
                            num = num +1
            else:
                #print(model_unit)
                class_name = str(model_unit.__class__).split('.')[-1].split("'")[0]
                if class_name == 'UpsampleBlock':
                    for model in model_unit.children():
                        #print(len(model))
                        if (len(model) > 1):
                            ptr = self.load_weights_bn(model, ptr, weights)
                            print(num, ptr)
                            num += 1
                else:
                    for model in model_unit.children():
                        if len(model) > 5:
                            for j in range(5):
                                ptr = self.load_weights_bn(model[3*j:3*(j+1)], ptr, weights)
                                print(num, ptr)
                                num += 1
                        else:
                            model1 = model[0:3]
                            model2 = model[3]
                            ptr = self.load_weights_bn(model1, ptr, weights)
                            print(num, ptr)
                            num += 1
                            ptr = self.load_weights_nobn(model2, ptr, weights)
                            print(num, ptr)
                            num += 1



    def load_weights_bn(self, model, ptr, weights):
        conv = model[0]
        bn = model[1]

        # Get the number of weights of Batch Norm Layer
        num_bn_biases = bn.bias.numel()

        # Load the weights
        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
        ptr += num_bn_biases

        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr += num_bn_biases

        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr += num_bn_biases

        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr += num_bn_biases

        # Cast the loaded weights into dims of model weights.
        bn_biases = bn_biases.view_as(bn.bias.data)
        bn_weights = bn_weights.view_as(bn.weight.data)
        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
        bn_running_var = bn_running_var.view_as(bn.running_var)

        # Copy the data to model
        bn.bias.data.copy_(bn_biases)
        bn.weight.data.copy_(bn_weights)
        bn.running_mean.copy_(bn_running_mean)
        bn.running_var.copy_(bn_running_var)

        # Let us load the weights for the Convolutional layers
        num_weights = conv.weight.numel()
        # Do the same as above for weights
        conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
        ptr = ptr + num_weights

        conv_weights = conv_weights.view_as(conv.weight.data)
        conv.weight.data.copy_(conv_weights)

        return ptr

    def load_weights_nobn(self, model, ptr, weights):

        try:
            conv = model[0]
        except:
            conv = model
        # Number of biases
        num_biases = conv.bias.numel()

        # Load the weights
        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
        ptr = ptr + num_biases

        # reshape the loaded weights according to the dims of the model weights
        conv_biases = conv_biases.view_as(conv.bias.data)

        # Finally copy the data
        conv.bias.data.copy_(conv_biases)

        # Let us load the weights for the Convolutional layers
        num_weights = conv.weight.numel()

        # Do the same as above for weights
        conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
        ptr = ptr + num_weights

        conv_weights = conv_weights.view_as(conv.weight.data)
        conv.weight.data.copy_(conv_weights)

        return ptr

#model = YOLOV3()
#model.load_weights("model/yolov3.weights")

