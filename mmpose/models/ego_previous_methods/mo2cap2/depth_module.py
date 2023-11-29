#  Copyright Jian Wang @ MPI-INF (c) 2023.


import torch
import torch.nn as nn

BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, expansion=2, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FinalLayer(nn.Module):
    def __init__(self, input_features=256):
        super(FinalLayer, self).__init__()
        self.final_conv1 = nn.Conv2d(in_channels=input_features, out_channels=128, kernel_size=4, stride=2, padding=1,
                                     bias=False)
        self.final_bn1 = nn.BatchNorm2d(num_features=128)
        self.final_relu1 = nn.ReLU()
        self.final_conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True)
        self.final_relu2 = nn.ReLU()
        self.final_depth = nn.Linear(in_features=16 * 64, out_features=15, bias=True)

    def forward(self, x):
        x = self.final_conv1(x)
        x = self.final_bn1(x)
        x = self.final_relu1(x)
        x = self.final_conv2(x)
        x = self.final_relu2(x)
        x = x.view(size=(-1, 16 * 64))
        x = self.final_depth(x)
        return x


class DepthModule(nn.Module):
    def __init__(self, input_features):
        super(DepthModule, self).__init__()
        downsample1 = nn.Sequential(
                nn.Conv2d(input_features, 512 * 2,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512 * 2, momentum=BN_MOMENTUM),
            )
        self.resblock1 = Bottleneck(inplanes=input_features, planes=512, expansion=2, downsample=downsample1)
        downsample2 = nn.Sequential(
                nn.Conv2d(512 * 2, 256 * 1,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256 * 1, momentum=BN_MOMENTUM),
            )
        self.init_weights(self.resblock1)
        self.resblock2 = Bottleneck(inplanes=1024, planes=256, expansion=1, downsample=downsample2)
        self.init_weights(self.resblock2)
        self.final_layer = FinalLayer(input_features=256)
        self.init_weights(self.final_layer)

    def init_weights(self, network):
        for m in network.children():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, layer3_mid_out, layer4_mid_out, zoom_layer3_mid_out, zoom_layer4_mid_out):
        x = torch.cat([layer3_mid_out, layer4_mid_out, zoom_layer3_mid_out, zoom_layer4_mid_out], dim=1)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.final_layer(x)
        return x
