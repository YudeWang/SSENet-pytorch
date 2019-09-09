import torch
import torch.nn as nn
import torch.nn.functional as F

import network.vgg16d

class Net(network.vgg16d.Net):

    def __init__(self):
        super(Net, self).__init__()

        self.drop7 = nn.Dropout2d(p=0.5)
        self.fc8 = nn.Conv2d(1024, 20, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.not_training = [self.conv1_1, self.conv1_2,
                             self.conv2_1, self.conv2_2]
        self.from_scratch_layers = [self.fc8]

    def forward(self, x):
        N, C, H, W = x.size()
        x = super().forward(x)
        x = self.drop7(x)
        x = self.fc8(x)
        x = F.interpolate(x,(H,W),mode='bilinear')

        return x

    def fix_bn(self):
        self.bn8.eval()
        self.bn8.weight.requires_grad = False
        self.bn8.bias.requires_grad = False

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)):

                if m.weight is not None and m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
