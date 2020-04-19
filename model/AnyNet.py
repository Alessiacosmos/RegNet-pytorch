# -*- encoding: utf-8 -*-
"""
@File    : AnyNet.py
@Time    : 2020/4/13 14:39
@Author  : Alessia K
@Email   : ------
"""

import torch.nn as nn
# from utils.parse_cfg import load_cfg

class AnyStem(nn.Module):
    """AnyNet stem part"""

    def __init__(self, w_in, w_out):
        super(AnyStem, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn   = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.pool(x)

        return x

class SE(nn.Module):
    """Squeeze-and-Excitation block"""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.se1   = nn.Conv2d(w_in, w_se, kernel_size=1, bias=True)
        self.reluse= nn.ReLU(inplace=True)
        self.se2   = nn.Conv2d(w_se, w_in, kernel_size=1, bias=True)
        self.sm    = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)

        out = self.se1(out)
        out = self.reluse(out)
        out = self.se2(out)
        out = self.sm(out)
        out = x * out

        return out


class ResBottleneckBlock(nn.Module):
    """Block i: Residual bottleneck block: x + F(x)"""

    def __init__(self, w_in, w_out, stride=1, bm=1.0, gw=1, se_ratio=0):
        super(ResBottleneckBlock, self).__init__()

        w_out_b = int(round(w_out*bm))  # width with bottleneck
        num_group = w_out_b // gw       # number of groups
        # 1*1, BN, ReLU
        self.conv1 = nn.Conv2d(w_in, w_out_b, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(w_out_b, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)

        # 3*3, BN, ReLU
        self.conv2 = nn.Conv2d(w_out_b, w_out_b, kernel_size=3, stride=stride, padding=1, groups=num_group, bias=False)
        self.bn2   = nn.BatchNorm2d(w_out_b, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(inplace=True)

        # SE block
        self.with_se = se_ratio > 0
        if self.with_se:
            w_se = int(round(w_in * se_ratio))
            self.se = SE(w_out_b, w_se)

        # 1*1, BN
        self.conv3 = nn.Conv2d(w_out_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3   = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)


        # skip connection
        if stride !=1 or w_in != w_out:
            # self.skiptrans = nn.Sequential(
            #     nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False),
            #     nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
            # )
            self.skiptrans = nn.Sequential()
            self.skiptrans.add_module(
                'conv_skip', nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False)
            )
            self.skiptrans.add_module(
                'bn_skip', nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
            )
        else:
            self.skiptrans = None

        # Relu
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        if self.with_se:
            out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.skiptrans is not None:
            residual = self.skiptrans(x)

        out += residual
        out = self.relu(out)

        return out


class AnyStage(nn.Module):
    """Stage i: AnyNet stage (containing a group of blocks)"""

    def __init__(self, w_in, w_out, d, stride=1, bm=1.0, gw=1, se_ratio=0):
        super(AnyStage, self).__init__()

        for i in range(d):
            stride_bi = stride if i==0 else 1       # only the first block of every stage processing downsample
            w_in_bi   = w_in   if i==0 else w_out   # only the first block of every stage have different number of C

            self.add_module(
                "block{}".format(i+1), ResBottleneckBlock(w_in_bi, w_out, stride_bi, bm, gw, se_ratio)
            )
    def forward(self, x):
        for block in self.children():
            x = block(x)

        return x


class AnyHead(nn.Module):
    """AnyNet Head part"""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc       = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)   # [C, 1, 1] -> [C, 1]
        x = self.fc(x)              # [C, 1]    -> [Num_classes, 1]

        return x




class AnyNet(nn.Module):
    """AnyNet model"""

    def __init__(self, cfg):
        super(AnyNet, self).__init__()
        # configs:
        stem_w  = cfg['ANYNET']['STEM_W']
        depths  = cfg['ANYNET']['DEPTHS']       # an array
        widths  = cfg['ANYNET']['WIDTHS']       # an array
        strides = cfg['ANYNET']['STRIDES']      # an array
        bottles = cfg['ANYNET']['BOT_MULS']     # an array
        groups  = cfg['ANYNET']['GROUP_WS']     # an array
        se_ratio= cfg['ANYNET']['SE_R']
        num_cls = cfg['MODEL']['NUM_CLASSES']

        stage_params = list(zip(depths, widths, strides, bottles, groups))

        # start AnyNet ########################
        # stem
        self.stem = AnyStem(w_in=3, w_out=stem_w)

        # stage
        body_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            self.add_module(
                "stage{}".format(i+1), AnyStage(w_in=body_w, w_out=w, d=d, stride=s, bm=bm, gw=gw, se_ratio=se_ratio)
            )
            body_w = w

        # head
        self.head = AnyHead(w_in=body_w, nc=num_cls)

    def forward(self, x):
        for module in self.children():
            x = module(x)

        return x




if __name__ == '__main__':
    from utils.parse_cfg import load_cfg

    cfg = load_cfg('../data/AnyNet_cpu.yaml')
    model = AnyNet(cfg)


    print(model)

