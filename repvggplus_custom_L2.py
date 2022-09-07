# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from se_block import SEBlock
import torch
import numpy as np


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    result.add_module('relu', nn.ReLU())
    return result

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGplusBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros',
                 deploy=False,
                 use_post_se=False):
        super(RepVGGplusBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        self.nonlinearity = nn.ReLU()

        if use_post_se:
            self.post_se = SEBlock(out_channels, internal_neurons=out_channels // 4)
        else:
            self.post_se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            padding_11 = padding - kernel_size // 2
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, x, L2):

        if self.deploy:
            return self.post_se(self.nonlinearity(self.rbr_reparam(x))), None

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        out = self.rbr_dense(x) + self.rbr_1x1(x) + id_out
        out = self.post_se(self.nonlinearity(out))

        #   Custom L2
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:,:,1:2,1:2] * t3 + K1 * t1
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()

        return out, L2 + l2_loss_circle + l2_loss_eq_kernel


    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            #   For the 1x1 or 3x3 branch
            kernel, running_mean, running_var, gamma, beta, eps = branch.conv.weight, branch.bn.running_mean, branch.bn.running_var, branch.bn.weight, branch.bn.bias, branch.bn.eps
        else:
            #   For the identity branch
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                #   Construct and store the identity kernel in case it is used multiple times
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel, running_mean, running_var, gamma, beta, eps = self.id_tensor, branch.running_mean, branch.running_var, branch.weight, branch.bias, branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True



class RepVGGplusStage(nn.Module):

    def __init__(self, in_planes, planes, num_blocks, stride, use_checkpoint, use_post_se=False, deploy=False):
        super().__init__()
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        self.in_planes = in_planes
        for stride in strides:
            cur_groups = 1
            blocks.append(RepVGGplusBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=deploy, use_post_se=use_post_se))
            self.in_planes = planes
        self.blocks = nn.ModuleList(blocks)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, L2):
        for block in self.blocks:
            if self.use_checkpoint:
                x, L2 = checkpoint.checkpoint(block, x, L2)
            else:
                x, L2 = block(x, L2)
        return x, L2


class RepVGGplus(nn.Module):

    def __init__(self, num_blocks, num_classes,
                 width_multiplier, override_groups_map=None,
                 deploy=False,
                 use_post_se=False,
                 use_checkpoint=False):
        super().__init__()

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_post_se = use_post_se
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        self.nonlinear = 'relu'

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGplusBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_post_se=use_post_se)
        self.cur_layer_idx = 1
        self.stage1 = RepVGGplusStage(self.in_planes, int(64 * width_multiplier[0]), num_blocks[0], stride=2, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        self.stage2 = RepVGGplusStage(int(64 * width_multiplier[0]), int(128 * width_multiplier[1]), num_blocks[1], stride=2, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        #   split stage3 so that we can insert an auxiliary classifier
        self.stage3_first = RepVGGplusStage(int(128 * width_multiplier[1]), int(256 * width_multiplier[2]), num_blocks[2] // 2, stride=2, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        self.stage3_second = RepVGGplusStage(int(256 * width_multiplier[2]), int(256 * width_multiplier[2]), num_blocks[2] // 2, stride=1, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        self.stage4 = RepVGGplusStage(int(256 * width_multiplier[2]), int(512 * width_multiplier[3]), num_blocks[3], stride=2, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)
        #   aux classifiers
        if not self.deploy:
            self.stage1_aux = self._build_aux_for_stage(self.stage1)
            self.stage2_aux = self._build_aux_for_stage(self.stage2)
            self.stage3_first_aux = self._build_aux_for_stage(self.stage3_first)

    def _build_aux_for_stage(self, stage):
        stage_out_channels = list(stage.blocks.children())[-1].rbr_dense.conv.out_channels
        downsample = conv_bn_relu(in_channels=stage_out_channels, out_channels=stage_out_channels, kernel_size=3, stride=2, padding=1)
        fc = nn.Linear(stage_out_channels, self.num_classes, bias=True)
        return nn.Sequential(downsample, nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc)

    def forward(self, x):
        if self.deploy:
            out, _ = self.stage0(x, L2=None)
            out, _ = self.stage1(out, L2=None)
            out, _ = self.stage2(out, L2=None)
            out, _ = self.stage3_first(out, L2=None)
            out, _ = self.stage3_second(out, L2=None)
            out, _ = self.stage4(out, L2=None)
            y = self.gap(out)
            y = y.view(y.size(0), -1)
            y = self.linear(y)
            return y

        else:
            out, L2 = self.stage0(x, L2=0.0)
            out, L2 = self.stage1(out, L2=L2)
            stage1_aux = self.stage1_aux(out)
            out, L2 = self.stage2(out, L2=L2)
            stage2_aux = self.stage2_aux(out)
            out, L2 = self.stage3_first(out, L2=L2)
            stage3_first_aux = self.stage3_first_aux(out)
            out, L2 = self.stage3_second(out, L2=L2)
            out, L2 = self.stage4(out, L2=L2)
            y = self.gap(out)
            y = y.view(y.size(0), -1)
            y = self.linear(y)
            return {
                'main': y,
                'stage1_aux': stage1_aux,
                'stage2_aux': stage2_aux,
                'stage3_first_aux': stage3_first_aux,
                'L2': L2
            }

    def switch_repvggplus_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
            if hasattr(m, 'use_checkpoint'):
                m.use_checkpoint = False        #   Disable checkpoint. I am not sure whether using checkpoint slows down inference.
        if hasattr(self, 'stage1_aux'):
            self.__delattr__('stage1_aux')
        if hasattr(self, 'stage2_aux'):
            self.__delattr__('stage2_aux')
        if hasattr(self, 'stage3_first_aux'):
            self.__delattr__('stage3_first_aux')
        self.deploy = True


#   torch.utils.checkpoint can reduce the memory consumption during training with a minor slowdown. Don't use it if you have sufficient GPU memory.
#   Not sure whether it slows down inference
#   pse for "post SE", which means using SE block after ReLU
def create_RepVGGplus_L2pse(deploy=False, use_checkpoint=False):
    return RepVGGplus(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_post_se=True,
                      use_checkpoint=use_checkpoint)

repvggplus_func_dict = {
'RepVGGplus-L2pse': create_RepVGGplus_L2pse,
}
def get_RepVGGplus_func_by_name(name):
    return repvggplus_func_dict[name]