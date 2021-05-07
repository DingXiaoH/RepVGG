import torch
import torch.nn as nn
from repvgg import get_RepVGG_func_by_name
from utils import load_checkpoint

# class RepVGGRestoredBlock(nn.Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
#         super().__init__()
#         sq = nn.Sequential()
#         sq.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                                   kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
#         sq.add_module('bn', nn.BatchNorm2d(out_channels))
#         sq.add_module('relu', nn.ReLU())
#         self.restore = sq
#
#     def forward(self, input):
#         return self.restore(input)


class RepVGGQuant(nn.Module):

    def __init__(self,
                 repvgg_model,
                 stage3_splits,
                 width_multiplier=None,
                 quant_stages=None):
        super(RepVGGQuant, self).__init__()
        # repvgg = get_RepVGG_func_by_name(repvgg_name)(deploy=is_base_deploy)
        # load_checkpoint(repvgg, base_weights)

        self.stage0, self.stage1, self.stage2 = repvgg_model.stage0, repvgg_model.stage1, repvgg_model.stage2
        if use_aux and split_stage3:
            stage3_blocks = list(repvgg_model.stage3.children())
            num_blocks = len(stage3_blocks)
            assert num_blocks % 2 == 0
            self.stage3_first = nn.Sequential(*stage3_blocks[:num_blocks // 2])
            self.stage3_second = nn.Sequential(*stage3_blocks[num_blocks // 2:])
        else:
            self.stage3 = repvgg.stage3

        self.stage4, self.gap, self.linear = repvgg.stage4, repvgg.gap, repvgg.linear



        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RestoredBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2, do_quant=True)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2, do_quant=True)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)


    # def _make_stage(self, planes, num_blocks, stride, do_quant=False):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     blocks = []
    #     if do_quant:
    #         blocks.append(QuantStub())
    #     for stride in strides:
    #         cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
    #         blocks.append(RestoredBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
    #                                   stride=stride, padding=1, groups=cur_groups))
    #         self.in_planes = planes
    #         self.cur_layer_idx += 1
    #     if do_quant:
    #         blocks.append(DeQuantStub())
    #     return nn.Sequential(*blocks)

    def _make_stage(self, planes, num_blocks, stride, do_quant=False):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = OrderedDict()
        if do_quant:
            blocks['quant'] = QuantStub()
        for i, stride in enumerate(strides):
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks[str(i)] = RestoredBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups)
            self.in_planes = planes
            self.cur_layer_idx += 1
        if do_quant:
            blocks['deq'] = DeQuantStub()
        return nn.Sequential(blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        # print(out[0,:1,:1,:])
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        # out = self.linear_quant(out)
        out = self.linear(out)
        return out

    def fuse_model(self):
        # pass
        for m in self.modules():
            if type(m) == nn.Sequential and hasattr(m, 'conv'):
                torch.quantization.fuse_modules(m, ['conv', 'bn', 'relu'], inplace=True)