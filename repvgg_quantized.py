import torch
import torch.nn as nn
import math
from collections import OrderedDict
from torch.quantization import QuantStub, DeQuantStub

class RepVGGQuant(nn.Module):

    #   {0:
    def __init__(self,
                 repvgg_model,
                 stage_sections,
                 quant_stagesections):
        super(RepVGGQuant, self).__init__()

        self.body = nn.Sequential()

        assert 0 not in stage_sections
        self.body.add_module('stage0', repvgg_model.stage0)

        for stage_idx in [1, 2, 3, 4]:
            origin_stage = repvgg_model.__getattr__('stage{}'.format(stage_idx))
            if stage_idx in stage_sections:
                sections = stage_sections[stage_idx]
                origin_blocks = list(origin_stage.children())
                blocks_per_sections = math.ceil(len(origin_blocks) / sections)
                for section_idx in range(sections):
                    cur_section_blocks = origin_blocks[section_idx * blocks_per_sections : min(len(origin_blocks), (section_idx + 1) * blocks_per_sections)]
                    od = OrderedDict()      #   We don't use a list to construct nn.Sequential because we don't want the existence of QuantStub and DeQuantStub to change the param names
                    do_quant = (stage_idx, section_idx) in quant_stagesections
                    if do_quant:    #   Quant this section. Insert the quant and dequant stubs
                        od['quant'] = QuantStub()
                    for i, b in enumerate(cur_section_blocks):
                        od[str(i)] = b
                    if do_quant:
                        od['dequant'] = DeQuantStub()
                    cur_section = nn.Sequential(od)
                    self.body.add_module('stage{}_{}'.format(stage_idx, section_idx), cur_section)
            else:
                if stage_idx in quant_stagesections:
                    od = OrderedDict()
                    od['quant'] = QuantStub()
                    for i, b in enumerate(origin_stage.children()):
                        od[str(i)] = b
                    od['dequant'] = DeQuantStub()
                    self.body.add_module('stage{}'.format(stage_idx), nn.Sequential(od))
                else:
                    self.body.add_module('stage{}'.format(stage_idx), origin_stage)

        self.quant_stagesections = quant_stagesections
        print('quant setting: ', self.quant_stagesections)

        self.gap = repvgg_model.gap
        self.linear = repvgg_model.linear

    def forward(self, x):
        out = self.body(x)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    #   From https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential and hasattr(m, 'conv'):
                torch.quantization.fuse_modules(m, ['conv', 'bn', 'relu'], inplace=True)    #TODO note this
                # torch.quantization.fuse_modules(m, ['conv', 'bn'], inplace=True)

    def _get_qconfig(self):
        return torch.quantization.get_default_qat_qconfig('fbgemm')

    def prepare_quant(self):
        #   From https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
        self.fuse_model()
        qconfig = self._get_qconfig()
        for q in self.quant_stagesections:
            if type(q) is int:
                quant_stage_or_section = self.body.__getattr__('stage{}'.format(q))
                print('prepared quant for stage', q)
            else:
                quant_stage_or_section = self.body.__getattr__('stage{}_{}'.format(q[0], q[1]))
                print('prepared quant for stage', q[0], 'section', q[1])
            quant_stage_or_section.qconfig = qconfig
            torch.quantization.prepare_qat(quant_stage_or_section, inplace=True)

    def quant_a_new_part(self, qs):
        if type(qs) is int:
            name = 'stage{}'.format(qs)
        else:
            name = 'stage{}_{}'.format(qs[0], qs[1])
        quant_stage_or_section = self.body.__getattr__(name)
        od = OrderedDict()
        od['quant'] = QuantStub()
        for i, b in enumerate(quant_stage_or_section.children()):
            od[str(i)] = b
        od['dequant'] = DeQuantStub()
        se = nn.Sequential(od)
        se.qconfig = self._get_qconfig()
        torch.quantization.prepare_qat(se, inplace=True)
        self.body.__setattr__(name, se)

