import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class RepVGGWholeQuant(nn.Module):

    #   {0:
    def __init__(self, repvgg_model):
        super(RepVGGWholeQuant, self).__init__()
        self.quant = QuantStub()
        self.stage0, self.stage1, self.stage2, self.stage3, self.stage4 = repvgg_model.stage0, repvgg_model.stage1, repvgg_model.stage2, repvgg_model.stage3, repvgg_model.stage4
        self.gap, self.linear = repvgg_model.gap, repvgg_model.linear
        self.dequant = DeQuantStub()


    def forward(self, x):
        x = self.quant(x)
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.dequant(out)
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
        self.qconfig = qconfig
        torch.quantization.prepare_qat(self, inplace=True)

    def freeze_quant_bn(self):
        self.apply(torch.nn.intrinsic.qat.freeze_bn_stats)