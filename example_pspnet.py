import torch
from torch import nn
import torch.nn.functional as F
from repvgg import get_RepVGG_func_by_name

#   The PSPNet parts are from
#   https://github.com/hszhao/semseg

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self,
                backbone_name, backbone_file, deploy,
                 bins=(1, 2, 3, 6), dropout=0.1, classes=2,
                 zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d,
                 pretrained=True):
        super(PSPNet, self).__init__()
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4

        #   The last two stages should have stride=1 for semantic segmentation
        #   Note that the stride of 1x1 should be the same as the 3x3
        #   Use dilation following the implementation of PSPNet
        secondlast_channel = 0
        for n, m in self.layer3.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                print('change dilation, padding, stride of ', n)
                secondlast_channel = m.out_channels
            elif 'rbr_1x1' in n and isinstance(m, nn.Conv2d):
                m.stride = (1, 1)
                print('change stride of ', n)
        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                print('change dilation, padding, stride of ', n)
                last_channel = m.out_channels
            elif 'rbr_1x1' in n and isinstance(m, nn.Conv2d):
                m.stride = (1, 1)
                print('change stride of ', n)

        fea_dim = last_channel
        aux_in = secondlast_channel

        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
            fea_dim *= 2

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(aux_in, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)

        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x


if __name__ == '__main__':
    #   1.  Build the PSPNet with RepVGG backbone. Download the ImageNet-pretrained weight file and load it.
    model = PSPNet(backbone_name='RepVGG-A0', backbone_file='RepVGG-A0-train.pth', deploy=False, classes=19, pretrained=True)

    #   2.  Train it
    #   seg_train(model)

    #   3.  Convert and check the equivalence
    input = torch.rand(4, 3, 713, 713)
    model.eval()
    print(model)
    y_train = model(input)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    y_deploy = model(input)
    print('output is ', y_deploy.size())
    print('=================== The diff is')
    print(((y_deploy - y_train) ** 2).sum())

    #   4.  Save the converted model
    torch.save(model.state_dict(), 'PSPNet-RepVGG-A0-deploy.pth')
    del model   #   Or do whatever you want with it

    #   5.  For inference, load the saved model. There is no need to load the ImageNet-pretrained weights again.
    deploy_model = PSPNet(backbone_name='RepVGG-A0', backbone_file=None, deploy=True, classes=19, pretrained=False)
    deploy_model.eval()
    deploy_model.load_state_dict(torch.load('PSPNet-RepVGG-A0-deploy.pth'))

    #   6.  Check again or do whatever you want
    y_deploy = deploy_model(input)
    print('=================== The diff is')
    print(((y_deploy - y_train) ** 2).sum())