import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import accuracy, ProgressMeter, AverageMeter
from repvgg import get_RepVGG_func_by_name, RepVGGBlock
import PIL
from utils import load_checkpoint


#   Get the mean and std on every conv3x3 (before the bias-adding) on the train set. Then use such data to initialize BN layers and insert them after conv3x3.
#   May, 07, 2021

parser = argparse.ArgumentParser(description='Get the mean and std on every conv3x3 (before the bias-adding) on the train set. Then use such data to initialize BN layers and insert them after conv3x3.')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('weights', metavar='WEIGHTS', help='path to the weights file')
parser.add_argument('save', metavar='SAVE', help='path to save the model with BN')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: 100) for test')
parser.add_argument('-n', '--num-batches', default=500, type=int,
                    metavar='N',
                    help='number of batches (default: 500) to record the mean and std on the train set')
parser.add_argument('-r', '--resolution', default=224, type=int,
                    metavar='R',
                    help='resolution (default: 224) for test')


def update_running_mean_var(x, running_mean, running_var, momentum=0.9):
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    running_mean = momentum * running_mean + (1.0 - momentum) * mean
    running_var = momentum * running_var + (1.0 - momentum) * var
    return running_mean, running_var

#   Record the mean and std like a BN layer but do no normalization
class BNStatistics(nn.Module):
    def __init__(self, num_features):
        super(BNStatistics, self).__init__()
        shape = (1, num_features, 1, 1)
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))
    def forward(self, x):
        if self.running_mean.device != x.device:
            self.running_mean = self.running_mean.to(x.device)
            self.running_var = self.running_var.to(x.device)
        self.running_mean, self.running_var = update_running_mean_var(x, self.running_mean, self.running_var, momentum=0.9)
        return x

#   This is designed to insert BNStat layer between Conv2d(without bias) and its bias
class BiasAdd(nn.Module):
    def __init__(self, num_features):
        super(BiasAdd, self).__init__()
        self.bias = torch.nn.Parameter(torch.Tensor(num_features))
    def forward(self, x):
        return x + self.bias.view(1, -1, 1, 1)

def switch_repvggblock_to_bnstat(block):
    assert hasattr(block, 'rbr_reparam')
    stat = nn.Sequential()
    stat.add_module('conv', nn.Conv2d(block.rbr_reparam.in_channels, block.rbr_reparam.out_channels,
                                           block.rbr_reparam.kernel_size,
                                           block.rbr_reparam.stride, block.rbr_reparam.padding, block.rbr_reparam.dilation,
                                           block.rbr_reparam.groups, bias=False))   # Note bias=False
    stat.add_module('bnstat', BNStatistics(block.rbr_reparam.out_channels))
    stat.add_module('biasadd', BiasAdd(block.rbr_reparam.out_channels))     # Bias is here
    stat.conv.weight.data = block.rbr_reparam.weight.data
    stat.biasadd.bias.data = block.rbr_reparam.bias.data
    block.__delattr__('rbr_reparam')
    block.rbr_reparam = stat

def switch_bnstat_to_convbn(block):
    assert hasattr(block, 'rbr_reparam')
    assert hasattr(block.rbr_reparam, 'bnstat')
    conv = nn.Conv2d(block.rbr_reparam.conv.in_channels, block.rbr_reparam.conv.out_channels, block.rbr_reparam.conv.kernel_size,
                     block.rbr_reparam.conv.stride, block.rbr_reparam.conv.padding, block.rbr_reparam.conv.dilation,
                     block.rbr_reparam.conv.groups, bias=False)
    bn = nn.BatchNorm2d(block.rbr_reparam.conv.out_channels)
    bn.running_mean = block.rbr_reparam.bnstat.running_mean.squeeze()   #   Initialize the mean and var of BN with the statistics
    bn.running_var = block.rbr_reparam.bnstat.running_var.squeeze()
    std = (bn.running_var + bn.eps).sqrt()
    conv.weight.data = block.rbr_reparam.conv.weight.data
    bn.weight.data = std
    bn.bias.data = block.rbr_reparam.biasadd.bias.data + bn.running_mean    # Initialize gamma = std and beta = bias + mean

    convbn = nn.Sequential()
    convbn.add_module('conv', conv)
    convbn.add_module('bn', bn)
    block.__delattr__('rbr_reparam')
    block.rbr_reparam = convbn


def insert_bn():
    args = parser.parse_args()

    repvgg_build_func = get_RepVGG_func_by_name(args.arch)

    model = repvgg_build_func(deploy=True)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        use_gpu = False
    else:
        model = model.cuda()
        use_gpu = True

    load_checkpoint(model, args.weights)

    for n, m in model.named_modules():
        if isinstance(m, RepVGGBlock):
            switch_repvggblock_to_bnstat(m)
            print('switch to BN Statistics: ', n)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.resolution == 224:
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
    else:
        trans = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize])

    traindir = os.path.join(args.data, 'train')
    train_dataset = datasets.ImageFolder(traindir, trans)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        min(len(train_loader), args.num_batches),   #   If num_batches > the total num of batches in the train set, we just use the whole set
        [batch_time],
        prefix='BN Stats: ')

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            if i == args.num_batches:
                break
            if use_gpu:
                images = images.cuda(non_blocking=True)
            # Just forward
            model(images)
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 10 == 0:
                progress.display(i)

    for n, m in model.named_modules():
        if isinstance(m, RepVGGBlock):
            switch_bnstat_to_convbn(m)
            print('switch to ConvBN: ', n)

    torch.save(model.state_dict(), args.save)




if __name__ == '__main__':
    insert_bn()