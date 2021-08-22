import torch
import math
import torchvision.datasets as datasets
import os
import torchvision.transforms as transforms
import PIL

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    ckpt = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            ckpt[k[7:]] = v
        else:
            ckpt[k] = v
    model.load_state_dict(ckpt)

def read_hdf5(file_path):
    import h5py
    import numpy as np
    result = {}
    with h5py.File(file_path, 'r') as f:
        for k in f.keys():
            value = np.asarray(f[k])
            result[str(k).replace('+', '/')] = value
    print('read {} arrays from {}'.format(len(result), file_path))
    f.close()
    return result

def model_load_hdf5(model:torch.nn.Module, hdf5_path, ignore_keys='stage0.'):
    weights_dict = read_hdf5(hdf5_path)
    for name, param in model.named_parameters():
        print('load param: ', name, param.size())
        if name in weights_dict:
            np_value = weights_dict[name]
        else:
            np_value = weights_dict[name.replace(ignore_keys, '')]
        value = torch.from_numpy(np_value).float()
        assert tuple(value.size()) == tuple(param.size())
        param.data = value
    for name, param in model.named_buffers():
        print('load buffer: ', name, param.size())
        if name in weights_dict:
            np_value = weights_dict[name]
        else:
            np_value = weights_dict[name.replace(ignore_keys, '')]
        value = torch.from_numpy(np_value).float()
        assert tuple(value.size()) == tuple(param.size())
        param.data = value



class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, T_cosine_max, eta_min=0, last_epoch=-1, warmup=0):
        self.eta_min = eta_min
        self.T_cosine_max = T_cosine_max
        self.warmup = warmup
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            return [self.last_epoch / self.warmup * base_lr for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup) / (self.T_cosine_max - self.warmup))) / 2
                    for base_lr in self.base_lrs]


def log_msg(message, log_file):
    print(message)
    with open(log_file, 'a') as f:
        print(message, file=f)


def get_ImageNet_train_dataset(args, trans):
    if os.path.exists('/home/dingxiaohan/ndp/imagenet.train.nori.list'):
        #   This is the data source on our machine. You won't need it.
        from noris_dataset import ImageNetNoriDataset
        train_dataset = ImageNetNoriDataset('/home/dingxiaohan/ndp/imagenet.train.nori.list', trans)
    else:
        #   Your ImageNet directory
        traindir = os.path.join(args.data, 'train')
        train_dataset = datasets.ImageFolder(traindir, trans)
    return train_dataset


def get_ImageNet_val_dataset(args, trans):
    if os.path.exists('/home/dingxiaohan/ndp/imagenet.val.nori.list'):
        #   This is the data source on our machine. You won't need it.
        from noris_dataset import ImageNetNoriDataset
        val_dataset = ImageNetNoriDataset('/home/dingxiaohan/ndp/imagenet.val.nori.list', trans)
    else:
        #   Your ImageNet directory
        traindir = os.path.join(args.data, 'val')
        val_dataset = datasets.ImageFolder(traindir, trans)
    return val_dataset


def get_default_train_trans(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if (not hasattr(args, 'resolution')) or args.resolution == 224:
        trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        raise ValueError('Not yet implemented.')
    return trans


def get_default_val_trans(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if (not hasattr(args, 'resolution')) or args.resolution == 224:
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
            normalize,
        ])
    return trans


def get_default_ImageNet_train_sampler_loader(args):
    train_trans = get_default_train_trans(args)
    train_dataset = get_ImageNet_train_dataset(args, train_trans)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    return train_sampler, train_loader


def get_default_ImageNet_val_loader(args):
    val_trans = get_default_val_trans(args)
    val_dataset = get_ImageNet_val_dataset(args, val_trans)
    if hasattr(args, 'val_batch_size'):
        bs = args.val_batch_size
    else:
        bs = args.batch_size
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=bs, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return val_loader