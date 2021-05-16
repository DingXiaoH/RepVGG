import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from repvgg import get_RepVGG_func_by_name, repvgg_model_convert

parser = argparse.ArgumentParser(description='RepVGG Conversion')
parser.add_argument('load', metavar='LOAD', help='path to the weights file')
parser.add_argument('save', metavar='SAVE', help='path to the weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0')

def convert():
    args = parser.parse_args()

    repvgg_build_func = get_RepVGG_func_by_name(args.arch)

    train_model = repvgg_build_func(deploy=False)

    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        print(ckpt.keys())
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    repvgg_model_convert(train_model, save_path=args.save)


if __name__ == '__main__':
    convert()