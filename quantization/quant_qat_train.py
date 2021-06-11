import argparse
import random
import shutil
import time
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from utils import *

best_acc1 = 0

IMAGENET_TRAINSET_SIZE = 1281167

parser = argparse.ArgumentParser(description='PyTorch Whole Model Quant')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=8, type=int, metavar='N',
                    help='number of epochs for each run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--val-batch-size', default=100, type=int, metavar='V',
                    help='validation batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='learning rate for finetuning', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23333', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--base-weights', default=None, type=str,
                    help='weights of the base model.')
parser.add_argument('--tag', default='testtest', type=str,
                    help='the tag for identifying the log and model files. Just a string.')
parser.add_argument('--fpfinetune', dest='fpfinetune', action='store_true',
                    help='full precision finetune')
parser.add_argument('--fixobserver', dest='fixobserver', action='store_true',
                    help='fix observer?')
parser.add_argument('--fixbn', dest='fixbn', action='store_true',
                    help='fix bn?')
parser.add_argument('--quantlayers', default='all', type=str, choices=['all', 'exclud_first_and_linear', 'exclud_first_and_last'],
                    help='the tag for identifying the log and model files. Just a string.')



def sgd_optimizer(model, lr, momentum, weight_decay):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr
        if value.ndimension() < 2:  #TODO note this
            apply_weight_decay = 0
            print('set weight decay=0 for {}'.format(key))
        if 'bias' in key:
            apply_lr = 2 * lr       #   Just a Caffe-style common practice. Made no difference.
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    log_file = 'quant_{}_exp.txt'.format(args.tag)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    #   1.  Build and load base model
    from repvgg import get_RepVGG_func_by_name
    repvgg_build_func = get_RepVGG_func_by_name(args.arch)
    base_model = repvgg_build_func(deploy=True)
    from insert_bn import directly_insert_bn_without_init
    directly_insert_bn_without_init(base_model)
    if args.base_weights is not None:
        load_checkpoint(base_model, args.base_weights)

    #   2.
    if not args.fpfinetune:
        from quantization.repvgg_quantized import RepVGGWholeQuant
        qat_model = RepVGGWholeQuant(repvgg_model=base_model, quantlayers=args.quantlayers)
        qat_model.prepare_quant()
    else:
        qat_model = base_model
        log_msg('===================== not QAT, just full-precision finetune ===========', log_file)

    #===================================================
    #   From now on, the code will be very similar to ordinary training
    # ===================================================

    is_main = not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

    if is_main:
        for n, p in qat_model.named_parameters():
            print(n, p.size())
        for n, p in qat_model.named_buffers():
            print(n, p.size())
        log_msg('epochs {}, lr {}, weight_decay {}'.format(args.epochs, args.lr, args.weight_decay), log_file)
        #   You will see it now has quantization-related parameters (zero-points and scales)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            qat_model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            qat_model = torch.nn.parallel.DistributedDataParallel(qat_model, device_ids=[args.gpu])
        else:
            qat_model.cuda()
            qat_model = torch.nn.parallel.DistributedDataParallel(qat_model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        qat_model = qat_model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        qat_model = torch.nn.DataParallel(qat_model).cuda()


    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = sgd_optimizer(qat_model, args.lr, args.momentum, args.weight_decay)

    warmup_epochs = 1
    lr_scheduler = WarmupCosineAnnealingLR(optimizer=optimizer, T_cosine_max=args.epochs * IMAGENET_TRAINSET_SIZE // args.batch_size // ngpus_per_node,
                            eta_min=0, warmup=warmup_epochs * IMAGENET_TRAINSET_SIZE // args.batch_size // ngpus_per_node)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            qat_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_sampler, train_loader = get_default_ImageNet_train_sampler_loader(args)
    val_loader = get_default_ImageNet_val_loader(args)

    if args.evaluate:
        validate(val_loader, qat_model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, qat_model, criterion, optimizer, epoch, args, lr_scheduler, is_main=is_main)

        if args.fixobserver and epoch > (3 * args.epochs // 8):
            # Freeze quantizer parameters
            qat_model.apply(torch.quantization.disable_observer)  #TODO testing. May not be useful
            log_msg('fix observer after epoch {}'.format(epoch), log_file)

        if args.fixbn and epoch > (2 * args.epochs // 8):    #TODO testing. May not be useful
        #     Freeze batch norm mean and variance estimates
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            log_msg('fix bn after epoch {}'.format(epoch), log_file)

        # evaluate on validation set
        if is_main:
            acc1 = validate(val_loader, qat_model, criterion, args)
            msg = '{}, base{}, quant, epoch {}, QAT acc {}'.format(args.arch, args.base_weights, epoch, acc1)
            log_msg(msg, log_file)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': qat_model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
            }, is_best,
                filename = '{}_{}.pth.tar'.format(args.arch, args.tag),
                best_filename='{}_{}_best.pth.tar'.format(args.arch, args.tag))


def train(train_loader, model, criterion, optimizer, epoch, args, lr_scheduler, is_main):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, ],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output

        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if is_main and i % args.print_freq == 0:
            progress.display(i)
        if is_main and i % 1000 == 0 and lr_scheduler is not None:
            print('cur lr: ', lr_scheduler.get_lr()[0])




def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)





if __name__ == '__main__':
    main()