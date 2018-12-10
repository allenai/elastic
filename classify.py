import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
import os
import datetime
from utils import add_flops_counting_methods, accuracy, save_checkpoint, AverageMeter


model_names = ['resnext50', 'resnext50_elastic', 'resnext101', 'resnext101_elastic',
               'dla60x', 'dla60x_elastic', 'dla102x', 'dla102x_elastic',
               'se_resnext50', 'se_resnext50_elastic', 'densenet201', 'densenet201_elastic']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnext50_elastic', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext50_elastic)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-g', '--num-gpus', default=8, type=int,
                    metavar='N', help='number of GPUs to match (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=117, type=int,
                    metavar='N', help='print frequency (default: 117)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_err1 = 100


def main():
    global args, best_err1
    args = parser.parse_args()
    print('config: wd', args.weight_decay, 'lr', args.lr, 'batch_size', args.batch_size, 'num_gpus', args.num_gpus)
    iteration_size = args.num_gpus // torch.cuda.device_count()  # do multiple iterations
    assert iteration_size >= 1
    args.weight_decay = args.weight_decay * iteration_size  # will cancel out with lr
    args.lr = args.lr / iteration_size
    args.batch_size = args.batch_size // iteration_size
    print('real: wd', args.weight_decay, 'lr', args.lr, 'batch_size', args.batch_size, 'iteration_size', iteration_size)

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # count number of parameters
    count = 0
    params = list()
    for n, p in model.named_parameters():
        if '.ups.' not in n:
            params.append(p)
            count += np.prod(p.size())
    print('Parameters:', count)

    # count flops
    model = add_flops_counting_methods(model)
    model.eval()
    image = torch.randn(1, 3, 224, 224)

    model.start_flops_count()
    model(image).sum()
    model.stop_flops_count()
    print("GFLOPs", model.compute_average_flops_cost() / 1000000000.0)

    # normal code
    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # cuda warm up
    model = model.cuda()
    image = torch.randn(args.batch_size, 3, 224, 224)
    image_cuda = image.cuda()

    for i in range(3):
        start = time.time()
        model(image_cuda).sum().backward()  # Warmup CUDA memory allocator
        print(time.time() - start)

    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     start = time.time()
    #     model(image_cuda).sum().backward()
    #     print(time.time() - start)
    # prof.export_chrome_trace('trace_gpu')

    # import cProfile, pstats, io
    # pr = cProfile.Profile(time.perf_counter)
    # pr.enable()
    # model(image_cuda).sum().backward()
    # pr.disable()
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD([{'params': iter(params), 'lr': args.lr},
                                 ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            model.load_state_dict(checkpoint['state_dict'], strict=False) if 'state_dict' in checkpoint else print('no state_dict found')
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else print('no optimizer found')
            args.start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else args.start_epoch
            best_err1 = checkpoint['best_err1'] if 'best_err' in checkpoint else best_err1

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch'] if 'epoch' in checkpoint else 'unknown'))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, iteration_size)

        # evaluate on validation set
        err1 = validate(val_loader, model, criterion)

        # remember best err@1 and save checkpoint
        is_best = err1 < best_err1
        best_err1 = min(err1, best_err1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=args.arch + '_checkpoint.pth.tar')
        print(str(float(best_err1)))


def train(train_loader, model, criterion, optimizer, epoch, iteration_size):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    optimizer.zero_grad()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(float(loss), input.size(0))
        top1.update(100 - float(prec1), input.size(0))
        top5.update(100 - float(prec5), input.size(0))
        # compute gradient and do SGD step
        loss.backward()

        if i % iteration_size == iteration_size - 1:
            optimizer.step()
            optimizer.zero_grad()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Err@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(float(loss), input.size(0))
        top1.update(100 - float(prec1), input.size(0))
        top5.update(100 - float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Err@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(str(datetime.datetime.now()) + ' * Err@1 {top1.avg:.3f} Err@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return top1.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
