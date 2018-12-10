import argparse
import os
import shutil
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import models
import numpy as np
import gc

from utils import add_flops_counting_methods, save_checkpoint, AverageMeter, accuracy

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

model_names = ['resnext50', 'resnext50_elastic', 'resnext101', 'resnext101_elastic',
               'dla60x', 'dla60x_elastic', 'dla102x', 'dla102x_elastic',
               'se_resnext50', 'se_resnext50_elastic', 'densenet201', 'densenet201_elastic']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnext50_elastic', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext50_elastic)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-g', '--num-gpus', default=8, type=int,
                    metavar='N', help='number of GPUs we pretend to have (default: 8)')
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
parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dist-url', default='file://sync.file', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='Number of GPUs to use. Can either be manually set ' +
                    'or automatically set by using \'python -m multiproc\'.')
parser.add_argument('--rank', default=0, type=int,
                    help='Used for multi-process training. Can either be manually set ' +
                    'or automatically set by using \'python -m multiproc\'.')

cudnn.benchmark = True


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)
        
    return tensor, targets


best_err1 = 100
args = parser.parse_args()


def main():
    global best_err1, args

    iteration_size = args.num_gpus // args.world_size
    args.weight_decay = args.weight_decay * iteration_size  # will cancel out with lr
    args.lr = args.lr / iteration_size
    print('real: wd', args.weight_decay, 'lr', args.lr, 'batch_size', args.batch_size, 'iteration_size', iteration_size)
    args.distributed = args.world_size > 1
    args.gpu = 0
    if args.distributed:
        args.gpu = args.rank % torch.cuda.device_count()

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend=args.dist_backend, 
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

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

    model = model.cuda()
    if args.fp16:
        model = network_to_half(model)
    if args.distributed:
        #shared param turns off bucketing in DDP, for lower latency runs this can improve perf
        model = DDP(model, shared_param=True)

    global model_params, master_params
    if args.fp16:
        model_params, master_params = prep_param_lists(model)
    else:
        master_params = list(model.parameters())

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

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    crop_size = 224
    val_size = 256

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(), Too slow
            # normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop_size),
        ]))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop_size),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=fast_collate)
    # print(len(train_loader), len(val_loader))
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)
        print('allocated before', torch.cuda.memory_allocated())
        print('cached before', torch.cuda.memory_cached())
        gc.collect()
        torch.cuda.empty_cache()
        print('allocated after', torch.cuda.memory_allocated())
        print('cached after', torch.cuda.memory_cached())
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, iteration_size)

#        # sync models on multiple GPUs
#        if args.rank == 0:
#            save_checkpoint({
#                'epoch': epoch + 1,
#                'arch': args.arch,
#                'state_dict': model.state_dict(),
#                'optimizer' : optimizer.state_dict(),
#            }, False, 'temp.pth.tar')
#        # barrier
#        loss = torch.FloatTensor([args.rank]).cuda()
#        reduced_loss = reduce_tensor(loss.data)
#        print(loss.data, reduced_loss)
#        if os.path.isfile('temp.pth.tar'):
#            print("=> loading checkpoint '{}'".format('temp.pth.tar'))
#            checkpoint = torch.load('temp.pth.tar', map_location = lambda storage, loc: storage.cuda(args.gpu))
#            model.load_state_dict(checkpoint['state_dict'], strict=False)
#            optimizer.load_state_dict(checkpoint['optimizer'])
#            print("=> loaded checkpoint '{}' (epoch {})"
#                  .format('temp.pth.tar', checkpoint['epoch']))
#            assert checkpoint['epoch'] == epoch + 1

        # evaluate on validation set
        err1 = validate(val_loader, model, criterion)
        # remember best err@1 and save checkpoint
        if args.rank == 0:
            is_best = err1 < best_err1
            best_err1 = min(err1, best_err1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_err1': best_err1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
            print(str(float(best_err1)))


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        if args.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(async=True)
            self.next_target = self.next_target.cuda(async=True)
            if args.fp16:
                self.next_input = self.next_input.half()
            else:
                self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


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

    prefetcher = data_prefetcher(train_loader)
    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1
        
        # measure data loading time
        data_time.update(time.time() - end)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(100 - to_python_float(prec1), input.size(0))
        top5.update(100 - to_python_float(prec5), input.size(0))

        loss = loss*args.static_loss_scale
        # compute gradient and do SGD step
        loss.backward()
        if i % iteration_size == iteration_size - 1:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()
        input, target = prefetcher.next()
        if args.rank == 0 and i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Err@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader),
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1

        target = target.cuda(async=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(100 - to_python_float(prec1), input.size(0))
        top5.update(100 - to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Err@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader),
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
        input, target = prefetcher.next()
    print(' * Err@1 {top1.avg:.3f} Err@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return top1.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()
