import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from utils import AverageMeter, inter_and_union, VOCSegmentation
import models

model_names = ['resnext50', 'resnext50_elastic', 'resnext101', 'resnext101_elastic', 'dla60x', 'dla60x_elastic']

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0,
                    help='test time gpu device id')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnext50_elastic', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext50_elastic)')
parser.add_argument('--dataset', type=str, default='pascal',
                    help='pascal')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.007,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--crop_size', type=int, default=513,
                    help='image crop size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
args = parser.parse_args()


def main():
    assert torch.cuda.is_available()
    model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(
        args.arch, args.dataset, args.exp)
    if args.dataset == 'pascal':
        dataset = VOCSegmentation('data/VOCdevkit',
                                  train=args.train, crop_size=args.crop_size)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if 'resnext' in args.arch:
        model = models.__dict__[args.arch](seg=True, num_classes=len(dataset.CLASSES))
    elif 'dla' in args.arch:
        model = models.__dict__[args.arch + '_seg'](classes=len(dataset.CLASSES))
    else:
        raise ValueError('Unknown arch: {}'.format(args.arch))

    if args.train:
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        model = nn.DataParallel(model).cuda()
        model.train()
        if args.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        if 'resnext' in args.arch:
            arch_params = (
                    list(model.module.conv1.parameters()) +
                    list(model.module.bn1.parameters()) +
                    list(model.module.layer1.parameters()) +
                    list(model.module.layer2.parameters()) +
                    list(model.module.layer3.parameters()) +
                    list(model.module.layer4.parameters()))
            last_params = list(model.module.aspp.parameters())
        else:
            arch_params = list(model.module.base.parameters())
            last_params = list()
            for n, p in model.named_parameters():
                if 'base' not in n and 'up.weight' not in n:
                    last_params.append(p)

        optimizer = optim.SGD([
            {'params': filter(lambda p: p.requires_grad, arch_params)},
            {'params': filter(lambda p: p.requires_grad, last_params)}],
            lr=args.base_lr, momentum=0.9, weight_decay=0.0005 if 'resnext' in args.arch else 0.0001)
        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=args.train,
            pin_memory=True, num_workers=args.workers)
        max_iter = args.epochs * len(dataset_loader)
        losses = AverageMeter()
        start_epoch = 0

        if args.resume:
            if os.path.isfile(args.resume):
                print('=> loading checkpoint {0}'.format(args.resume))
                checkpoint = torch.load(args.resume)

                resume = False
                for n, p in list(checkpoint['state_dict'].items()):
                    if 'aspp' in n or 'dla_up' in n:
                        resume = True
                        break
                if resume:
                    # True resume: resume training on pascal
                    model.load_state_dict(checkpoint['state_dict'], strict=True)
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    start_epoch = checkpoint['epoch']
                else:
                    # Fake resume: transfer from ImageNet
                    if 'resnext' in args.arch:
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        pretrained_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
                        model.module.base.load_state_dict(pretrained_dict, strict=False)
                print('=> loaded checkpoint {0} (epoch {1})'.format(
                    args.resume, start_epoch))
            else:
                print('=> no checkpoint found at {0}'.format(args.resume))

        for epoch in range(start_epoch, args.epochs):
            for i, (inputs, target, _, _, _, _) in enumerate(dataset_loader):
                cur_iter = epoch * len(dataset_loader) + i
                lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
                optimizer.param_groups[0]['lr'] = lr
                optimizer.param_groups[1]['lr'] = lr * args.last_mult

                inputs = Variable(inputs.cuda())
                target = Variable(target.cuda())

                outputs = model(inputs)
                loss = criterion(outputs, target)
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    pdb.set_trace()
                losses.update(loss.item(), args.batch_size)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print('epoch: {0}\t'
                      'iter: {1}/{2}\t'
                      'lr: {3:.6f}\t'
                      'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                    epoch + 1, i + 1, len(dataset_loader), lr, loss=losses))

            if epoch % 10 == 9:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_fname % (epoch + 1))

    else:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
        model.eval()
        checkpoint = torch.load(model_fname % args.epochs)
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        model.load_state_dict(state_dict)
        cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
        cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

        inter_meter = AverageMeter()
        union_meter = AverageMeter()
        for i in range(len(dataset)):
            inputs, target, a, b, h, w = dataset[i]
            inputs = inputs.unsqueeze(0)
            inputs = Variable(inputs.cuda())
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
            mask = target.numpy().astype(np.uint8)
            imname = dataset.masks[i].split('/')[-1]

            inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
            inter_meter.update(inter)
            union_meter.update(union)

            mask_pred = Image.fromarray(pred[a:a + h, b:b + w])
            mask_pred.putpalette(cmap)
            mask_pred.save(os.path.join('data/val', imname))
            print('eval: {0}/{1}'.format(i + 1, len(dataset)))

        iou = inter_meter.sum / (union_meter.sum + 1e-10)
        for i, val in enumerate(iou):
            print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
        print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))


if __name__ == "__main__":
    main()
