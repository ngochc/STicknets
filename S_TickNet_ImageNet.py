# -*- coding: utf-8 -*-

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.STickNet import *
from models.common import AverageMeter
from util import get_device

model_names = ['basic', 'small', 'large']
parser = argparse.ArgumentParser(
    description='PyTorch Spatial TickNet Training')

parser.add_argument('-r', '--data', type=str,
                    default='../../../datasets/ImageNet', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='basic',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: basic)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--action', default='', type=str,
                    help='other information.')


best_prec1 = 0


def main():
  global args, best_prec1
  args = parser.parse_args()

  if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU. This will completely '
                  'disable data parallelism.')

  torch.autograd.set_detect_anomaly(True)
  # create model
  model = build_SpatialTickNet(1000, typesize=args.arch, cifar=False)

  device = get_device(args.gpu)
  model = model.to(device)
  print(model)

  # get the number of models parameters
  print('Number of models parameters: {}'.format(
      sum([p.data.nelement() for p in model.parameters()])))

  # define loss function (criterion) and optimizer
  criterion = nn.CrossEntropyLoss().cuda(args.gpu)

  optimizer = torch.optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

  # optionally resume from a checkpoint
  if args.evaluate:
    pathcheckpoint = "./checkpoints/ImageNet1k/small/model_best.pth.tar"
    if os.path.isfile(pathcheckpoint):
      print("=> loading checkpoint '{}'".format(pathcheckpoint))
      checkpoint = torch.load(pathcheckpoint)
      model.load_state_dict(checkpoint['state_dict'])
      # optimizer.load_state_dict(checkpoint['optimizer'])
      del checkpoint
    else:
      print("=> no checkpoint found at '{}'".format(pathcheckpoint))
      return
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume)
      args.start_epoch = checkpoint['epoch']
      best_prec1 = checkpoint['best_prec1']
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))
      del checkpoint
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

  train_sampler = None

  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=(
          train_sampler is None),
      num_workers=args.workers, pin_memory=True, sampler=train_sampler)

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
    m = time.time()
    _, _ = validate(val_loader, model, criterion)
    n = time.time()
    print((n-m)/3600)
    return

  directory = "runs/%s/" % (args.arch + '_' + args.action)
  if not os.path.exists(directory):
    os.makedirs(directory)

  Loss_plot = {}
  train_prec1_plot = {}
  train_prec5_plot = {}
  val_prec1_plot = {}
  val_prec5_plot = {}

  for epoch in range(args.start_epoch, args.epochs):
    start_time = time.time()

    adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    # train(train_loader, model, criterion, optimizer, epoch)
    loss_temp, train_prec1_temp, train_prec5_temp = train(
        train_loader, model, criterion, optimizer, epoch)
    Loss_plot[epoch] = loss_temp
    train_prec1_plot[epoch] = train_prec1_temp
    train_prec5_plot[epoch] = train_prec5_temp

    # evaluate on validation set
    # prec1 = validate(val_loader, model, criterion)
    prec1, prec5 = validate(val_loader, model, criterion)
    val_prec1_plot[epoch] = prec1
    val_prec5_plot[epoch] = prec5

    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best)

    # 将Loss,train_prec1,train_prec5,val_prec1,val_prec5用.txt的文件存起来
    data_save(directory + 'Loss_plot.txt', Loss_plot)
    data_save(directory + 'train_prec1.txt', train_prec1_plot)
    data_save(directory + 'train_prec5.txt', train_prec5_plot)
    data_save(directory + 'val_prec1.txt', val_prec1_plot)
    data_save(directory + 'val_prec5.txt', val_prec5_plot)

    end_time = time.time()
    time_value = (end_time - start_time) / 3600
    print("-" * 80)
    print(time_value)
    print("-" * 80)


def train(train_loader, model, criterion, optimizer, epoch):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  # switch to train mode
  model.train()

  end = time.time()
  for i, (input, target) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    if args.gpu is not None:
      input = input.cuda(args.gpu, non_blocking=True)
    target = target.cuda(args.gpu, non_blocking=True)

    # compute output
    output = model(input)
    loss = criterion(output, target)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % args.print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

  return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion):
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
      if args.gpu is not None:
        input = input.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)

      # compute output
      output = model(input)
      loss = criterion(output, target)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), input.size(0))
      top1.update(prec1[0], input.size(0))
      top5.update(prec5[0], input.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % args.print_freq == 0:
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  i, len(val_loader), batch_time=batch_time, loss=losses,
                  top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

  return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
  directory = "runs/%s/" % (args.arch + '_' + args.action)

  filename = directory + filename
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, directory + 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.lr * (0.1 ** (epoch // 30))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      correct_k = correct[:k].contiguous(
      ).view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


def data_save(root, file):
  if not os.path.exists(root):
    os.mknod(root)
  file_temp = open(root, 'r')
  lines = file_temp.readlines()
  if not lines:
    epoch = -1
  else:
    epoch = lines[-1][:lines[-1].index(' ')]
  epoch = int(epoch)
  file_temp.close()
  file_temp = open(root, 'a')
  for line in file:
    if line > epoch:
      file_temp.write(str(line) + " " + str(file[line]) + '\n')
  file_temp.close()


if __name__ == '__main__':
  main()
