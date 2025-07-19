# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time

import torch

from models.STickNet import *
from models.datasets import *
from util import get_device
import writeLogAcc as wA


def get_args():
  parser.add_argument(
      '--verbal',
      action='store_true',
      help='Print per-epoch training/validation output if set.'
  )
  """
  Parse the command line arguments for STickNet training.

  Returns:
    argparse.Namespace: Parsed command line arguments with training configuration
  """
  parser = argparse.ArgumentParser(
      description='STickNet training script for image classification on CIFAR-10, CIFAR-100, and Stanford Dogs datasets. '
                  'Supports multiple architecture configurations and comprehensive training options.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  # Dataset configuration
  parser.add_argument(
      '-r',
      '--data-root',
      type=str,
      default='../../../datasets/StanfordDogs',
      help='Root directory path containing the dataset. For Stanford Dogs, this should point to the folder containing train/val subdirectories.',
  )
  parser.add_argument(
      '-d',
      '--dataset',
      type=str,
      choices=['cifar10', 'cifar100', 'dogs'],
      default='dogs',
      help='Dataset to use for training. CIFAR-10 (10 classes), CIFAR-100 (100 classes), or Stanford Dogs (120 classes).',
  )
  parser.add_argument(
      '--download',
      action='store_true',
      help='Automatically download the specified dataset if not already present. Useful for CIFAR datasets.',
  )

  # Model architecture configuration
  parser.add_argument(
      '--architecture-types',
      nargs='+',
      default=['basic'],
      choices=['basic', 'small', 'large'],
      help='STickNet architecture variants to train. Options: basic (balanced), small (compact), large (expanded). '
           'Multiple architectures can be specified for comparison.',
  )
  parser.add_argument(
      '--config',
      default=0,
      type=int,
      help='Configuration index for model architecture. Different indices may use different channel configurations '
           'within the same architecture type.',
  )

  # Hardware and performance settings
  parser.add_argument(
      '-g',
      '--gpu-id',
      default=1,
      type=int,
      help='GPU device ID to use for training. Set to -1 to use CPU, 0 for first GPU, 1 for second GPU, etc. '
           'Supports CUDA, MPS (Apple Silicon), and CPU fallback.',
  )
  parser.add_argument(
      '-j',
      '--workers',
      default=4,
      type=int,
      help='Number of data loading worker processes. Higher values can speed up data loading but use more memory. '
           'Recommended: 4-8 for most systems.',
  )

  # Training hyperparameters
  parser.add_argument(
      '-b',
      '--batch-size',
      default=64,
      type=int,
      help='Training batch size. Larger batches may provide more stable gradients but require more GPU memory. '
           'Adjust based on your GPU capacity.',
  )
  parser.add_argument(
      '-e',
      '--epochs',
      default=200,
      type=int,
      help='Total number of training epochs. Each epoch processes the entire training dataset once. '
           'Longer training may improve accuracy but risks overfitting.',
  )
  parser.add_argument(
      '-l',
      '--learning-rate',
      default=0.1,
      type=float,
      help='Initial learning rate for the optimizer. Higher values may converge faster but risk instability. '
           'Typical range: 0.01-0.1 for SGD.',
  )
  parser.add_argument(
      '-s',
      '--schedule',
      nargs='+',
      default=[100, 150, 180],
      type=int,
      help='Learning rate schedule milestones. The learning rate will be reduced by a factor of 10 at these epochs. '
           'Format: space-separated list of epoch numbers (e.g., "50 100 150").',
  )
  parser.add_argument(
      '-m',
      '--momentum',
      default=0.9,
      type=float,
      help='SGD momentum parameter. Controls the contribution of previous gradients to current update. '
           'Typical range: 0.8-0.99. Higher values can help escape local minima.',
  )
  parser.add_argument(
      '-w',
      '--weight-decay',
      default=1e-4,
      type=float,
      help='Weight decay (L2 regularization) coefficient. Helps prevent overfitting by penalizing large weights. '
           'Typical range: 1e-5 to 1e-3.',
  )

  # Output and evaluation settings
  parser.add_argument(
      '--base-dir',
      type=str,
      default='.',
      help='Base directory for saving model checkpoints, logs, and results. '
           'Checkpoints will be saved in base_dir/checkpoints/architecture_name/.',
  )
  parser.add_argument(
      '--evaluate',
      dest='evaluate',
      action='store_true',
      help='Run evaluation mode instead of training. Loads the best model checkpoint and evaluates on validation set. '
           'Useful for testing trained models or computing final accuracy.',
  )

  return parser.parse_args()


def get_data_loader(dataset_name, data_root, batch_size, workers, download=False, train=True):
  """
  Return the data loader for the given dataset configuration.

  Args:
    dataset_name (str): Name of the dataset ('cifar10', 'cifar100', 'dogs')
    data_root (str): Root directory path containing the dataset
    batch_size (int): Batch size for the DataLoader
    workers (int): Number of data loading worker processes
    download (bool): Whether to download the dataset if not present
    train (bool): Whether to load training or validation data

  Returns:
    torch.utils.data.DataLoader: DataLoader for the specified dataset
  """
  if dataset_name in ('cifar10', 'cifar100'):
    # select transforms based on train/val
    if train:
      transform = torchvision.transforms.Compose(
          [
              torchvision.transforms.RandomCrop(32, padding=4),
              torchvision.transforms.RandomHorizontalFlip(),
              torchvision.transforms.ToTensor(),
          ]
      )
    else:
      transform = torchvision.transforms.Compose(
          [
              torchvision.transforms.ToTensor(),
          ]
      )

    # cifar10 vs. cifar100
    if dataset_name == 'cifar10':
      dataset_class = torchvision.datasets.CIFAR10
    else:
      dataset_class = torchvision.datasets.CIFAR100

  elif dataset_name in ('dogs',):
    # select transforms based on train/val
    if train:
      transform = torchvision.transforms.Compose(
          [
              torchvision.transforms.Resize(size=(256, 256)),
              torchvision.transforms.RandomCrop(224),
              torchvision.transforms.RandomHorizontalFlip(),
              torchvision.transforms.ColorJitter(0.4),
              torchvision.transforms.ToTensor(),
          ]
      )
    else:
      transform = torchvision.transforms.Compose(
          [
              torchvision.transforms.Resize(size=(256, 256)),
              torchvision.transforms.CenterCrop(224),
              torchvision.transforms.ToTensor(),
          ]
      )

    # dataset_class = models.datasets.StanfordDogs
    dataset_class = StanfordDogs

  else:
    raise NotImplementedError(
        'Can\'t determine data loader for dataset \'{}\''.format(dataset_name)
    )

  # trigger download only once
  if download:
    dataset_class(
        root=data_root, train=train, download=True, transform=transform
    )

  # instantiate dataset class and create data loader from it
  dataset = dataset_class(
      root=data_root, train=train, download=False, transform=transform
  )
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=True if train else False,
      num_workers=workers,
  )


def calculate_accuracy(output, target):
  """
  Top-1 classification accuracy.
  """
  with torch.no_grad():
    batch_size = output.shape[0]
    prediction = torch.argmax(output, dim=1)
    return torch.sum(prediction == target).item() / batch_size


def run_epoch(train, data_loader, model, criterion, optimizer, n_epoch, total_epochs, device):
  """
  Run one epoch. If `train` is `True` perform training, otherwise validate.

  Args:
    train (bool): Whether to train or validate
    data_loader (torch.utils.data.DataLoader): Data loader for the epoch
    model (torch.nn.Module): The model to train/validate
    criterion (torch.nn.Module): Loss function
    optimizer (torch.optim.Optimizer): Optimizer (can be None for validation)
    n_epoch (int): Current epoch number (0-indexed)
    total_epochs (int): Total number of epochs for progress display
    device (torch.device): Device to run the model on

  Returns:
    tuple: (average_loss, average_accuracy)
  """
  if train:
    model.train()
    torch.set_grad_enabled(True)
  else:
    model.eval()
    torch.set_grad_enabled(False)

  batch_count = len(data_loader)
  losses = []
  accs = []
  for n_batch, (images, target) in enumerate(data_loader):
    images = images.to(device)
    target = target.to(device)

    output = model(images)
    loss = criterion(output, target)

    # record loss and measure accuracy
    loss_item = loss.item()
    losses.append(loss_item)
    acc = calculate_accuracy(output, target)
    accs.append(acc)

    # compute gradient and do SGD step
    if train:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    if (n_batch % 10) == 0:
      print(
          '[{}]  epoch {}/{},  batch {}/{},  loss_{}={:.5f},  acc_{}={:.2f}%'.format(
              'train' if train else ' val ',
              n_epoch + 1,
              total_epochs,
              n_batch + 1,
              batch_count,
              "train" if train else "val",
              loss_item,
              "train" if train else "val",
              100.0 * acc,
          )
      )

  return (sum(losses) / len(losses), sum(accs) / len(accs))


def main():
  """
  Run the complete model training.
  """
  args = get_args()
  print('Command: {}'.format(' '.join(sys.argv)))

  device = get_device(args.gpu_id)
  print('Using device {}'.format(device))

  # print model with parameter and FLOPs counts
  torch.autograd.set_detect_anomaly(True)

  # Set the base directory
  arr_architecture_types = args.architecture_types
  cf_index = args.config
  for typesize in arr_architecture_types:
    strmode = f'StanfordDogs_S_TickNet_{typesize}_SE_config_{cf_index}'
    pathout = f'{args.base_dir}/checkpoints/{strmode}'

    filenameLOG = pathout + '/' + strmode + '.txt'
    result_file_path = pathout + '/' + strmode + '.csv'
    if not os.path.exists(pathout):
      os.makedirs(pathout)

    # get model
    model = build_SpatialTickNet(120, typesize=typesize, cifar=False)
    model = model.to(device)

    print(model)
    print(
        'Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=args.schedule, gamma=0.1
    )

    # get train and val data loaders
    train_loader = get_data_loader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        download=args.download,
        train=True
    )
    val_loader = get_data_loader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        download=args.download,
        train=False
    )

    if args.evaluate:
      pathcheckpoint = f'{args.base_dir}/checkpoints/StanfordDogs_S_TickNet/{strmode}/model_best.pth'
      if os.path.isfile(pathcheckpoint):
        print("=> loading checkpoint '{}'".format(pathcheckpoint))
        checkpoint = torch.load(pathcheckpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
      else:
        print("=> no checkpoint found at '{}'".format(pathcheckpoint))
        return
      m = time.time()
      (val_loss, val_accuracy) = run_epoch(
          train=False,
          data_loader=val_loader,
          model=model,
          criterion=criterion,
          optimizer=None,
          n_epoch=0,
          total_epochs=args.epochs,
          device=device,
      )
      print(
          f'[ validating: ], loss_val={val_loss:.5f}, acc_val={100.0 * val_accuracy:.2f}%'
      )
      n = time.time()
      print((n - m) / 3600)
      return

    # for each epoch...
    val_accuracy_max = None
    val_accuracy_argmax = None
    for n_epoch in range(args.epochs):
      current_learning_rate = optimizer.param_groups[0]['lr']
      if args.verbal:
          print(
              f'Starting epoch {n_epoch + 1}/{args.epochs}, learning_rate={current_learning_rate}'
          )
  
      # train
      (train_loss, train_accuracy) = run_epoch(
          train=True,
          data_loader=train_loader,
          model=model,
          criterion=criterion,
          optimizer=optimizer,
          n_epoch=n_epoch,
          total_epochs=args.epochs,
          device=device,
      )

      # validate
      (val_loss, val_accuracy) = run_epoch(
          train=False,
          data_loader=val_loader,
          model=model,
          criterion=criterion,
          optimizer=None,
          n_epoch=n_epoch,
          total_epochs=args.epochs,
          device=device,
      )
      if (val_accuracy_max is None) or (val_accuracy > val_accuracy_max):
        val_accuracy_max = val_accuracy
        val_accuracy_argmax = n_epoch
        torch.save(
            {"model_state_dict": model.state_dict()},
            f'{pathout}/checkpoint_epoch{n_epoch + 1:>04d}_{100.0 * val_accuracy_max:.2f}.pth',
        )

      # adjust learning rate
      scheduler.step()

      # print epoch summary
      line = (
          '=================================================================================='
          f'Epoch {n_epoch + 1}/{args.epochs} summary: '
          f'loss_train={train_loss:.5f}, '
          f'acc_train={100.0 * train_accuracy:.2f}%, '
          f'loss_val={val_loss:.2f}, '
          f'acc_val={100.0 * val_accuracy:.2f}% '
          f'(best: {100.0 * val_accuracy_max:.2f}% @ epoch {(val_accuracy_argmax or 0) + 1})'
          '=================================================================================='
      )
      if args.verbal:
        print(line)
      wA.writeLogAcc(filenameLOG, line)
      wA.log_results_to_csv(
          result_file_path,
          n_epoch + 1,
          train_loss,
          100.0 * train_accuracy,
          val_loss,
          100.0 * val_accuracy,
      )


if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print('Stopped')
    sys.exit(0)
  except Exception as e:
    print('Error: {}'.format(e))
    sys.exit(1)
