import test_utils

FLAGS = test_utils.parse_common_options(
    datadir='../cifar-data',
    batch_size=125,
    num_epochs=254,
    momentum=0.9,
    lr=0.8)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import torchvision
import torchvision.transforms as transforms

# Import utilities and models
from torch.optim.lr_scheduler import MultiStepLR
from utilities import Cutout, RandomPixelPad, CosineAnnealingRestartsLR
from models import WRN_McDonnell


def train_cifar():
  print('==> Preparing data..')

  transform_train = transforms.Compose([
    transforms.Lambda(lambda x: RandomPixelPad(x, padding=4)),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    Cutout(18, random_pixel=True),  # add Cutout
    transforms.Normalize((0.5071, 0.4865, 0.4409),
                         (0.2673, 0.2564, 0.2762)),
  ])

  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4865, 0.4409),
                           (0.2673, 0.2564, 0.2762)),
  ])

  trainset = torchvision.datasets.CIFAR100(
      root=FLAGS.datadir,
      train=True,
      download=True,
      transform=transform_train)
  train_loader = torch.utils.data.DataLoader(
      trainset,
      batch_size=FLAGS.batch_size,
      shuffle=True,
      num_workers=FLAGS.num_workers)

  testset = torchvision.datasets.CIFAR100(
      root=FLAGS.datadir,
      train=False,
      download=True,
      transform=transform_test)
  test_loader = torch.utils.data.DataLoader(
      testset,
      batch_size=FLAGS.batch_size,
      shuffle=False,
      num_workers=FLAGS.num_workers)

  torch.manual_seed(42)

  devices = (
      xm.get_xla_supported_devices(
          max_devices=FLAGS.num_cores) if FLAGS.num_cores !=0 else [])

  # Define model here
  model = WRN_McDonnell(20, 10, 100, binarize=True)

  # Pass [] as device_ids to run using the PyTorch/CPU engine.
  model_parallel = dp.DataParallel(model, device_ids=devices)

  def train_loop_fn(model, loader, device, context):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = context.getattr_or(
        'optimizer', lambda: optim.SGD(
        model.parameters(),
        lr=FLAGS.lr,
        momentum=FLAGS.momentum,
        weight_decay=5e-4))

    # LR scheduler
    scheduler = context.getattr_or(
        'scheduler', lambda: CosineAnnealingRestartsLR(
        optimizer, T=2, eta_min=1e-4))

    model.train()
    for x, (data, target) in loader:
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      xm.optimizer_step(optimizer)
      if x % FLAGS.log_steps == 0:
        print('[{}]({}) Loss={:.5f}'.format(device, x, loss.item()))

    # Step LR scheduler
    scheduler.step()

  def test_loop_fn(model, loader, device, context):
    total_samples = 0
    correct = 0
    model.eval()
    for x, (data, target) in loader:
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
      total_samples += data.size()[0]

    return correct / total_samples

  best_accuracy = 0.0
  for epoch in range(1, FLAGS.num_epochs + 1):
    model_parallel(train_loop_fn, train_loader)
    accuracies = model_parallel(test_loop_fn, test_loader)
    accuracy = sum(accuracies) / len(devices)

    print('Epoch {}, Accuracy={:.2f}%'.format(epoch, 100.0 * accuracy))

    # Keep track of best model
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      torch.save(model_parallel._models[0].state_dict(), 'model.pt')

    if FLAGS.metrics_debug:
      print(torch_xla._XLAC._xla_metrics_report())

  return accuracy * 100.0


# Train the model
torch.set_default_tensor_type('torch.FloatTensor')
acc = train_cifar()

print('Final accuracy: {}'.format(acc))
