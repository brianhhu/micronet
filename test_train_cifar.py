import test_utils

FLAGS = test_utils.parse_common_options(
    datadir='../cifar-data',
    batch_size=128,
    num_epochs=300,
    momentum=0.9,
    lr=0.1,
    target_accuracy=80.0)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import torchvision
import torchvision.transforms as transforms

# Import utilities and models
from utilities import Cutout
from models import BaiduNet8, ResNet9, ResNet18, WRN_McDonnell


def train_cifar():
  print('==> Preparing data..')

  if FLAGS.fake_data:
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 3, 32,
                          32), torch.zeros(FLAGS.batch_size,
                                           dtype=torch.int64)),
        sample_count=50000 // FLAGS.batch_size)
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 3, 32,
                          32), torch.zeros(FLAGS.batch_size,
                                           dtype=torch.int64)),
        sample_count=10000 // FLAGS.batch_size)
  else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(8),  # add Cutout
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
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

  devices = xm.get_xla_supported_devices(max_devices=FLAGS.num_cores)

  # Select model here
  # model = BaiduNet8()
  model = ResNet9(40, 80, 160, 320)
  # model = ResNet18()
  # model = WRN_McDonnell(20, 10, 100, binarize=True)

  # Pass [] as device_ids to run using the PyTorch/CPU engine.
  model_parallel = dp.DataParallel(model, device_ids=devices)

  def train_loop_fn(model, loader, device, context):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=FLAGS.lr,
        momentum=FLAGS.momentum,
        weight_decay=5e-4)

    # LR scheduler
    scheduler = MultiStepLR(optimizer, milestones=[140, 190], gamma=0.1)

    tracker = xm.RateTracker()

    for x, (data, target) in loader:
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      xm.optimizer_step(optimizer)
      tracker.add(FLAGS.batch_size)
      if x % FLAGS.log_steps == 0:
        print('[{}]({}) Loss={:.5f} Rate={:.2f}'.format(device, x, loss.item(),
                                                        tracker.rate()))
    return scheduler

  def test_loop_fn(model, loader, device, context):
    total_samples = 0
    correct = 0
    for x, (data, target) in loader:
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
      total_samples += data.size()[0]

    print('[{}] Accuracy={:.2f}%'.format(device,
                                         100.0 * correct / total_samples))
    return correct / total_samples

  best_accuracy = 0.0
  for epoch in range(1, FLAGS.num_epochs + 1):
    scheduler = model_parallel(train_loop_fn, train_loader)

    # Step LR scheduler
    [s.step() for s in scheduler]

    accuracies = model_parallel(test_loop_fn, test_loader)
    accuracy = sum(accuracies) / len(devices)

    # Keep track of best model
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      torch.save(model.state_dict(), 'model.pt')

    if FLAGS.metrics_debug:
      print(torch_xla._XLAC._xla_metrics_report())

  return accuracy * 100.0


# Train the model
torch.set_default_tensor_type('torch.FloatTensor')
acc = train_cifar()

print('Final accuracy: {}'.format(acc))
