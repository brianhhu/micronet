import argparse
import torch
from torch.utils.data import DataLoader
from models import WRN_McDonnell
from utilities import Cutout, RandomPixelPad

from prune import rsetattr
import torchvision.transforms as T
import torchvision.datasets as datasets


def parse_args():
    parser = argparse.ArgumentParser(description='Binary Wide Residual Networks')
    # Model options
    parser.add_argument('--checkpoint', required=True, type=str)
    return parser.parse_args()


def create_dataset(train):
    if train:
        transform = T.Compose([
            T.Lambda(lambda x: RandomPixelPad(x, padding=4)),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            Cutout(18, random_pixel=True),  # add Cutout
            T.Normalize((0.5071, 0.4865, 0.4409),
                        (0.2673, 0.2564, 0.2762)),
         ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4865, 0.4409),
                        (0.2673, 0.2564, 0.2762)),
        ])
    return getattr(datasets, 'CIFAR100')('../cifar-data', train=train, download=True, transform=transform)


def main():
    args = parse_args()

    have_cuda = torch.cuda.is_available()

    def cast(x):
        return x.cuda() if have_cuda else x

    model = WRN_McDonnell(20, 10, 100, binarize=True)
    # Reset batch norm track_running_stats
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            rsetattr(model, name, torch.nn.BatchNorm2d(module.num_features, affine=False))
    checkpoint = torch.load(args.checkpoint)

    # Create dataloader
    train_data_loader = DataLoader(create_dataset(train=True), 1000, shuffle=True)
    data_loader = DataLoader(create_dataset(train=False), 1000)

    model.load_state_dict(checkpoint, strict=False)
    model = cast(model)

    # Learn batch norm means and variances
    model.train()
    for inputs, _ in train_data_loader:
        with torch.no_grad():

            inputs = inputs.to('cuda')

            _ = model.forward(inputs)

    model.eval()
    correct = 0
    total = 0
    for inputs, targets in data_loader:
        with torch.no_grad():

            inputs, targets = cast(inputs), cast(targets)

            outputs = model.forward(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(100.*correct/total)


if __name__ == '__main__':
    main()
