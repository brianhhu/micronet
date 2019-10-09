import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from heapq import nsmallest
from itertools import groupby
from operator import itemgetter

from prune import rgetattr, rsetattr
from models import WRN_McDonnell


def create_dataset(train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409),
                    (0.2673, 0.2564, 0.2762)),
    ])
    return getattr(datasets, 'CIFAR100')('../cifar-data', train=train, download=True, transform=transform)


class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.handle_list = []
        self.activations = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.named_modules()):
            # Note: we only select bn1 outputs
            if 'bn1' in name:
                self.handle_list += [module.register_forward_hook(self.get_activation_and_grad)]
                self.activation_to_layer[activation_index] = name
                activation_index += 1

        # forward pass
        out = model(x)

        # delete hooks
        for h in self.handle_list:
            h.remove()

        return out

    def get_activation_and_grad(self, m, i, o):
        self.activations.append(o)
        o.register_hook(self.compute_rank)

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter,
        # across all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            if args.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / torch.sqrt(torch.sum(v*v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune


class PrunningFineTuner:
    def __init__(self, model):
        self.train_data_loader = DataLoader(create_dataset(train=True), 32)
        self.test_data_loader = DataLoader(create_dataset(train=False), 32)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()

    def test(self):
        # return
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            if args.use_cuda:
                batch = batch.cuda()
            output = model(batch)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy :", float(correct) / total)

        self.model.train()

    def train(self, optimizer=None, epochs=10):
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

        for i in range(epochs):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")

    def train_batch(self, optimizer, batch, label, rank_filters):
        if args.use_cuda:
            batch = batch.cuda()
            label = label.cuda()

        self.model.zero_grad()
        input = batch

        if rank_filters:
            output = self.prunner.forward(input)
            self.criterion(output, label).backward()
        else:
            self.criterion(self.model(input), label).backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, rank_filters=False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)
            break

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters=True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name, param in self.model.named_parameters():
            filters = filters + param.shape[0]
        return filters

    def prune(self):
        #Get the accuracy before prunning
        self.test()
        self.model.train()

        #Make sure all the layers are trainable
        for param in self.model.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 32  # 512
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = 5  # int(iterations * 2.0 / 3)

        print("Number of prunning iterations", iterations)

        for _ in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            prune_targets = sorted([(k, list(list(zip(*g))[1])) for k, g in groupby(prune_targets, itemgetter(0))])
            layers_prunned = {}
            for layer_name, filter_index in prune_targets:
                layers_prunned[layer_name] = len(filter_index)

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()
            for layer_name, filter_index in prune_targets:
                # conv0
                name = layer_name[:-3]+'conv0'
                param = rgetattr(model, name)
                clust_id = np.setdiff1d(list(range(param.shape[0])), filter_index)
                n_clust = len(clust_id)
                rsetattr(model, name, nn.Parameter(param[clust_id]))

                # bn1
                old_bn = rgetattr(model, layer_name)
                new_bn = nn.BatchNorm2d(num_features = n_clust, affine = False).to('cuda')
                new_bn.running_mean = old_bn.running_mean[clust_id]
                new_bn.running_var = old_bn.running_var[clust_id]
                new_bn.num_batches_tracked = old_bn.num_batches_tracked
                rsetattr(model, layer_name, new_bn)

                # conv1
                name = layer_name[:-3]+'conv1'
                param = rgetattr(model, name)
                rsetattr(model, name, nn.Parameter(param[:, clust_id]))

            self.model = model
            if args.use_cuda:
                self.model = self.model.cuda()

            message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)  # Should lr be 0.0001?
            self.train(optimizer, epochs = 10)


        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epochs=15)
        torch.save(model.state_dict(), "model_prune.pt")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args


if __name__ == '__main__':
    args = get_args()

    model = WRN_McDonnell(20, 10, 100, binarize=True)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            rsetattr(model, name, torch.nn.BatchNorm2d(module.num_features, affine=False))

    if args.prune:
        model.load_state_dict(torch.load("checkpoints/model.pt"))

    if args.use_cuda:
        model = model.cuda()

    fine_tuner = PrunningFineTuner(model)

    if args.train:
        fine_tuner.train(epochs=10)
        torch.save(model, "model_finetune.pt")

    elif args.prune:
        fine_tuner.prune()
