import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from models import WRN_McDonnell_Eval


def create_dataset(train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409),
                    (0.2673, 0.2564, 0.2762)),
    ])
    return getattr(datasets, 'CIFAR100')('../cifar-data', train=train, download=True, transform=transform)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    import functools
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def save_conv_output(activations, name):
    """
    Saves layer output in activations dict with name key
    """
    def get_activation(m, i, o):
        activations[name] = o.data.cpu().numpy()

    return get_activation


def extract_features(model, input):
    """
    model: Pytorch model
    input: batch of data to pass through model (B x C x H x W)
    """
    handle_list = []
    activations = {}

    # register hooks
    for name, module in model.named_modules():
        # Note: we only select rectified outputs
        # if isinstance(module, nn.ReLU):
        if 'relu1' in name:
            handle_list += [module.register_forward_hook(
                save_conv_output(activations, name))]

    # forward pass
    _ = model(input)

    # remove hooks
    for h in handle_list:
        h.remove()

    return activations


def activation_channels_l1(activation):
    """Calculate the L1-norms of an activation's channels.
    The activation usually has the shape: (batch_size, num_channels, h, w).
    Returns - for each channel: the batch-mean of its L1 magnitudes (i.e. over all of the
    activations in the mini-batch, compute the mean of the L! magnitude of each channel).
    """
    if activation.ndim == 4:
        view_2d = activation.reshape(activation.shape[0], activation.shape[1], -1)
        featuremap_norms_mat = np.linalg.norm(view_2d, ord=1, axis=2)
    elif activation.ndim == 2:
        featuremap_norms_mat = np.linalg.norm(activation, ord=1, axis=1)  # batch x 1
    else:
        raise ValueError("activation_channels_l1: Unsupported shape: ".format(activation.shape))

    return featuremap_norms_mat.mean(axis=0)


def activation_channels_means(activation):
    """Calculate the mean of each of an activation's channels.
    The activation usually has the shape: (batch_size, num_channels, h, w).
    "We first use global average pooling to convert the output of layer i, which is a
    c x h x w tensor, into a 1 x c vector."
    Returns - for each channel: the batch-mean of its mean magnitudes (i.e. over all of the
    activations in the mini-batch, compute the mean of the mean magnitude of each channel).
    """
    if activation.ndim == 4:
        featuremap_means_mat = activation.mean(axis=(2, 3))
    elif activation.ndim == 2:
        featuremap_means_mat = activation.mean(axis=1)  # batch x 1
    else:
        raise ValueError("activation_channels_means: Unsupported shape: ".format(activation.shape))

    return featuremap_means_mat.mean(axis=0)


# IDEA: Max over space, and L2 norm over examples
def activation_channels_max(activation):
    """Calculate the max of each of an activation's channels.
    The activation usually has the shape: (batch_size, num_channels, h, w).
    "We first use max pooling to convert the output of layer i, which is a
    c x h x w tensor, into a 1 x c vector."
    Returns - for each channel: the batch-mean of its max magnitudes (i.e. over all of the
    activations in the mini-batch, compute the mean of the max magnitude of each channel).
    """
    if activation.ndim == 4:
        featuremap_max_mat = activation.max(axis=(2, 3))
    elif activation.ndim == 2:
        featuremap_max_mat = activation.max(axis=1)  # batch x 1
    else:
        raise ValueError("activation_channels_means: Unsupported shape: ".format(activation.shape))

    return featuremap_max_mat.mean(axis=0)


def activation_channels_apoz(activation):
    """Calculate the APoZ of each of an activation's channels.
    APoZ is the Average Percentage of Zeros (or simply: average sparsity) and is defined in:
    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures".
    The activation usually has the shape: (batch_size, num_channels, h, w).
    "We first use global average pooling to convert the output of layer i, which is a
    c x h x w tensor, into a 1 x c vector."
    Returns - for each channel: the batch-mean of its sparsity.
    """
    if activation.ndim == 4:
        featuremap_apoz_mat = (np.abs(activation) > 0).sum(axis=(2, 3)) / (activation.shape[2] * activation.shape[3])
    elif activation.ndim == 2:
        featuremap_apoz_mat = (np.abs(activation) > 0).sum(axis=1) / activation.shape[1]  # batch x 1
    else:
        raise ValueError("activation_channels_apoz: Unsupported shape: ".format(activation.shape))
    return 100 - featuremap_apoz_mat.mean(axis=0)*100


# From "Redundant feature pruning for accelerated inference in deep neural networks"
def cluster_weights_agglo(weight, threshold):
    import scipy.cluster.hierarchy as hac

    # cosine/hamming metric, complete/average method
    threshold = 1.0 - threshold   # Conversion to distance measure
    z = hac.linkage(weight, metric='hamming', method='complete')
    labels = hac.fcluster(z, threshold, criterion='distance')

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    a = np.array(labels)
    sort_idx = np.argsort(a)
    a_sorted = a[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    first_ele = [unq_idx[idx][-1] for idx in range(len(unq_idx))]

    return n_clusters_, first_ele


def main():
    # Create model
    model = WRN_McDonnell_Eval(20, 10, 100)
    model.load_state_dict(torch.load('./checkpoints/model_trained.pt'))
    model.to("cuda")
    model.eval()

    # Create dataloader
    train_data_loader = DataLoader(create_dataset(train=True), 32)
    data_loader = DataLoader(create_dataset(train=False), 32)

    """
    Activation-based pruning
    """
    # Get a batch of inputs
    outputs = None
    for inputs, _ in train_data_loader:
        with torch.no_grad():

            inputs = inputs.to('cuda')

            if outputs is None:
                outputs = extract_features(model, inputs)
                break
            # else:
            #     temp = extract_features(model, inputs)
            #     for key, val in outputs.items():
            #         outputs[key] = np.concatenate((val, temp[key]))

    # Fraction of channels to keep
    keep_frac = 0.95

    # Compute masks
    mask_dict = {}
    for key, val in outputs.items():
        # choose channel importance metric
        # channel_act = activation_channels_l1(val)  # 75.07
        # channel_act = activation_channels_means(val)  # 75.07
        # channel_act = activation_channels_max(val)  # 71.16
        channel_act = activation_channels_apoz(val)  # 72.74

        # sort by metric
        ids = np.argsort(channel_act)[::-1]
        ids = ids[:int(keep_frac*len(channel_act))]

        mask_dict[key] = np.zeros_like(channel_act)
        mask_dict[key][ids] = 1

    # Convert dict to list of tuples
    mask_list = [(k, v) for k, v in mask_dict.items()]

    # Iterate over model
    cnt = 0
    for (name, module) in list(model.named_modules())[2:]:
        if 'conv0' in name:
            clust_id = np.where(mask_list[cnt][1])[0].tolist()
            n_clust = len(clust_id)

            # Create new conv
            weight = module.weight.data.cpu().numpy()
            new_conv = nn.Conv2d(in_channels = module.in_channels,
                                 out_channels = n_clust,
                                 kernel_size = module.kernel_size,
                                 stride = module.stride,
                                 padding = module.padding,
                                 bias = (module.bias is not None))
            new_conv.weight.data = torch.from_numpy(weight[clust_id]).to('cuda')
            rsetattr(model, name, new_conv)
        if 'bn1' in name:
            # Create new bn
            new_bn = nn.BatchNorm2d(num_features = n_clust, affine = False).to('cuda')
            new_bn.running_mean = module.running_mean[clust_id]
            new_bn.running_var = module.running_var[clust_id]
            new_bn.num_batches_tracked = module.num_batches_tracked
            rsetattr(model, name, new_bn)
        if 'conv1' in name:
            # Create new conv
            weight = module.weight.data.cpu().numpy()
            new_conv = nn.Conv2d(in_channels = n_clust,
                                 out_channels = module.out_channels,
                                 kernel_size = module.kernel_size,
                                 stride = module.stride,
                                 padding = module.padding,
                                 bias = (module.bias is not None))
            new_conv.weight.data = torch.from_numpy(weight[:, clust_id]).to('cuda')
            # Re-scale weight for next conv
            new_conv.weight.data = new_conv.weight.data.sign() * math.sqrt(2. /
                                (n_clust * new_conv.weight.data.shape[2] * new_conv.weight.data.shape[3]))
            rsetattr(model, name, new_conv)
            cnt += 1

    # # Check if activations are zeroed out
    # outputs = extract_features(model, inputs)
    # outputs_filtered = {key: outputs[key] for key in outputs.keys() if 'relu1' in key}

    # # Confirm units are actually zero-ed out
    # [l[:,np.where(ids==0)].sum() for l,ids in zip(outputs_filtered.values(), mask_dict.values())]


    # """
    # Weight-based pruning
    # """
    # threshold = 0.75
    # # Iterate over model
    # for (name, module) in list(model.named_modules())[2:]:
    #     if 'conv0' in name:
    #         weight = module.weight.data.cpu().numpy()

    #         # clust_id are the filters to keep
    #         n_clust, clust_id = cluster_weights_agglo(weight.reshape(weight.shape[0], -1), threshold)
    #         print(n_clust)

    #         # Create new conv
    #         new_conv = nn.Conv2d(in_channels = module.in_channels,
    #                              out_channels = n_clust,
    #                              kernel_size = module.kernel_size,
    #                              stride = module.stride,
    #                              padding = module.padding,
    #                              bias = (module.bias is not None))
    #         new_conv.weight.data = torch.from_numpy(weight[clust_id]).to('cuda')
    #         rsetattr(model, name, new_conv)
    #     if 'bn1' in name:
    #         # Create new bn
    #         new_bn = nn.BatchNorm2d(num_features = n_clust, affine = False).to('cuda')
    #         new_bn.running_mean = module.running_mean[clust_id]
    #         new_bn.running_var = module.running_var[clust_id]
    #         new_bn.num_batches_tracked = module.num_batches_tracked
    #         rsetattr(model, name, new_bn)
    #     if 'conv1' in name:
    #         # Create new conv
    #         weight = module.weight.data.cpu().numpy()
    #         new_conv = nn.Conv2d(in_channels = n_clust,
    #                              out_channels = module.out_channels,
    #                              kernel_size = module.kernel_size,
    #                              stride = module.stride,
    #                              padding = module.padding,
    #                              bias = (module.bias is not None))
    #         new_conv.weight.data = torch.from_numpy(weight[:, clust_id]).to('cuda')
    #         # Re-scale weight for next conv
    #         new_conv.weight.data = new_conv.weight.data.sign() * math.sqrt(2. /
    #                             (n_clust * new_conv.weight.data.shape[2] * new_conv.weight.data.shape[3]))
    #         rsetattr(model, name, new_conv)

    model.eval()
    correct = 0
    total = 0
    for inputs, targets in data_loader:
        with torch.no_grad():

            inputs, targets = inputs.to('cuda'), targets.to('cuda')

            outputs = model.forward(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(100.*correct/total)


if __name__ == '__main__':
    main()
