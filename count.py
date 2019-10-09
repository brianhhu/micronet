import torch
from models.WRN_McDonnell_Eval import WRN_McDonnell_Eval, EltwiseAdd


# Adapted from: https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/cifar/l1-norm-pruning/compute_flops.py
def print_model_param_nums(model, binary_conv=True):
    total_params = sum([param.nelement() for param in model.parameters()])
    if binary_conv:
        total_params /= 32
        # Add fp32 weight multipliers for binary convolution layers
        total_params += len(set([tuple(m.weight.shape[1:]) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]))

    print('  + Number of params: %.2fM' % (total_params / 1e6))

    return total_params


def print_model_param_flops(model, input_res=32, multiply_adds=True, binary_conv=True):
    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)

    # For binary convolution
    def binary_conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # Multiplication with binary weight counts as 1/32 of a flop
        # Add a flop for scaling with fp32 weight multiplier
        flops = (kernel_ops * (1/32 + 1) + bias_ops + 1) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    list_add=[]
    # For elementwise addition
    def add_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()

        # Number of elements many additions.
        flops = batch_size * input_channels * input_height * input_width
        list_add.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                if binary_conv:
                    net.register_forward_hook(binary_conv_hook)
                else:
                    net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            if isinstance(net, EltwiseAdd):
                net.register_forward_hook(add_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    input = torch.rand(3, 3, input_res, input_res)
    _ = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample) + sum(list_add))

    print('  + Number of FLOPs: %.5fG' % (total_flops / 3 / 1e9))

    return total_flops / 3


def main():
    # Load state dict
    state_dict = torch.load('./checkpoints/model_prune.pt')

    # Create model
    model = WRN_McDonnell_Eval(20, 10, 100, cfg=state_dict['config'])

    # Print params and flops
    print_model_param_nums(model)
    print_model_param_flops(model)


if __name__ == '__main__':
    main()
