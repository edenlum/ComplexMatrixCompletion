import torch
from torch import nn
from models import MatrixMultiplier


class ExpandedLinear(nn.Module):
    def __init__(self, in_features, out_features, depth, mode, bias, init_scale, diag_init_scale=0, diag_noise_std=0):
        super(ExpandedLinear, self).__init__()
        self.matrix_module = MatrixMultiplier(depth, in_features, mode, init_scale, diag_init_scale, diag_noise_std, out_features)
        self.bias = bias

    def forward(self, x):
        # Get the end-to-end weight matrix from your custom module
        weight_matrix = self.matrix_module()

        # Perform the matrix multiplication with the input
        return torch.matmul(x, weight_matrix) + self.bias


def replace_linear_layers(module, d, mode='real', init_scale=0.001, diag_init_scale=0.001):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Replace with custom ExpandedLinear module
            expanded_linear = ExpandedLinear(child.in_features, child.out_features, d, mode, child.bias, child.weight.std(), diag_init_scale)
            print(f"Replacing {name} with ExpandedLinear")
            print(f"Original norm, std: {child.weight.norm()}, {child.weight.std()}")
            print(f"New norm, std: {expanded_linear.matrix_module().norm()}, {expanded_linear.matrix_module().std()}")
            setattr(module, name, expanded_linear)
        else:
            replace_linear_layers(child, d, mode, init_scale)

if __name__ == "__main__":
    class CustomNet(nn.Module):
        def __init__(self):
            super(CustomNet, self).__init__()
            self.linear1 = nn.Linear(10, 5)
            self.relu = nn.ReLU()
            self.block = nn.Sequential(nn.Linear(5, 3), nn.ReLU())
            self.linear2 = nn.Linear(3, 2)

        def forward(self, x):
            x = self.relu(self.linear1(x))
            x = self.block(x)
            return self.linear2(x)

    original_net = CustomNet()
    depth = 3
    new_layer_size = 4
    replace_linear_layers(original_net, depth, new_layer_size)

    print(original_net)
