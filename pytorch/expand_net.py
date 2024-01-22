import torch
from torch import nn
from models import MatrixMultiplier


class ExpandedLinear(nn.Module):
    def __init__(self, in_features, out_features, depth, mode, init_scale, diag_init_scale=0, diag_noise_std=0):
        super(ExpandedLinear, self).__init__()
        self.matrix_module = MatrixMultiplier(depth, in_features, mode, init_scale, diag_init_scale, diag_noise_std, out_features)

    def forward(self, x):
        # Get the end-to-end weight matrix from your custom module
        weight_matrix = self.matrix_module()

        # Perform the matrix multiplication with the input
        return torch.matmul(x, weight_matrix)


def replace_linear_layers(module, d, mode='real', init_scale=0.001):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Replace with custom ExpandedLinear module
            setattr(module, name, ExpandedLinear(child.in_features, child.out_features, d, mode, init_scale))
        else:
            replace_linear_layers(child, d, new_layer_size)

# Example usage
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
