import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path
from typing import List, Type
import matplotlib.pyplot as plt

def spatial_argmax(logit: torch.Tensor) -> torch.Tensor:
    """
    Computes the soft-argmax of a heatmap.
    :param logit: A tensor of size (BS, H, W)
    :return: A tensor of size (BS, 2) containing the soft-argmax in normalized coordinates (-1 to 1)
    """
    BS, H, W = logit.size()
    weights = F.softmax(logit.view(BS, -1), dim=-1).view(BS, H, W)
    grid_x = torch.linspace(-1, 1, W, device=logit.device)
    grid_y = torch.linspace(-1, 1, H, device=logit.device)
    x_coords = (weights.sum(dim=1) * grid_x).sum(dim=1)
    y_coords = (weights.sum(dim=2) * grid_y).sum(dim=1)
    return torch.stack((x_coords, y_coords), dim=1)

class BlockConv(nn.Module):
    def __init__(self, n_input, n_output, kernel_size=3, stride=1, dilation=1, residual=True, activation_fn=nn.ReLU):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.residual = residual
        self.downsample = nn.Sequential(
            nn.Conv2d(n_input, n_output, 1, stride, bias=False),
            nn.BatchNorm2d(n_output)
        ) if stride != 1 or n_input != n_output else None

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_input, n_output, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm2d(n_output),
            activation_fn(inplace=True),
            nn.Conv2d(n_output, n_output, kernel_size, 1, padding, dilation, bias=False),
            nn.BatchNorm2d(n_output),
            activation_fn(inplace=True),
            nn.Conv2d(n_output, n_output, kernel_size, 1, padding, dilation, bias=False),
            nn.BatchNorm2d(n_output),
            activation_fn(inplace=True),
            nn.Dropout(0.1)  # adding dropout for regularization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x
        x = self.conv_layers(x)
        return x + identity if self.residual else x

class Planner(nn.Module):
    def __init__(self, dim_layers: List[int], n_input: int = 3, input_normalization: bool = True,
                 skip_connections: bool = False, residual: bool = True, activation_fn: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.norm = nn.BatchNorm2d(n_input) if input_normalization else None
        self.skip_connections = skip_connections
        self.min_size = 2 ** (len(dim_layers) + 1)

        current_channels = n_input
        self.conv_layers = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(n_input, dim_layers[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(dim_layers[0]),
            activation_fn(inplace=True)
        )

        for channels in dim_layers:
            self.conv_layers.append(BlockConv(current_channels, channels, stride=2, residual=residual, activation_fn=activation_fn))
            current_channels = channels

        for channels in reversed(dim_layers):
            output_channels = dim_layers[dim_layers.index(channels)-1] if dim_layers.index(channels) > 0 else 1
            self.upconv_layers.insert(0, BlockConv(channels, output_channels, stride=2, residual=residual, activation_fn=activation_fn))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm:
            x = self.norm(x)

        x = self.initial_conv(x)
        skip_connections = []

        for layer in self.conv_layers:
            x = layer(x)
            skip_connections.append(x)

        for layer in self.upconv_layers:
            if self.skip_connections and skip_connections:
                x = torch.cat([x, skip_connections.pop()], dim=1)
            x = layer(x)

        return spatial_argmax(x)

def save_model(model: nn.Module, filename: str = 'planner.th') -> None:
    torch.save({'model_state': model.state_dict()}, filename)

def load_model(filename: str = 'planner.th', model_params: dict = {}) -> Planner:
    model = Planner(**model_params)
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state'])
    return model

if __name__ == "__main__":
    model = Planner([32, 64, 128])
    save_model(model, 'planner_model.pth')
    loaded_model = load_model('planner_model.pth', {'dim_layers': [32, 64, 128], 'n_input': 3})
    
# Dummy data for demonstration
epochs = list(range(1, 101))
training_loss_modified = [0.8 - 0.006 * epoch + 0.015 * epoch**0.5 for epoch in epochs]
validation_loss_modified = [1.0 - 0.006 * epoch + 0.025 * epoch**0.5 for epoch in epochs]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(epochs, training_loss_modified, label='Training Loss (Modified)', color='blue')
plt.plot(epochs, validation_loss_modified, label='Validation Loss (Modified)', color='orange')
plt.title('Training and Validation Losses (Modified Version)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()