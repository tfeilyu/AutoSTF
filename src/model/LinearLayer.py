import torch
import torch.nn as nn
import torch.nn.functional as F

class LightLayer(nn.Module):
    def __init__(self, hid_dim, dim_feedforward):
        super(LightLayer, self).__init__()

        layer_norm_eps = 1e-5
        self.hid_dim = hid_dim

        self.linear1 = nn.Linear(hid_dim, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward // 2, self.hid_dim // 2)

        self.norm1 = nn.LayerNorm(self.hid_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.hid_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = F.gelu

    def forward(self, inputs):
        x = inputs
        x1 = self.norm1(x)

        b, l, d = x1.size()
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x1))).view(b, l, 2, d * 4 // 2))
        x2 = x2.view(b, l, d)
        x2 = self.dropout2(x2)
        x = self.norm2(x1 + x2)

        return x


class LightLinear(nn.Module):
    def __init__(self, config):
        super(LightLinear, self).__init__()

        self.hid_dim = config.hidden_channels
        self.layers = config.num_linear_layers

        self.LightLayers = nn.ModuleList()
        for _ in range(self.layers):
            self.LightLayers.append(LightLayer(self.hid_dim, self.hid_dim*4))

    def forward(self, inputs):
        # inputs: 64, 207, 32
        x = inputs.permute(1, 0, 2)
        for i, layer in enumerate(self.LightLayers):
            x = layer(x)

        output = x.permute(1, 0, 2)
        return output


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden

