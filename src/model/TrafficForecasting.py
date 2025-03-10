import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from src.model.STLayers import TemporalSearchLayer, SpatialSearchLayer
from src.model.LinearLayer import MultiLayerPerceptron, LightLinear
from src.model.mode import Mode


def create_layer(name, node_embedding_1=None, node_embedding_2=None, adj_mx=None, config=None, device=None):
    if name == 'TemporalSearch':
        return TemporalSearchLayer(node_embedding_1, node_embedding_2, adj_mx, config, device, tag='temporal')
    if name == 'SpatialSearch':
        return SpatialSearchLayer(node_embedding_1, node_embedding_2, adj_mx, config, device, tag='spatial')
    if name == 'ConvPooling':
        return nn.Conv2d(in_channels=config.hidden_channels, out_channels=config.hidden_channels,
                         kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))
    if name == 'AvgPooling':
        return nn.AvgPool2d(kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))
    raise Exception('unknown layer name!')


class AutoSTF(nn.Module):
    def __init__(self, in_length, out_length,
                 mask_support_adj, adj_mx, num_sensors,
                 in_channels, out_channels, hidden_channels, end_channels,
                 layer_names, config, device):
        super(AutoSTF, self).__init__()

        self.config = config
        self.scale_list = config.scale_list
        self.scale_step = int(config.in_length / len(config.scale_list))

        self.in_length = in_length
        self.out_length = out_length
        self.num_sensors = num_sensors

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.end_channels = end_channels

        self.mask_support_adj = mask_support_adj
        self.layer_names = config.layer_names

        self.IsUseLinear = config.IsUseLinear

        # input (64, 2, 207, 12)

        # Start MLP
        self.mlp_node_dim = 32
        self.temp_dim_tid = 32
        self.temp_dim_diw = 32

        self.useDayTime = True
        self.useWeekTime = True

        if self.in_channels == 1:
            self.useDayTime = False
            self.useWeekTime = False
            self.temp_dim_tid = 0
            self.temp_dim_diw = 0
        elif self.in_channels == 2:
            self.useWeekTime = False
            self.temp_dim_diw = 0

        self.num_mlp_layer = config.num_mlp_layers
        self.time_of_day_size = 288
        self.day_of_week_size = 7

        # node embeddings
        self.node_emb = nn.Parameter(torch.empty(self.num_sensors, self.mlp_node_dim))
        nn.init.xavier_uniform_(self.node_emb)

        # temporal embeddings
        if self.useDayTime:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.useWeekTime:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # feature embeddings
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.in_channels * self.in_length,
            out_channels=self.hidden_channels, kernel_size=(1, 1), bias=True)

        # fusion layer
        self.hidden_dim = self.hidden_channels + self.mlp_node_dim + self.temp_dim_tid + self.temp_dim_diw
        self.start_encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_mlp_layer)])
        self.fusion_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.hidden_channels, kernel_size=(1, 1), bias=True)

        # self.start_light = nn.Conv2d(in_channels=self.hidden_channels * self.in_length,
        #                                  out_channels=self.hidden_channels,
        #                                  kernel_size=(1, 1), bias=True)
        if self.IsUseLinear:
            self.light_linear_layer = LightLinear(config)
            self.start_scale_light = nn.Conv2d(in_channels=self.scale_step,
                                                out_channels=1,
                                                kernel_size=(1, 1), bias=True)
            self.start_linear_light = nn.Conv2d(in_channels=self.hidden_channels*2,
                                                out_channels=self.hidden_channels,
                                                kernel_size=(1, 1), bias=True)

        # Spatial Layer
        mask0 = mask_support_adj[0].detach()
        mask1 = mask_support_adj[1].detach()
        mask = mask0 + mask1
        self.mask = mask == 0

        self.node_vec1 = nn.Parameter(torch.randn(self.config.num_sensors, 10).to(device), requires_grad=True).to(
            device)
        self.node_vec2 = nn.Parameter(torch.randn(10, self.config.num_sensors).to(device), requires_grad=True).to(
            device)

        self.TemporalSearchLayers = nn.ModuleList()
        self.SpatialSearchLayers = nn.ModuleList()
        for name in layer_names:
            if name == 'TemporalSearch':
                self.TemporalSearchLayers += [create_layer(name,
                                                           node_embedding_1=self.node_vec1,
                                                           node_embedding_2=self.node_vec2,
                                                           adj_mx=adj_mx, config=config, device=device)]
                self.start_temporal = nn.Conv2d(in_channels=1, out_channels=self.hidden_channels,
                                                kernel_size=(1, 1), bias=True)
                # self.end_temporal = nn.Conv2d(in_channels=self.in_length, out_channels=1,
                #                               kernel_size=(1, 1), bias=True)
            elif name == 'SpatialSearch':
                self.SpatialSearchLayers += [create_layer(name,
                                                          node_embedding_1=self.node_vec1,
                                                          node_embedding_2=self.node_vec2,
                                                          adj_mx=adj_mx, config=config, device=device)]

        self.spatial_fusion = nn.Linear(self.hidden_channels * len(self.scale_list), self.hidden_channels)

        # End Layer
        cnt = 0
        if self.IsUseLinear:
            cnt += 1
        self.end_conv1 = nn.Linear(self.hidden_channels * 2, self.end_channels)
        self.end_conv2 = nn.Linear(self.end_channels, self.out_length * self.out_channels)

    def forward(self, inputs, mode):
        batch_size, num_features, num_nodes, num_timestep = inputs.shape
        # inputs [64, 3, 307, 12]
        self.set_mode(mode)

        # Input Layer
        history_data = inputs.transpose(1, 3)

        # temporal embedding
        input_data = history_data[..., range(self.in_channels)]
        if self.useDayTime:
            # day embeddings
            t_i_d_data = history_data[..., 1]
            day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        if self.useWeekTime:
            # week embeddings
            d_i_w_data = history_data[..., 2]
            week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]

        # feature embedding
        # feature_data = input_data.transpose(1, 3).contiguous()
        # feature_emb = self.feature_emb_layer(feature_data).transpose(1, 3)

        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        # node embeddings
        # a = self.node_emb.unsqueeze(0).expand(num_timestep, -1, -1)
        # node_emb = a.unsqueeze(0).expand(batch_size, -1, -1, -1)

        embeddings_list = []
        embeddings_list += [time_series_emb]
        embeddings_list += [self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)]
        if self.useDayTime:
            embeddings_list += [day_emb.transpose(1, 2).unsqueeze(-1)]
        if self.useWeekTime:
            embeddings_list += [week_emb.transpose(1, 2).unsqueeze(-1)]

        hidden = torch.cat(embeddings_list, dim=1)

        # encoding
        x = self.start_encoder(hidden)

        # fusion
        x = self.fusion_layer(x).squeeze(dim=-1).transpose(1, 2)  # x: 64, 207, 32
        mlp_residual = x

        # Temporal Search
        if 'TemporalSearch' in self.layer_names:
            x_t = inputs[:, 0:1, :, :]
            x_t = self.start_temporal(x_t)
            # x_t = self.TemporalLayer(x_t, self.mask)
            temporal_residual = 0
            for TLayer in self.TemporalSearchLayers:
                x_t = TLayer(x_t, self.mask)
                temporal_residual += x_t

            x_t = temporal_residual.transpose(1, 3)
            # x_t = self.end_temporal(x_t).squeeze(dim=1)

        start_scale = 0
        spatial_embedding = []
        for i in range(len(self.scale_list)):
            x_scale = x_t[:, start_scale:start_scale+self.scale_step, :, :]
            start_scale = start_scale+self.scale_step

            x_scale = self.start_scale_light(x_scale).squeeze(dim=1)
            # if self.IsUseLinear:
            x = torch.cat([mlp_residual] + [x_scale], dim=-1).unsqueeze(dim=-1).transpose(1, 2)
            x = self.start_linear_light(x).squeeze(dim=-1).transpose(1, 2)
            x = self.light_linear_layer(x)
            x = self.SpatialSearchLayers[0](x, i, self.mask)
            spatial_embedding.append(x)
        x = torch.cat(spatial_embedding, dim=-1)
        x = self.spatial_fusion(x)
        spatial_residual = x

        # Spatial Search
        # spatial_residual = 0
        # for SLayer in self.SpatialSearchLayers:
        #     x = SLayer(x, self.mask)
        #     spatial_residual += x

        # Final Outputs
        outputs = [mlp_residual]
        # if self.IsUseLinear:
        #     outputs.append(linear_residual)
        outputs.append(spatial_residual)

        output = torch.cat(outputs, dim=-1)

        # end conv
        x = self.end_conv1(output)
        x = F.relu(x)
        x = self.end_conv2(x)
        x = x.unsqueeze(dim=1)

        self.set_mode(Mode.NONE)

        return x

    def asym_adj(self, adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        return d_mat.dot(adj).astype(np.float32).todense()

    def set_mode(self, mode):
        self._mode = mode
        if 'TemporalSearch' in self.layer_names:
            for l in self.TemporalSearchLayers:
                l.set_mode(mode)

        for l in self.SpatialSearchLayers:
            l.set_mode(mode)

    def weight_parameters(self):
        # # start conv x
        # for m in [self.filter_convs, self.gate_convs]:
        #     for p in m.parameters():
        #         yield p
        # for m in self.conv_x.parameters():
        #     yield m

        # Start MLP
        for m in [self.node_emb]:
            yield m
        if self.useDayTime:
            for m in [self.time_in_day_emb]:
                yield m
        if self.useWeekTime:
            for m in [self.day_in_week_emb]:
                yield m
        for m in [self.time_series_emb_layer]:
            for p in m.parameters():
                yield p
        for m in self.start_encoder:
            for p in m.parameters():
                yield p
        for m in [self.fusion_layer]:
            for p in m.parameters():
                yield p

        # Light Linear
        if self.IsUseLinear:
            for m in [self.light_linear_layer]:
                for p in m.parameters():
                    yield p
            for m in [self.start_scale_light, self.start_linear_light]:
                for p in m.parameters():
                    yield p

        # Temporal Layer
        if 'TemporalSearch' in self.layer_names:
            for m in self.TemporalSearchLayers:
                for p in m.weight_parameters():
                    yield p
            for m in [self.start_temporal]:
                for p in m.parameters():
                    yield p

        # Spatial Layer
        for m in self.SpatialSearchLayers:
            for p in m.weight_parameters():
                yield p
        for m in [self.node_vec1, self.node_vec2]:
            yield m

        # End Layer
        for m in [self.spatial_fusion, self.end_conv1, self.end_conv2]:
            for p in m.parameters():
                yield p

    def arch_parameters(self):
        if 'TemporalSearch' in self.layer_names:
            for m in self.TemporalSearchLayers:
                for p in m.arch_parameters():
                    yield p
        for m in self.SpatialSearchLayers:
            for p in m.arch_parameters():
                yield p
