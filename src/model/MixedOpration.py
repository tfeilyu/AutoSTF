import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.CandidateOpration import create_op
from src.model.mode import Mode


class TemporalLayerMixedOp(nn.Module):
    def __init__(self, node_embedding_1, node_embedding_2, adj_mx, config, device, tag):
        super(TemporalLayerMixedOp, self).__init__()

        used_operations = None
        if tag == 'temporal':
            used_operations = config.temporal_operations
        elif tag == 'spatial':
            used_operations = config.spatial_operations
        else:
            print('search operations error')

        self._num_ops = len(used_operations)
        self._candidate_ops = nn.ModuleList()
        for op_name in used_operations:
            self._candidate_ops += [create_op(op_name, node_embedding_1, node_embedding_2, adj_mx, config, device)]
        # self._candidate_alphas = nn.Parameter(torch.normal(mean=torch.zeros(self._num_ops), std=1), requires_grad=True)
        self._candidate_alphas = nn.Parameter(torch.zeros(self._num_ops), requires_grad=True)

        # self.start_linear = linear(in_channels, in_channels//self._k)
        # self.end_linear = linear(in_channels//self._k, in_channels)

        self.set_mode(Mode.NONE)

    def set_mode(self, mode):
        self._mode = mode
        if mode == Mode.NONE:
            self._sample_idx = None

        elif mode == Mode.ONE_PATH_FIXED:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            op = torch.argmax(probs).item()
            self._sample_idx = np.array([op], dtype=np.int32)

        elif mode == Mode.ONE_PATH_RANDOM:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            self._sample_idx = torch.multinomial(probs, 1, replacement=True).cpu().numpy()

        elif mode == Mode.TWO_PATHS:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            self._sample_idx = torch.multinomial(probs, 2, replacement=True).cpu().numpy()

        elif mode == Mode.ALL_PATHS:
            self._sample_idx = np.arange(self._num_ops)

    def forward(self, inputs, mask):
        inputs = inputs[0]
        # inputs = self.start_linear(inputs)

        a = self._candidate_alphas[self._sample_idx]
        probs = F.softmax(a, dim=0)
        output = 0
        for i, idx in enumerate(self._sample_idx):
            output += probs[i] * self._candidate_ops[idx](inputs, mask=mask)
        # output = self.end_linear(output)
        return output

    def arch_parameters(self):
        yield self._candidate_alphas

    def weight_parameters(self):
        for i in range(self._num_ops):
            for p in self._candidate_ops[i].parameters():
                yield p
                

class SpatialLayerMixedOp(nn.Module):
    def __init__(self, node_embedding_1, node_embedding_2, adj_mx, config, device, tag):
        super(SpatialLayerMixedOp, self).__init__()

        self.scale_list = config.scale_list
        used_operations = config.spatial_operations

        self._num_ops = len(used_operations)
        self._candidate_ops = nn.ModuleList()
        for op_name in used_operations:
            self._candidate_ops += [create_op(op_name, node_embedding_1, node_embedding_2, adj_mx, config, device)]

    def forward(self, inputs, candidate_alphas,  mask):
        inputs = inputs[0]
        # inputs = self.start_linear(inputs)

        probs = F.softmax(candidate_alphas, dim=0)
        sample_idx = torch.multinomial(probs, 2, replacement=True).cpu().numpy()

        a = candidate_alphas[sample_idx]
        p = F.softmax(a, dim=0)
        output = 0
        for i, idx in enumerate(sample_idx):
            output += p[i] * self._candidate_ops[idx](inputs, mask=mask)
        # output = self.end_linear(output)
        return output

    def weight_parameters(self):
        for i in range(self._num_ops):
            for p in self._candidate_ops[i].parameters():
                yield p

