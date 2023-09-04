
import torch.nn as nn
import torch
from src.model.mode import Mode
from src.model.MixedOpration import TemporalLayerMixedOp, SpatialLayerMixedOp


class TemporalSearchLayer(nn.Module):
    def __init__(self, node_embedding_1, node_embedding_2, adj_mx, config, device, tag=None):
        super(TemporalSearchLayer, self).__init__()
        # create mixed operations
        self.scale_list = config.scale_list

        self._mixed_ops = nn.ModuleList()

        num_search_node = config.num_temporal_search_node
        self._num_mixed_ops = self.get_num_mixed_ops(num_search_node)
        for i in range(self._num_mixed_ops):
            self._mixed_ops += [
                TemporalLayerMixedOp(node_embedding_1, node_embedding_2, adj_mx, config, device, tag)]
        self.set_mode(Mode.NONE)

    def forward(self, x, mask):

        # calculate outputs
        node_idx = 0
        current_output = 0

        node_outputs = [x]
        for i in range(self._num_mixed_ops):
            a = self._mixed_ops[i]
            b = [node_outputs[node_idx]]
            c = a(b, mask)
            current_output += c
            if node_idx + 1 >= len(node_outputs):
                node_outputs += [current_output]
                current_output = 0
                node_idx = 0
            else:
                node_idx += 1

        if node_idx != 0:
            node_outputs += [current_output]

        ret = 0
        for x in node_outputs[:]:
            ret = ret + x
        return ret

    def set_mode(self, mode):
        self._mode = mode
        for op in self._mixed_ops:
            op.set_mode(mode)

    def arch_parameters(self):
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops[i].arch_parameters():
                yield p

    def weight_parameters(self):
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops[i].weight_parameters():
                yield p

    def num_weight_parameters(self):
        count = 0
        for i in range(self._num_mixed_ops):
            count += self._mixed_ops[i].num_weight_parameters()
        return count

    def get_num_mixed_ops(self, num):
        i = 1
        s = 0
        while (i < num):
            s += i
            i += 1
        return s


class SpatialSearchLayer(nn.Module):
    def __init__(self, node_embedding_1, node_embedding_2, adj_mx, config, device, tag=None):
        super(SpatialSearchLayer, self).__init__()
        # create mixed operations
        self.scale_list = config.scale_list
        self.tag = tag

        num_search_node = config.num_temporal_search_node
        self._num_mixed_ops = self.get_num_mixed_ops(num_search_node)

        self._mixed_ops = nn.ModuleList()
        for i in range(self._num_mixed_ops):
            self._mixed_ops += [
                SpatialLayerMixedOp(node_embedding_1, node_embedding_2, adj_mx, config, device, tag)]

        self.spatial_dag = []
        for dag_num in range(len(self.scale_list)):
            dag_ops = []
            for i in range(self._num_mixed_ops):
                dag_ops += [nn.Parameter(torch.zeros(len(config.spatial_operations)), requires_grad=True)]
            self.spatial_dag.append(dag_ops)

        # self.set_mode(Mode.NONE)

    def forward(self, x, dag_i, mask):

        # calculate outputs
        node_idx = 0
        current_output = 0

        node_outputs = [x]
        for i in range(self._num_mixed_ops):
            a = self._mixed_ops[i]
            b = [node_outputs[node_idx]]
            c = a(b, self.spatial_dag[dag_i][i], mask)
            current_output += c
            if node_idx + 1 >= len(node_outputs):
                node_outputs += [current_output]
                current_output = 0
                node_idx = 0
            else:
                node_idx += 1

        if node_idx != 0:
            node_outputs += [current_output]

        ret = 0
        for x in node_outputs[:]:
            ret = ret + x
        return ret

    def set_mode(self, mode):
        self._mode = mode


    def arch_parameters(self):
        for dag in self.spatial_dag:
            for param in dag:
                yield param

    def weight_parameters(self):
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops[i].weight_parameters():
                yield p

    def num_weight_parameters(self):
        count = 0
        for i in range(self._num_mixed_ops):
            count += self._mixed_ops[i].num_weight_parameters()
        return count

    def get_num_mixed_ops(self, num):
        i = 1
        s = 0
        while (i < num):
            s += i
            i += 1
        return s

