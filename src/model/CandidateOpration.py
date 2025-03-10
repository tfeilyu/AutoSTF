import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from src.model.transformer import LinearFormerLayer, LinearFormer, LearnedPositionalEncoding


def create_op(op_name, node_embedding_1, node_embedding_2, adj_mx, config, device):
    name2op = {
        'Zero': lambda: Zero(),
        'Identity': lambda: Identity(),

        'Informer': lambda: InformerLayer(),
        'DCC_2': lambda: DCCLayer(config, dilation=2),

        'GNN_fixed': lambda: GNN_fixed(config, adj_mx, device),
        'GNN_adap': lambda: GNN_adap(config, node_embedding_1, node_embedding_2),
        'GNN_att': lambda: GNN_att(config),
    }
    op = name2op[op_name]()
    return op


def gconv(x, A):
    A = A.to(torch.float32)
    x = torch.einsum('bnh,nn->bnh', (x, A))
    return x.contiguous()


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, inputs, **kwargs):
        return inputs.mul(0.)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs, **kwargs):
        return inputs


class CausalConv2d(nn.Conv2d):
    """
    单向padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self._padding = (kernel_size[-1] - 1) * dilation
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=(0, self._padding), dilation=dilation, groups=groups, bias=bias)

    def forward(self, inputs):
        result = super(CausalConv2d, self).forward(inputs)
        if self._padding != 0:
            return result[:, :, :, :-self._padding]
        return result


class DCCLayer(nn.Module):
    """
    dilated causal convolution layer with GLU function
    暂时用GTU代替
    """
    def __init__(self, config, kernel_size=(1, 2), stride=1, dilation=1):
        super(DCCLayer, self).__init__()
        c_in = config.hidden_channels
        c_out = config.hidden_channels
        self.relu = nn.ReLU()
        self.filter_conv = CausalConv2d(c_in, c_out, kernel_size, stride, dilation=dilation)
        self.gate_conv = CausalConv2d(c_in, c_out, kernel_size, stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(c_out, affine=False)

    def forward(self, x,  **kwargs):
        """
        :param x: [batch_size, f_in, N, T]这些block的input必须具有相同的shape？
        :return:
        """
        x = self.relu(x)
        filter = torch.tanh(self.filter_conv(x))
        # filter = self.filter_conv(x)
        gate = torch.sigmoid(self.gate_conv(x))
        output = filter * gate
        output = self.bn(output)
        # output = F.dropout(output, 0.5, training=self.training)

        return output


class GNN_fixed(nn.Module):
    """
    K-order diffusion convolution layer with self-adaptive adjacency matrix (N, N)
    """
    def __init__(self, config, adj_mx, device, use_bn=True, dropout=0.15):
        super(GNN_fixed, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.supports = []
        self.num_hop = config.num_hop

        adj_mx_dcrnn = adj_mx[0]

        supports = [adj_mx_dcrnn]
        # supports.append(calculate_random_walk_matrix(adj_mx_dcrnn.cpu()))
        # supports.append(calculate_random_walk_matrix(adj_mx_dcrnn.T.cpu()))
        for support in supports:
            self.supports.append(torch.tensor(support).to(device))

        self.linear = nn.Conv2d(config.hidden_channels * (len(self.supports)*self.num_hop+1), config.hidden_channels,
                                kernel_size=(1, 1), stride=(1, 1))
        if use_bn:
            self.bn = nn.BatchNorm2d(config.hidden_channels)

    def forward(self, inputs, **kwargs):
        x = torch.relu(inputs)

        outputs = [x]

        for support in self.supports:
            for j in range(self.num_hop):
                x = gconv(x, support)
                outputs += [x]

        h = torch.cat(outputs, dim=2)
        h = h.unsqueeze(dim=1)
        h = h.transpose(1, 3)

        h = self.linear(h)
        if self.use_bn:
            h = self.bn(h)
        if self.dropout > 0:
            h = F.dropout(h, self.dropout, training=self.training)

        h = h.transpose(1, 3)
        h = h.squeeze(dim=1)

        return h



class GNN_adap(nn.Module):
    def __init__(self, config, node_vec1, node_vec2, use_bn=True, dropout=0.15):
        super(GNN_adap, self).__init__()

        self.use_bn = use_bn
        self.dropout = dropout
        self.num_hop = config.num_hop

        self.node_vec1 = node_vec1
        self.node_vec2 = node_vec2

        self.linear = nn.Conv2d(config.hidden_channels*(self.num_hop+1), config.hidden_channels, kernel_size=(1, 1), stride=(1, 1))
        if use_bn:
            self.bn = nn.BatchNorm2d(config.hidden_channels)

    def forward(self, inputs, **kwargs):
        x = torch.relu(inputs)

        # adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # bug?
        adp = F.relu(torch.mm(self.node_vec1, self.node_vec2))
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)

        outputs = [x]
        for _ in range(self.num_hop):
            x = gconv(x, adp)
            outputs += [x]

        h = torch.cat(outputs, dim=2)
        h = h.unsqueeze(dim=1)
        h = h.transpose(1, 3)

        h = self.linear(h)
        if self.use_bn:
            h = self.bn(h)
        if self.dropout > 0:
            h = F.dropout(h, self.dropout, training=self.training)

        h = h.transpose(1, 3)
        h = h.squeeze(dim=1)

        return h


class GNN_att(nn.Module):
    def __init__(self, config):
        super(GNN_att, self).__init__()

        self.heads = 8
        self.layers = config.num_att_layers
        self.hid_dim = config.hidden_channels

        self.attention_layer = LinearFormerLayer(self.hid_dim, self.heads, self.hid_dim * 4)
        self.attention_norm = nn.LayerNorm(self.hid_dim)
        self.attention = LinearFormer(self.attention_layer, self.layers, self.attention_norm)
        self.lpos = LearnedPositionalEncoding(self.hid_dim, max_len=config.num_sensors)

    def forward(self, input, mask):
        # print('hid_dim: ', self.hid_dim)
        x = input.permute(1, 0, 2)
        x = self.lpos(x)
        output = self.attention(x, mask)
        output = output.permute(1, 0, 2)
        return output


######################################################################
# Informer
######################################################################
class InformerLayer(nn.Module):
    def __init__(self, d_model=32, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(InformerLayer, self).__init__()
        # d_ff = d_ff or 4*d_model
        self.attention = AttentionLayer(
            ProbAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.pe = PositionalEmbedding(d_model)
        self.d_model = d_model

    def forward(self, x, attn_mask=None, **kwargs):
        # x = x[0]
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [64, 207, 12, 32]
        x = x.reshape(-1, T, C)  # [64*207, 12, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        # x = x * math.sqrt(self.d_model)
        # x = x + self.pe(x)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, T, C)
        output = output.permute(0, 3, 1, 2)

        return output


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=3, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        :param Q: [b, heads, T, d_k]
        :param K: 采样的K? 长度为Ln(L_K)?
        :param sample_k: c*ln(L_k), set c=3 for now
        :param n_top: top_u queries?
        :return: Q_K and Top_k query index
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

        # kernel_size = 3
        # pad = (kernel_size - 1) // 2
        # self.query_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        # self.key_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        # self.value_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # queries = queries.transpose(-1, 1)
        # keys = keys.transpose(-1, 1)
        # values = values.transpose(-1, 1)
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

