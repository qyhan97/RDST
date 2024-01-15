import torch
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
import copy
import random
import numpy as np
import math
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)


    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_mean_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)

class Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, num_gc_layers, num_classes):
        super(Net,self).__init__()

        self.embedding_dim = hid_feats * num_gc_layers
        self.encoder = Encoder(in_feats, hid_feats, num_gc_layers)
        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)  
        self.fc = th.nn.Linear(self.embedding_dim, num_classes)
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, th.nn.Linear):
                th.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        x, _= self.encoder(data.x, data.edge_index, data.batch)
        x = self.fc(x)
        return x, F.log_softmax(x, dim=-1)

    def unsup_loss(self, data):
        device = data.x.device
        graph_embed, node_embed = self.encoder(data.x, data.edge_index, data.batch)
        global_embed = self.global_d(graph_embed)
        local_embed = self.local_d(node_embed)
        local_global_loss = local_global_loss_(local_embed, global_embed, data.edge_index, data.batch, device)
        return local_global_loss

class FF(th.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = th.nn.Sequential(
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU(),
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU(),
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU()
        )
        self.linear_shortcut = th.nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

def get_positive_expectation(p_samples, average=True):
    log_2 = math.log(2.)
    Ep = log_2 - F.softplus(- p_samples)
    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, average=True):
    log_2 = math.log(2.)

    Eq = F.softplus(-q_samples) + q_samples - log_2

    if average:
        return Eq.mean()
    else:
        return Eq

def local_global_loss_(l_enc, g_enc, edge_index, batch, device):

    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
    neg_mask = torch.ones((num_nodes, num_graphs)).to(device)
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos
