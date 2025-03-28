import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, drop_prob, Bias=False):
        self.edge_index1 = torch.load('data/edge_index1.pt')
        self.edge_index2 = torch.load('data/edge_index2.pt')
        self.edge_index3 = torch.load('data/edge_index3.pt')
        super(GCN, self).__init__()
        # GCN
        self.conv1 = GCNConv(in_ft, out_ft)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU()
        elif act == 'elu':
            self.act = nn.ELU()
        if Bias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)
        self.drop_prob = drop_prob
        self.Bias = Bias

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, shuf, origin, i, sparse=False):
        if i == 0:
            edge_index = self.edge_index1
        elif i == 1:
            edge_index = self.edge_index2
        elif i == 2:
            edge_index = self.edge_index3
        seq = F.dropout(shuf, self.drop_prob, training=self.training)
        seq = self.conv1(seq, edge_index)
        if sparse:
            seq = torch.unsqueeze(torch.spmm(origin, torch.squeeze(seq, 0)), 0)
        else:
            seq = torch.bmm(origin, seq)
        if self.Bias:
            seq += self.bias_1
        return self.act(seq)