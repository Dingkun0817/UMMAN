import torch.nn as nn
import torch
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args
        self.A = nn.ModuleList([nn.Linear(args.hidden_nodes, 1) for _ in range(args.graph_num)])
        if self.args.addVector:
            self.B = nn.ModuleList([nn.Linear(args.hidden_nodes + args.n, 1) for _ in range(args.graph_num)])
        self.weight_init()

    def weight_init(self):
        for i in range(self.args.graph_num):
            nn.init.xavier_normal_(self.A[i].weight)
            self.A[i].bias.data.fill_(0.0)

    def forward(self, feat_pos, feat_neg):
        feat_pos, feat_pos_attn = self.attn_feature(feat_pos)
        feat_neg, feat_neg_attn = self.attn_feature(feat_neg)
        return feat_pos, feat_neg

    def attn_feature(self, features):
        features_attn = []
        for i in range(self.args.graph_num):
            features_attn.append((self.A[i](features[i].squeeze())))
        features_attn = F.softmax(torch.cat(features_attn, 1), -1)
        features = torch.cat(features,1).squeeze(0)
        features_attn_reshaped = features_attn.transpose(1, 0).contiguous().view(-1, 1)
        features = features * features_attn_reshaped.expand_as(features)
        features = features.view(self.args.graph_num, self.args.nb_nodes, self.args.hidden_nodes)
        return features, features_attn