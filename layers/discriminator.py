import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, args):
        self.args = args
        super(Discriminator, self).__init__()
        self.bilinear_1 = nn.Bilinear(self.args.hidden_nodes, self.args.hidden_nodes, 1)
        if self.args.addVector:
            self.bilinear_2 = nn.Bilinear(self.args.hidden_nodes, self.args.hidden_nodes + self.args.n, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, p, pos, neg, s_bias1=None, s_bias2=None):
        a = torch.zeros(1, pos.shape[1], self.args.hidden_nodes + self.args.n)
        p_x = torch.unsqueeze(p, 1)  # p: summary vector, pos: positive, neg: negative
        if p_x.shape[2] != pos.shape[2]:
            p_x = p_x.expand_as(a)
            sc_pos = torch.squeeze(self.bilinear_2(pos, p_x), 2)
            sc_neg = torch.squeeze(self.bilinear_2(neg, p_x), 2)
        else:
            p_x = p_x.expand_as(pos)
            sc_pos = torch.squeeze(self.bilinear_1(pos, p_x), 2)
            sc_neg = torch.squeeze(self.bilinear_1(neg, p_x), 2)
        if s_bias1 is not None:
            sc_pos += s_bias1
        if s_bias2 is not None:
            sc_neg += s_bias2
        result_cons = torch.cat((sc_pos, sc_neg), 1)
        return result_cons