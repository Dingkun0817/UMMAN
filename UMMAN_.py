import torch.nn as nn
from initial import initial
from layers import GCN, Discriminator, Attention
import numpy as np
import time
import torch
from validation import validation

class UMMAN(initial):
    def __init__(self, args):
        initial.__init__(self, args)
        self.args = args

    def training(self):
        features = [feature.to(self.args.device) for feature in self.features]
        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        model = modeler(self.args).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        times = 0
        best_loss = 1e7
        b_xent = nn.BCEWithLogitsLoss()
        for epoch in range(self.args.epochs):
            xent_loss = None
            model.train()
            optimiser.zero_grad()
            idx = np.random.permutation(self.args.nb_nodes)

            shuf = [feature[:, idx, :] for feature in features]
            shuf = [shuf_ft.to(self.args.device) for shuf_ft in shuf]

            lbl_pos = torch.ones(self.args.batch_size, self.args.nb_nodes)
            lbl_neg = torch.zeros(self.args.batch_size, self.args.nb_nodes)
            lbl_con = torch.cat((lbl_pos, lbl_neg), 1).to(self.args.device)

            result = model(features, adj, shuf, self.args.sparse, None, None, None)
            logits = result['logits']

            for view_idx, logit in enumerate(logits):
                if xent_loss is None:
                    xent_loss = b_xent(logit, lbl_con)
                else:
                    xent_loss += b_xent(logit, lbl_con)

            loss = xent_loss

            attn_loss = result['attn_loss']
            loss += self.args.attn_coef * attn_loss

            if loss < best_loss:
                best_loss = loss
                times = 0
                torch.save(model.state_dict(),
                           'saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder,
                                                                  self.args.relationships))
            else:
                times += 1

            if times == self.args.limit:
                break

            loss.backward()
            optimiser.step()

        model.load_state_dict(torch.load(
            'saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, self.args.relationships)))

        # Evaluation
        model.eval()
        acc, acc_std, precision, precision_std, recall, recall_std, AUC, AUC_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std = validation(
            model.P.data.detach(), self.idx_train, self.idx_test, self.labels, self.args.device)
        return acc, acc_std, precision, precision_std, recall, recall_std, AUC, AUC_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        self.gcn = nn.ModuleList(
            [GCN(args.ft_size, args.hid_units, args.activation, args.drop_prob, args.isBias) for _ in
             range(args.nb_graphs)])

        self.disc = Discriminator(args)
        self.P = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hid_units))
        self.readout_func = self.args.readout_func
        if args.Attn:
            self.attn = nn.ModuleList([Attention(args) for _ in range(args.nheads)])

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.P)

    def forward(self, feature, adj, shuf, sparse, msk, samp_bias1, samp_bias2):
        pos_all = []
        neg_all = []
        p_all = []
        logits = []
        result = {}
        for i in range(self.args.nb_graphs):
            pos = self.gcn[i](feature[i], adj[i], i, sparse)

            # get positive summary vector
            p = self.readout_func(pos)
            p = self.args.readout_act_func(p)
            neg = self.gcn[i](shuf[i], adj[i], i, sparse)
            logit = self.disc(p, pos, neg, samp_bias1, samp_bias2)

            pos_all.append(pos)
            neg_all.append(neg)
            p_all.append(p)
            logits.append(logit)
        result['logits'] = logits

        # Attention or not
        if self.args.Attn:
            attn_pos_all_lst = []
            attn_neg_all_lst = []

            for h_idx in range(self.args.nheads):
                attn_pos_all_, attn_neg_all_= self.attn[h_idx](pos_all, neg_all)
                attn_pos_all_lst.append(attn_pos_all_)
                attn_neg_all_lst.append(attn_neg_all_)

            attn_pos_all = torch.mean(torch.cat(attn_pos_all_lst, 0), 0).unsqueeze(0)
            attn_neg_all = torch.mean(torch.cat(attn_neg_all_lst, 0), 0).unsqueeze(0)

        else:
            attn_pos_all = torch.mean(torch.cat(pos_all), 0).unsqueeze(0)
            attn_neg_all = torch.mean(torch.cat(neg_all), 0).unsqueeze(0)

        pos_attn_loss = ((self.P - attn_pos_all) ** 2).sum()
        neg_attn_loss = ((self.P - attn_neg_all) ** 2).sum()
        attn_loss = pos_attn_loss - neg_attn_loss
        result['attn_loss'] = attn_loss

        return result

class NFGI(nn.Module):
    def __init__(self, args):
        self.args = args
        super(NFGI, self).__init__()

    def forward(self, seq):
        # print(seq)
        seq_nor = softmax(seq)
        if self.args.addVector:
            return torch.cat([torch.mean(seq, 1), count(seq, seq_nor, self.args.n, 1).unsqueeze(0)], dim=1)
        else:
            return torch.mean(seq, 1)

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
        p_x = torch.unsqueeze(p, 1)
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