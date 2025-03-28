import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from initial import initial
from layers import GCN, Discriminator, Attention
import numpy as np
import torch as tr
np.random.seed(0)
from validation import validation

class UMMAN(initial):
    def __init__(self, args):
        initial.__init__(self, args)
        self.args = args

    def training(self):
        features = [feature.to(self.args.device) for feature in self.features]
        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        model = modeler(self.args).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.learningrate, weight_decay=self.args.l2_coef)
        times = 0
        best = 1e7
        BCE = nn.BCEWithLogitsLoss()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        Train_loss = []
        Train_acc = []
        Test_loss = []
        Test_acc = []
        for epoch in range(self.args.epochs):
            adv_loss = None
            model.train()
            optimiser.zero_grad()
            idx = np.random.permutation(self.args.nb_nodes)
            shuf = [feature[:, idx, :] for feature in features]
            shuf = [shuf_ft.to(self.args.device) for shuf_ft in shuf]
            lbl_pos = torch.ones(self.args.batch_size, self.args.nb_nodes)
            lbl_neg = torch.zeros(self.args.batch_size, self.args.nb_nodes)
            lbl_con = torch.cat((lbl_pos, lbl_neg), 1).to(self.args.device)
            result = model(features, adj, shuf, self.args.sparse, None, None, None)
            result_cons = result['result_cons']
            for view_idx, result_con in enumerate(result_cons):
                if adv_loss is None:
                    adv_loss = BCE(result_con, lbl_con)
                else:
                    adv_loss += BCE(result_con, lbl_con)
            loss = adv_loss
            attn_loss = result['attn_loss']
            loss += self.args.attn_coef * attn_loss
            if loss < best:
                best = loss
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
            model.eval()
        model.load_state_dict(torch.load(
            'saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, self.args.relationships)))

        model.eval()
        _, acc, acc_std, precision, precision_std, recall, recall_std, AUC, AUC_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std = validation(
            model.P.data.detach(), self.idx_train, self.idx_test, self.labels, self.args.device)
        return Train_loss, Train_acc, Test_loss, Test_acc, acc, acc_std, precision, precision_std, recall, recall_std, AUC, AUC_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        self.gcn = nn.ModuleList(
            [GCN(args.ft_size, args.hidden_nodes, args.activation, args.drop_prob, args.Bias) for _ in
             range(args.graph_num)])
        self.disc = Discriminator(args)
        self.P = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hidden_nodes))
        self.NFGI_func = self.args.NFGI_func
        if args.Attn:
            self.attn = nn.ModuleList([Attention(args) for _ in range(args.head_num)])
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.P)

    def forward(self, feature, origin, shuf, sparse, msk, samp_bias1, samp_bias2):
        pos_all = []
        neg_all = []
        p_all = []
        result_cons = []
        result = {}
        for i in range(self.args.graph_num):
            pos = self.gcn[i](feature[i], origin[i], i, sparse)
            # get positive summary vector
            p = self.NFGI_func(pos)
            p = self.args.NFGI_act(p)
            neg = self.gcn[i](shuf[i], origin[i], i, sparse)
            result_con = self.disc(p, pos, neg, samp_bias1, samp_bias2)
            pos_all.append(pos)
            neg_all.append(neg)
            p_all.append(p)
            result_cons.append(result_con)
        result['result_cons'] = result_cons

        # Attention
        if self.args.Attn:
            attn_pos_all_lst = []
            attn_neg_all_lst = []
            for h_idx in range(self.args.head_num):
                attn_pos_all_, attn_neg_all_= self.attn[h_idx](pos_all, neg_all)
                attn_pos_all_lst.append(attn_pos_all_)
                attn_neg_all_lst.append(attn_neg_all_)
            attn_pos_all = torch.mean(torch.cat(attn_pos_all_lst, 0), 0).unsqueeze(0)
            attn_neg_all = torch.mean(torch.cat(attn_neg_all_lst, 0), 0).unsqueeze(0)
        else:
            attn_pos_all = torch.mean(torch.cat(pos_all), 0).unsqueeze(0)
            attn_neg_all = torch.mean(torch.cat(neg_all), 0).unsqueeze(0)
        # attn_loss
        pos_attn_loss = ((self.P - attn_pos_all) ** 2).sum()
        neg_attn_loss = ((self.P - attn_neg_all) ** 2).sum()
        attn_loss = pos_attn_loss - neg_attn_loss
        attn_loss = pos_attn_loss
        result['attn_loss'] = attn_loss
        return result

def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * tr.log(input_ + epsilon)
    entropy = tr.sum(entropy, dim=1)
    return entropy

class ClassConfusionLoss(nn.Module):
    """
    The class confusion loss

    Parameters:
        - **t** Optional(float): the temperature factor used in MCC
    """

    def __init__(self, t):
        super(ClassConfusionLoss, self).__init__()
        self.t = t

    def forward(self, output: tr.Tensor) -> tr.Tensor:
        # output:  torch.Size([32, 2])
        n_sample, n_class = output.shape
        softmax_out = nn.Softmax(dim=1)(output / self.t)
        entropy_weight = Entropy(softmax_out).detach()
        entropy_weight = 1 + tr.exp(-entropy_weight)
        entropy_weight = (n_sample * entropy_weight / tr.sum(entropy_weight)).unsqueeze(dim=1)
        class_confusion_matrix = tr.mm((softmax_out * entropy_weight).transpose(1, 0), softmax_out)
        class_confusion_matrix = class_confusion_matrix / tr.sum(class_confusion_matrix, dim=1)
        mcc_loss = (tr.sum(class_confusion_matrix) - tr.trace(class_confusion_matrix)) / n_class
        return mcc_loss

