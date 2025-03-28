import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from models import Linear
import torch.nn as nn
from sklearn.metrics import *
import numpy as np
import torch as tr

def validation(embeds, idx_train, idx_test, labels, device, isTest=True):
    hidden_nodes = embeds.shape[2]
    nb_classes = labels.shape[2]
    xent = nn.CrossEntropyLoss()
    ccl = ClassConfusionLoss(1)
    train_embs = embeds[0, idx_train]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    softmax = nn.Softmax(dim=1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_test = []  ##
    precisions = []
    recalls = []
    AUCs = []
    fpr = []
    tpr = []

    for i in range(50):
        log = Linear(hidden_nodes, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.to(device)

        test_accs_1 = []
        test_accs = []
        test_micro_f1s_1 = []
        test_micro_f1s = []
        test_macro_f1s_1 = []
        test_macro_f1s = []
        test_precisions_1 = []
        test_precisions = []
        test_recalls_1 = []
        test_recalls = []
        test_AUCs_1 = []
        test_AUCs = []

        Train_loss = []
        Train_acc = []
        Test_loss = []
        Test_acc = []

        for iter_ in range(50):
            # train
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls) + ccl(logits)
            train_preds = torch.argmax(logits, dim=1)
            train_acc = torch.sum(train_preds == train_lbls).float() / train_lbls.shape[0]
            loss.backward()
            opt.step()
            logits = log(test_embs)
            pred_scores = softmax(logits)
            pred_scores = pred_scores[:, 1]
            preds = torch.argmax(logits, dim=1)
            pred_scores = pred_scores.detach().numpy()
            test_acc_1 = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro_1 = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro_1 = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')
            test_precision_1 = precision_score(test_lbls.cpu(), preds.cpu())
            test_recall_1 = recall_score(test_lbls.cpu(), preds.cpu())
            test_AUC_1 = roc_auc_score(test_lbls.cpu(), pred_scores)
            test_accs_1.append(test_acc_1.item())
            test_macro_f1s_1.append(test_f1_macro_1)
            test_micro_f1s_1.append(test_f1_micro_1)
            test_precisions_1.append(test_precision_1)
            test_recalls_1.append(test_recall_1)
            test_AUCs_1.append(test_AUC_1)
            # test
            logits = log(test_embs)
            test_loss = xent(logits, test_lbls)
            pred_scores = softmax(logits)
            pred_scores = pred_scores[:, 1]
            preds = torch.argmax(logits, dim=1)
            pred_scores = pred_scores.detach().numpy()
            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_precision = precision_score(test_lbls.cpu(), preds.cpu())
            test_recall = recall_score(test_lbls.cpu(), preds.cpu())
            test_f1_micro = 2 * test_recall * test_precision / (test_recall + test_precision)
            test_AUC = roc_auc_score(test_lbls.cpu(), pred_scores)
            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            test_precisions.append(test_precision)
            test_recalls.append(test_recall)
            test_AUCs.append(test_AUC)
            Train_loss.append(loss)
            Train_acc.append(train_acc.item())
            Test_loss.append(test_loss)
            Test_acc.append(test_acc.item())
        max_iter = test_accs_1.index(max(test_accs_1))
        accs.append(test_accs[max_iter])
        max_iter = test_macro_f1s_1.index(max(test_macro_f1s_1))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_test.append(test_macro_f1s_1[max_iter])
        max_iter = test_micro_f1s_1.index(max(test_micro_f1s_1))
        micro_f1s.append(test_micro_f1s[max_iter])
        max_iter = test_precisions_1.index(max(test_precisions_1))
        precisions.append(test_precisions[max_iter])
        max_iter = test_recalls_1.index(max(test_recalls_1))
        recalls.append(test_recalls[max_iter])
        max_iter = test_AUCs_1.index(max(test_AUCs_1))
        AUCs.append(test_AUCs[max_iter])
    if isTest:
        print("\tAcc:{:.4f} ({:.4f})".format(np.mean(accs), np.std(accs)))
        print("\tPrecision:{:.4f} ({:.4f})".format(np.mean(precisions), np.std(precisions)))
        print("\tRecall:{:.4f} ({:.4f})".format(np.mean(recalls), np.std(recalls)))
        print("\tAUC:{:.4f} ({:.4f})".format(np.mean(AUCs), np.std(AUCs)))
        print("\tF1: {:.4f} ({:.4f})".format(np.mean(micro_f1s),np.std(micro_f1s)))
    else:
        pass
    return (Train_loss[49], Train_acc[49], Test_loss[49], Test_acc[49]), np.mean(accs), np.std(accs), np.mean(precisions)\
        , np.std(precisions), np.mean(recalls), np.std(recalls), np.mean(AUCs), np.std(AUCs), np.mean(macro_f1s),\
           np.std(macro_f1s), np.mean(micro_f1s), np.std(micro_f1s)

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
        n_sample, n_class = output.shape
        softmax_out = nn.Softmax(dim=1)(output / self.t)
        entropy_weight = Entropy(softmax_out).detach()
        entropy_weight = 1 + tr.exp(-entropy_weight)
        entropy_weight = (n_sample * entropy_weight / tr.sum(entropy_weight)).unsqueeze(dim=1)
        class_confusion_matrix = tr.mm((softmax_out * entropy_weight).transpose(1, 0), softmax_out)
        class_confusion_matrix = class_confusion_matrix / tr.sum(class_confusion_matrix, dim=1)
        mcc_loss = (tr.sum(class_confusion_matrix) - tr.trace(class_confusion_matrix)) / n_class
        return mcc_loss