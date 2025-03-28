import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import numpy as np
import math

def count(seq, seq_nor, n, type):
    list1 = []
    if type:
        MAX = torch.max(seq_nor)
        MIN = torch.min(seq_nor)
    else:
        MAX = torch.max(seq)
        MIN = 0
    d = (MAX - MIN) / n
    for i in range(n):
        mask = (seq_nor >= MIN + i * d) & (seq_nor < MIN + (i + 1) * d)
        count = np.sum(torch.masked_select(seq_nor, mask).tolist())
        list1.append(float(count))
    return torch.tensor(list1)

def softmax1(inputMatrix, n):
    outputMatrix = np.mat(np.zeros(n))
    soft_sum = 0
    for idx in range(n):
        outputMatrix[0, idx] = math.log10(inputMatrix[idx])
        soft_sum += outputMatrix[0, idx]
    for idx in range(n):
        outputMatrix[0, idx] = outputMatrix[0, idx]/soft_sum
    return outputMatrix

def softmax(inputMatrix):
    inputMatrix = inputMatrix.squeeze(0)
    inputMatrix = inputMatrix.detach().numpy()
    r, c = np.shape(inputMatrix)
    outputMatrix = np.mat(np.zeros((r, c)))
    for i in range(r):
        soft_sum = 0
        for idx in range(c):
            outputMatrix[i, idx] = pow(math.e, inputMatrix[i, idx])
            soft_sum += outputMatrix[i, idx]
        for idx in range(c):
            outputMatrix[i, idx] = outputMatrix[i, idx]/soft_sum
    outputMatrix = torch.from_numpy(outputMatrix)
    outputMatrix.unsqueeze(0)
    return outputMatrix

class NFGI(nn.Module):
    def __init__(self, args):
        self.args = args
        super(NFGI, self).__init__()

    def forward(self, seq):
        seq_nor = softmax(seq)
        if self.args.addVector:
            return torch.cat([torch.mean(seq, 1), count(seq, seq_nor, self.args.n, 1).unsqueeze(0)], dim=1)
        else:
            return torch.mean(seq, 1)