import numpy as np
import torch
from utils import process
import torch.nn as nn
from layers import NFGI

class initial:
    def __init__(self, args):
        args.batch_size = 1
        args.sparse = True
        args.relationships_list = args.relationships.split(",")
        args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        adj, features, labels, idx_train, idx_test = process.load_data(args)
        features = [process.preprocess_features(feature) for feature in features]
        args.nb_nodes = features[0].shape[0]
        args.ft_size = features[0].shape[1]
        args.nb_classes = labels.shape[1]
        args.graph_num = len(adj)
        args.adj = adj
        adj = [process.normalize_adj(adj_) for adj_ in adj]
        self.adj = [process.sparse_mx_to_torch_sparse_tensor(adj_) for adj_ in adj]
        self.features = [torch.FloatTensor(feature[np.newaxis]) for feature in features]
        self.labels = torch.FloatTensor(labels[np.newaxis]).to(args.device)
        self.idx_train = torch.LongTensor(idx_train).to(args.device)
        self.idx_test = torch.LongTensor(idx_test).to(args.device)
        self.train_lbls = torch.argmax(self.labels[0, self.idx_train], dim=1)
        self.test_lbls = torch.argmax(self.labels[0, self.idx_test], dim=1)
        # Aggregate
        args.NFGI_func = NFGI(args)
        # Summary aggregation
        args.NFGI_act = nn.Sigmoid()
        self.args = args