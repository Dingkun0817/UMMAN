import torch
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import argparse
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='UMMAN')
    parser.add_argument('--embedder', nargs='?', default='UMMAN')
    parser.add_argument('--dataset', nargs='?', default='data')
    parser.add_argument('--relationships', nargs='?', default='euclidean,braycurtis,correlation')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden_nodes', type=int, default=256)
    parser.add_argument('--learningrate', type=float, default=0.0015)
    parser.add_argument('--l2_coef', type=float, default=0.000015)
    parser.add_argument('--drop_prob', type=float, default=0.8)
    parser.add_argument('--attn_coef', type=float, default=0.005)
    parser.add_argument('--self_conv', type=float, default=600)
    parser.add_argument('--limit', type=int, default=490)
    parser.add_argument('--head_num', type=int, default=1)
    parser.add_argument('--activation', nargs='?', default='relu')
    parser.add_argument('--Bias', action='store_true', default=True)
    parser.add_argument('--Attn', action='store_true', default=True)
    parser.add_argument('--addVector', action='store_true', default=True)
    parser.add_argument('--n', type=int, default=10)
    return parser.parse_known_args()

def main():
    args, unknown = parse_args()
    from models import UMMAN
    embedder = UMMAN(args)
    data = embedder.features[0].numpy().tolist()[0]
    embedder.training()
if __name__ == '__main__':
    main()
