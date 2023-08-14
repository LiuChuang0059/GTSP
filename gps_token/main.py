import os.path as osp
import os

import torch
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential

import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, GINConv
from gps_conv import GPSConv
from torch_geometric.utils import degree
from torch.utils.data import random_split


import random
import numpy as np
import configargparse

import torch.nn.functional as F

from utils import results_to_file



from datetime import datetime
now = datetime.now()
now = now.strftime("%m_%d-%H_%M_%S")



#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'zinc-pe')
transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
# train_dataset = ZINC(path, subset=True, split='train', pre_transform=transform)
# val_dataset = ZINC(path, subset=True, split='val', pre_transform=transform)
# test_dataset = ZINC(path, subset=True, split='test', pre_transform=transform)

def gene_arg():

    parser = configargparse.ArgumentParser(allow_abbrev=False,
                                    description='GNN baselines on ogbg-code data with Pytorch Geometrics')
    parser.add_argument('--configs', required=False, is_config_file=True)
    parser.add_argument('--wandb_run_idx', type=str, default=None)


    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='dataset name (default: ogbg-code)')

    parser.add_argument('--aug', type=str, default='baseline',
                        help='augment method to use [baseline|flag|augment]')

    parser.add_argument('--max_seq_len', type=int, default=None,
                        help='maximum sequence length to predict (default: None)')

    group = parser.add_argument_group('model')
    group.add_argument('--model_type', type=str, default='gnn', help='gnn|pna|gnn-transformer')
    group.add_argument('--graph_pooling', type=str, default='mean')
    group = parser.add_argument_group('gnn')
    group.add_argument('--gnn_type', type=str, default='gcn')
    group.add_argument('--gnn_virtual_node', action='store_true')
    group.add_argument('--gnn_dropout', type=float, default=0)
    group.add_argument('--gnn_num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    group.add_argument('--gnn_emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    group.add_argument('--gnn_JK', type=str, default='last')
    group.add_argument('--gnn_residual', action='store_true', default=False)
    group.add_argument('--num_layers', type=int, default=4,
                        help='number of GNN message passing layers (default: 5)')
    group.add_argument('--nhead', type=int, default=4,
                        help='number of GNN message passing layers (default: 5)')


    group = parser.add_argument_group('training')
    group.add_argument('--devices', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    group.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    group.add_argument('--eval_batch_size', type=int, default=None,
                        help='input batch size for training (default: train batch size)')
    group.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    group.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    group.add_argument('--scheduler', type=str, default=None)
    group.add_argument('--pct_start', type=float, default=0.3)
    group.add_argument('--weight_decay', type=float, default=0.0)
    group.add_argument('--grad_clip', type=float, default=None)
    group.add_argument('--lr', type=float, default=0.001)
    group.add_argument('--max_lr', type=float, default=0.001)
    group.add_argument('--runs', type=int, default=10)
    group.add_argument('--test-freq', type=int, default=1)
    group.add_argument('--start-eval', type=int, default=15)
    group.add_argument('--resume', type=str, default=None)
    group.add_argument('--seed', type=int, default=12344)
    group.add_argument('--token_ratio', type=float, default=0.5)

    # fmt: on

    args, _ = parser.parse_known_args()

    return args

args = gene_arg()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
            # cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def load_data(args, transform):

    data_name = args.dataset + '-pe'
    dataset = TUDataset(os.path.join(args.data_root, data_name),
                        name=args.dataset,
                        pre_transform=transform
                        )
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    num_tasks = dataset.num_classes
    num_features = dataset.num_features
    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(validation_set, batch_size=args.eval_batch_size)
    test_loader  = DataLoader(test_set, batch_size=args.eval_batch_size)

    return train_loader, val_loader, test_loader, num_tasks, num_features

train_loader, val_loader, test_loader, num_tasks, num_features = load_data(args, transform)

#raise Exception("pause!")



class GPS(torch.nn.Module):
    def __init__(self, fea_dim, channels: int, num_layers: int, num_tasks, args):
        super().__init__()

        self.node_emb = Linear(fea_dim, channels)
        self.pe_lin = Linear(20, channels)
        self.edge_emb = Embedding(4, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINConv(nn), heads=4,
                                                  attn_dropout=0.5,
                                                  token_ratio = args.token_ratio)
            self.convs.append(conv)

        self.lin = Linear(channels, num_tasks)

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x = self.node_emb(x) + self.pe_lin(pe)
        #edge_attr = self.edge_emb(edge_attr)
        edge_attr = None

        for conv in self.convs:
            x = conv(x, edge_index, batch)
        x = global_add_pool(x, batch)
        return self.lin(x)


device = torch.device('cuda:{}'.format(args.devices) if torch.cuda.is_available() else 'cpu')
model = GPS(fea_dim=num_features, channels=64, num_layers=args.num_layers, num_tasks = num_tasks, args =args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


def train(epoch):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        #print(data.x.size(), data.pe.size())
        out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                    data.batch)
        loss = F.cross_entropy(out, data.y)
        #loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()

        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                    data.batch)
        out = out.max(dim=1)[1]
        correct += out.eq(data.y).sum().item()
        #total_error += (out.squeeze() - data.y).abs().sum().item()
    return correct / len(loader.dataset)


run_name = f"{args.dataset}"
args.save_path = f"exps/{run_name}-{now}"
os.makedirs(os.path.join(args.save_path, str(args.seed)), exist_ok=True)


best_val, final_test = 0, 0
for epoch in range(1, args.epochs+1):
    loss = train(epoch)
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}

    if best_val < val_acc:
        best_val = val_acc
        #final_test = test_acc
        torch.save(state_dict, os.path.join(args.save_path, str(args.seed), "best_model.pt"))

    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')


## load
state_dict = torch.load(os.path.join(args.save_path, str(args.seed), "best_model.pt"))
model.load_state_dict(state_dict["model"])
best_val_acc = test(val_loader)
best_test_acc = test(test_loader)


results_to_file(args, best_test_acc, best_val_acc)
