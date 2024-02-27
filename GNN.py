import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv, GINConv, GATConv, SAGEConv
import torch


class LayerNorm(nn.Module):
    def __init__(self, num_features: int, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GCNLayer(nn.Module):
    def __init__(self, args, in_feats, out_feats, residual=False):
        super(GCNLayer, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = args.layer_norm
        self.residual = residual
        self.args = args
        if args.GNNConv == 'GCN':
            self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats,
                                        allow_zero_in_degree=True, activation=F.relu)
        elif args.GNNConv == 'GIN':
            self.linear = nn.Linear(in_feats, out_feats)
            self.graph_conv = GINConv(self.linear, activation=F.relu)
        elif args.GNNConv == 'GAT':
            self.graph_conv = GATConv(in_feats=in_feats, out_feats=out_feats // args.num_heads,
                                      num_heads=args.num_heads,
                                      allow_zero_in_degree=True, activation=F.relu)
        elif args.GNNConv == 'SAGE':
            self.graph_conv = SAGEConv(in_feats=in_feats, out_feats=out_feats, aggregator_type='gcn',
                                       activation=F.relu)
        if self.layer_norm:
            self.gnn_layer_norm = LayerNorm(num_features=in_feats)

        if self.residual:
            if in_feats == out_feats:
                self.res = nn.Identity()
            else:
                self.res = nn.Linear(in_feats, out_feats)
                nn.init.xavier_normal_(self.res.weight.data, gain=1.414)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        if self.args.GNNConv != 'GIN':
            self.graph_conv.reset_parameters()

    def forward(self, graph: DGLGraph, feat):
        graph = graph.local_var()
        if self.layer_norm:
            h = self.gnn_layer_norm(feat)
        else:
            h = feat

        h = self.graph_conv(graph, h)
        h = h.view(feat.shape[0], -1)

        if self.residual:
            res_feat = self.res(feat)
            h = h + res_feat
        feats = self.dropout(h)
        return feats


class GNN(nn.Module):
    def __init__(self, args, input_size, num_layer, residual=False):
        super(GNN, self).__init__()
        self.args = args
        self.num_layer = num_layer
        self.hidden_feats = args.hidden_size
        self.gnn_layers = nn.ModuleList()
        for i in range(self.num_layer):
            if i == 0:
                in_feats = input_size
            else:
                in_feats = self.hidden_feats
            self.gnn_layers.append(GCNLayer(args, in_feats, out_feats=self.hidden_feats, residual=residual))
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, graph: DGLGraph, feat):
        for layer in range(self.num_layer):
            feat = self.gnn_layers[layer](graph, feat)
        graph.ndata.update({'x': feat})
        return feat
