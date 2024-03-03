import torch
from GNN import GNN
import torch.nn as nn
import dgl
from dgl.nn.pytorch import AvgPooling, MaxPooling, SumPooling, WeightAndSum, Set2Set
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F
import random
from protein import ProTrans


class OutMLP(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(OutMLP, self).__init__()
        self.mlp = nn.Sequential(
            BatchNorm1d(in_feats),
            Linear(in_feats, 128),
            nn.ReLU(),
            BatchNorm1d(128),
            Linear(128, out_feats),
        )

    def forward(self, x):
        x_out = self.mlp(x)
        return x_out


class Mymodel(torch.nn.Module):
    def __init__(self, args):
        super(Mymodel, self).__init__()
        self.args = args
        self.gnn = GNN(args, input_size=39, num_layer=args.layer_num, residual=args.residual)
        self.softmax = torch.nn.Softmax()
        self.hidden_size = args.hidden_size
        self.prot_att_mlp = Linear(self.hidden_size, 1)  # prot attention
        self.node_att_mlp = Linear(self.hidden_size, 1)  # node attention

        self.prot_encoder = ProTrans(22, self.hidden_size, args.max_protein_len, num_attn_layer=args.num_attn_layer,
                                     num_lstm_layer=args.num_lstm_layer, num_attn_heads=args.num_attn_head_prot,
                                     ffn_hidden_size=self.hidden_size, lstm=args.lstm, dropout=args.dropout)

        self.mix_causal_shortcut = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
                                      nn.BatchNorm1d(self.hidden_size * 2),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size * 2, self.hidden_size * 2))

        if self.args.interaction == 'bilinear':
            self.bilinear_global = nn.Bilinear(self.hidden_size, self.hidden_size, self.hidden_size * 2)
            self.bilinear_causal = nn.Bilinear(self.hidden_size, self.hidden_size, self.hidden_size * 2)
            self.bilinear_shortcut = nn.Bilinear(self.hidden_size, self.hidden_size, self.hidden_size * 2)

        if self.args.graph_pool == 'sum':
            self.readout = SumPooling()
        elif self.args.graph_pool == 'mean':
            self.readout = AvgPooling()
        elif self.args.graph_pool == 'max':
            self.readout = MaxPooling()
        elif self.args.graph_pool == 'wsum':
            self.readout = WeightAndSum(self.hidden_size)
        elif self.args.graph_pool == 'set2set':
            self.readout = Set2Set(input_dim=self.hidden_size, n_iters=2, n_layers=1)

        if self.args.causal_inference:
            self.mlp = OutMLP(self.hidden_size * 2, 1)  # global
            self.cmlp = OutMLP(self.hidden_size * 2, 1)  # causal
            self.smlp = OutMLP(self.hidden_size * 2, 1)  # molecule shortcut
            self.intervmlp = OutMLP(self.hidden_size * 2, 1)  # intervention
        else:
            self.mlp = OutMLP(self.hidden_size * 2, 1)

    def random_readout_layer(self, xc, xs):
        num = xc.shape[0]
        l = [i for i in range(num)]
        if self.args.with_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)
        x = torch.cat((xc, xs[random_idx]), dim=1)
        x = self.mix_causal_shortcut(x)
        return x

    def simsiam_loss(self, causal_rep, mix_rep):
        causal_rep = causal_rep.detach()
        causal_rep = F.normalize(causal_rep, dim=1)
        mix_rep = F.normalize(mix_rep, dim=1)
        return -(causal_rep * mix_rep).sum(dim=1).mean()

    def forward(self, graph, protein, prot_mask, training=False):
        node_feat = graph.ndata['x']
        node_feat = self.gnn(graph, node_feat)
        prot_feat = self.prot_encoder(protein, prot_mask)

        drug_h = self.readout(graph, node_feat)
        prot_h = prot_feat.mean(1)

        if self.args.interaction == 'concat':
            h = torch.cat([drug_h, prot_h], dim=-1)
        if self.args.interaction == 'bilinear':
            h = self.bilinear_global(drug_h, prot_h)

        pred = self.mlp(h)

        if training and self.args.causal_inference:
            if self.args.node_attn:
                # sigmoid
                # node_att = torch.sigmoid(self.node_att_mlp(node_feat))
                # xc = node_att * node_feat  # causal node features
                # xs = (1 - node_att) * node_feat  # shortcut node features
                # softmax
                node_att = torch.sigmoid(self.node_att_mlp(node_feat))
                xc = node_att * node_feat
                xs = (1 - node_att) * node_feat
            else:
                xc = node_feat * 0.5
                xs = node_feat * 0.5

            if self.args.prot_attn:
                prot_att = torch.sigmoid(self.prot_att_mlp(prot_feat))
                pc = prot_att.expand_as(prot_feat) * prot_feat
                ps = (1 - prot_att).expand_as(prot_feat) * prot_feat
            else:
                pc = prot_feat * 0.5
                ps = prot_feat * 0.5

            # causal
            graph.ndata.update({'x': xc})
            c_graph_h = self.readout(graph, xc)
            c_prot_h = pc.mean(1)

            # shortcut
            graph.ndata.update({'x': xs})
            s_graph_h = self.readout(graph, xs)
            s_prot_h = ps.mean(1)

            if self.args.interaction == 'concat':
                c_h = torch.cat([c_graph_h, c_prot_h], dim=-1)
                s_h = torch.cat([s_graph_h, s_prot_h], dim=-1)

            if self.args.interaction == 'bilinear':
                c_h = self.bilinear_causal(c_graph_h, c_prot_h)
                s_h = self.bilinear_shortcut(s_graph_h, s_prot_h)

            c_logit = self.cmlp(c_h)
            # s_logit = self.smlp(s_h)

            # backdoor adjustment
            x_interv = self.random_readout_layer(c_h, s_h)
            simsiam_loss = self.simsiam_loss(c_h, x_interv)

            loss_reg = torch.abs(c_h / (c_h + s_h) - self.args.gamma * torch.ones_like(c_h)).mean()
            return c_logit, simsiam_loss, loss_reg
        else:
            return pred
