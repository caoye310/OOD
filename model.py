import torch
from GNN import GNN
import torch.nn as nn
import dgl
from dgl.nn.pytorch import AvgPooling, MaxPooling, SumPooling, WeightAndSum, Set2Set
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F
import random
from protein import ProTrans
from torch_scatter import scatter_add


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


class ProtAttn(torch.nn.Module):
    def __init__(self, in_feats):
        super(ProtAttn, self).__init__()
        self.linear1 = nn.Linear(in_feats, in_feats)
        self.bn = nn.BatchNorm1d(in_feats)
        self.out = nn.Sequential(nn.ReLU(),
                                 nn.Linear(in_feats, in_feats),
                                 nn.Sigmoid())  # node attention

    def forward(self, prot_feat):
        out = self.linear1(prot_feat).permute(0, 2, 1)
        out = self.bn(out)
        out = self.out(out.permute(0, 2, 1))
        return out


class Mymodel(torch.nn.Module):
    def __init__(self, args):
        super(Mymodel, self).__init__()
        self.args = args
        self.gnn = GNN(args, input_size=39, num_layer=args.layer_num, residual=args.residual)
        self.softmax = torch.nn.Softmax()
        self.hidden_size = args.hidden_size
        self.prot_att_mlp = ProtAttn(self.hidden_size)  # prot attention
        self.node_att_mlp = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                          nn.BatchNorm1d(self.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_size, self.hidden_size),
                                          nn.Sigmoid())  # node attention

        self.prot_encoder = ProTrans(22, self.hidden_size, args.max_protein_len, num_attn_layer=args.num_attn_layer,
                                     num_lstm_layer=args.num_lstm_layer, num_attn_heads=args.num_attn_head_prot,
                                     ffn_hidden_size=self.hidden_size, lstm=args.lstm, dropout=args.dropout)

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
            if self.args.shortcut_loss == 'MSE':
                self.smlp_mol = OutMLP(self.hidden_size, 1)  # molecule shortcut
                self.smlp_prot = OutMLP(self.hidden_size, 1)  # protein shortcut
            else:
                self.smlp_mol = OutMLP(self.hidden_size, 2)  # molecule shortcut
                self.smlp_prot = OutMLP(self.hidden_size, 2)  # protein shortcut
            feat_size = self.hidden_size * 2
            self.intervmlp = OutMLP(feat_size, 1)  # intervention
        else:
            self.mlp = OutMLP(self.hidden_size * 2, 1)

        self.mix_causal_shortcut = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
                                                 nn.BatchNorm1d(self.hidden_size * 2),
                                                 nn.ReLU(),
                                                 nn.Linear(self.hidden_size * 2, self.hidden_size * 2))

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
                node_att = self.node_att_mlp(node_feat)
                xc = node_att * node_feat
                xs = (1 - node_att) * node_feat
                pos_score = node_att.mean(1)
                pos_score_on_batch_mol = scatter_add(pos_score, graph.ndata['batch'].flatten(), dim=0) + 1e-8  # [B]
                neg_score_on_batch_mol = scatter_add((1 - pos_score), graph.ndata['batch'].flatten(), dim=0) + 1e-8 # [B]
                # printing = node_att[:, 1].view(-1, 1).detach().cpu().numpy()
                # print(printing)
            else:
                xc = node_feat * 0.5
                xs = node_feat * 0.5

            if self.args.prot_attn:
                prot_att = self.prot_att_mlp(prot_feat)
                pc = prot_att * prot_feat
                ps = (1 - prot_att) * prot_feat
                pos_score_on_batch_prot = torch.sum(prot_att.mean(2), dim=1) + 1e-8  # [B]
                neg_score_on_batch_prot = torch.sum((1 - prot_att.mean(2)), dim=1) + 1e-8  # [B]
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
            s_mol_logit = self.smlp_mol(s_graph_h)
            s_prot_logit = self.smlp_prot(s_prot_h)

            # backdoor adjustment
            x_interv = self.random_readout_layer(c_h, s_h)

            # inv_loss = self.simsiam_loss(c_h, x_interv)
            interv_logit = self.intervmlp(x_interv)

            loss_reg_mol = torch.abs(pos_score_on_batch_mol / (
                        pos_score_on_batch_mol + neg_score_on_batch_mol) - self.args.gamma1 * torch.ones_like(
                pos_score_on_batch_mol)).mean()
            loss_reg_prot = torch.abs(pos_score_on_batch_prot / (
                        pos_score_on_batch_prot + neg_score_on_batch_prot) - self.args.gamma2 * torch.ones_like(
                pos_score_on_batch_prot)).mean()

            return pred, c_logit, s_mol_logit, s_prot_logit, interv_logit, loss_reg_mol+loss_reg_prot

        else:
            return pred
