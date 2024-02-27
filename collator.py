from torch.utils.data import Dataset
import torch
import dgl
import pickle


class MoleDataset(Dataset):
    def __init__(self, graph_path, dataset, debug):
        self.graph_path = graph_path
        self.dataset = dataset
        self.debug = debug
        graphs_pos, seqs_pos, prot_mask_pos, labels_pos, graphs_neg, seqs_neg, prot_mask_neg, labels_neg = self.load()
        self.protein_pos = seqs_pos
        self.prot_mask_pos = prot_mask_pos
        self.graphs_pos = graphs_pos
        self.labels_pos = labels_pos

        self.protein_neg = seqs_neg
        self.prot_mask_neg = prot_mask_neg
        self.graphs_neg = graphs_neg
        self.labels_neg = labels_neg

    def __len__(self):
        """ Return the number of graphs. """
        return len(self.labels_pos)

    def __getitem__(self, idx):
        """ Return the whole graphs, subgraphs and label. """
        return self.graphs_pos[idx], self.protein_pos[idx], self.prot_mask_pos[idx], self.labels_pos[idx], \
               self.graphs_neg[idx], self.protein_neg[idx], self.prot_mask_neg[idx], self.labels_neg[idx]

    def load(self):
        """ Load the generated graphs. """
        print(f'Loading processed {self.dataset} data...')
        if self.debug:
            file = self.graph_path + '/' + self.dataset + '_debug.pkl'
        else:
            file = self.graph_path + '/' + self.dataset + '.pkl'
        with open(file, 'rb') as f:
            graphs_pos, labels_pos, seqs_pos, prot_mask_pos, graphs_neg, labels_neg, seqs_neg, prot_mask_neg = \
                pickle.load(f)
        return graphs_pos, seqs_pos, prot_mask_pos, labels_pos, graphs_neg, seqs_neg, prot_mask_neg, labels_neg


class Collator_fn(object):
    def __init__(self):
        pass

    def __call__(self, samples):
        '''
        Generate batched graphs.
        '''
        graphs_pos, protein_pos, prot_mask_pos, labels_pos, graphs_neg, protein_neg, prot_mask_neg, labels_neg = map(
            list, zip(*samples))
        pk_values_pos = torch.LongTensor(labels_pos)
        batched_graph_pos = dgl.batch(graphs_pos)
        batched_prots_pos = torch.LongTensor(torch.stack(protein_pos))
        batched_prots_mask_pos = torch.stack(prot_mask_pos)

        pk_values_neg = torch.LongTensor(labels_neg)
        batched_graph_neg = dgl.batch(graphs_neg)
        batched_prots_neg = torch.LongTensor(torch.stack(protein_neg))
        batched_prots_mask_neg = torch.stack(prot_mask_neg)

        return batched_graph_pos, batched_prots_pos, batched_prots_mask_pos, pk_values_pos, \
               batched_graph_neg, batched_prots_neg, batched_prots_mask_neg, pk_values_neg
