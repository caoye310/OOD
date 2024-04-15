from torch.utils.data import Dataset
import torch
import dgl
import pickle


class MoleDataset(Dataset):
    def __init__(self, graph_path, dataset, debug):
        self.graph_path = graph_path
        self.dataset = dataset
        self.debug = debug
        graphs, seq, prot_mask, labels = self.load()
        self.protein = seq
        self.graphs = graphs
        self.prot_mask = prot_mask
        self.labels = labels

    def __len__(self):
        """ Return the number of graphs. """
        return len(self.labels)

    def __getitem__(self, idx):
        """ Return the whole graphs, subgraphs and label. """
        return self.graphs[idx], self.protein[idx], self.prot_mask[idx], self.labels[idx]

    def load(self):
        """ Load the generated graphs. """
        print(f'Loading processed {self.dataset} data...')
        if self.debug:
            file = self.graph_path + '/' + self.dataset + '_debug.pkl'
        else:
            file = self.graph_path + '/' + self.dataset + '.pkl'
        with open(file, 'rb') as f:
            graphs, seq, prot_mask, labels = pickle.load(f)
        return graphs, seq, prot_mask, labels


class Collator_fn(object):
    def __init__(self):
        pass

    def __call__(self, samples):
        '''
        Generate batched graphs.
        '''
        graphs, protein, prot_mask, labels = map(list, zip(*samples))
        for i in range(len(graphs)):
            node_graph_id = torch.LongTensor([i for _ in range(graphs[i].num_nodes())]).view(-1, 1)
            graphs[i].ndata['batch'] = node_graph_id
        pk_values = torch.LongTensor(labels)
        batched_graph = dgl.batch(graphs)
        batched_prots = torch.LongTensor(torch.stack(protein))
        batched_prots_mask = torch.stack(prot_mask)
        return batched_graph, batched_prots, batched_prots_mask, pk_values
