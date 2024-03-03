import os
import pandas as pd
from torch_geometric.data import InMemoryDataset
import torch
from tqdm import tqdm
import numpy as np
import pickle
import dgl
import rdkit
from rdkit import Chem
import json
# protein
seq_voc = "ABCDEFGHIKLMNPQRSTVWXY"
seq_dict = {v: i + 1 for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)


class DrugOOD(InMemoryDataset):

    def __init__(self, args):
        """
        Load data from .json
        """
        super(DrugOOD, self).__init__()
        self.save_path = args.save_graph
        self.dataset = args.dataset
        self.root = args.root
        self.max_protein_len = args.max_protein_len
        self.debug = args.DEBUG
        self.data = self.load_data(args)
        self.process_data()

    def load_data(self, args):
        """
        Load raw data
        """
        with open(os.path.join(f'{self.root}/{self.dataset}.json')) as f:
            info = json.load(f)
        data_train = pd.DataFrame(info['split']['train'])
        data_val = pd.DataFrame(info['split']['ood_val'])
        data_test = pd.DataFrame(info['split']['ood_test'])
        data = dict()
        if args.DEBUG:
            data['train'], data['valid'], data['test'] = data_train[:50], data_val[:50], data_test[:50]
            return data
        else:
            data['train'], data['valid'], data['test'] = data_train, data_val, data_test
            return data

    def process_data(self):
        """
        Generate molecule graph
        """
        # Check cache file
        if self.debug:
            check_file = f'{self.save_path}/test_debug.pkl'
        else:
            check_file = f'{self.save_path}/test.pkl'
        if os.path.exists(check_file):
            print('The processed dataset is saved in ', self.save_path)
        else:
            os.makedirs(f'{self.save_path}', exist_ok=True)
            for dataset in ['train', 'valid', 'test']:
                print('Processing raw protein-ligand complex data...')
                smiles_list = self.data[dataset]['smiles'].tolist()
                labels_list = self.data[dataset]['cls_label'].tolist()
                protein_list = self.data[dataset]['protein'].tolist()
                print(sum(labels_list) / len(labels_list))
                escapes = 0

                mol_graphs = []
                prot_seqs = []
                labels = []
                prots_mask = []

                for i, (smiles, label, seq) in enumerate(
                        tqdm(zip(smiles_list, labels_list, protein_list))):
                    try:
                        graph = self.build_graph(smiles)
                        mol_graphs.append(graph)
                        labels.append(label)
                        prot_emb, prot_mask = self.prot_emb(seq)
                        prot_seqs.append(prot_emb)
                        prots_mask.append(prot_mask)

                    except:
                        escapes += 1
                print(f'[INFO] there are {escapes} mols escapes in {dataset} set')
                self.save(dataset, mol_graphs, labels, prot_seqs, prots_mask)

    def build_graph(self, smiles):
        """ Build whole graph from smiles sequences """
        mol_graph = self.smiles2graph(smiles)
        return mol_graph

    def save(self, dataset, mol_graphs, labels, prot_seqs, prots_mask):
        """ Save the generated graphs. """
        print('Saving processed complex data...')

        if self.debug:
            save_file = f'{self.save_path}/' + dataset + '_debug.pkl'
        else:
            save_file = f'{self.save_path}/' + dataset + '.pkl'
        with open(save_file, 'wb') as f:
            pickle.dump((mol_graphs, prot_seqs, prots_mask, labels), f)

    def prot_emb(self, seq):
        re = torch.zeros(self.max_protein_len, dtype=torch.int64)
        length = min(len(seq), self.max_protein_len)
        for i in range(length):
            re[i] = seq_dict[seq[i]]
        mask = torch.LongTensor(([0] * length) + ([1] * (self.max_protein_len - length)))
        return re, mask

    def get_atom_features(self, atom):
        """DrugOOD atom features"""
        # The usage of features is along with the Attentive FP.
        feature = np.zeros(39)
        # Symbol
        symbol = atom.GetSymbol()  # 16 (15 + 1 other)
        symbol_list = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']
        if symbol in symbol_list:
            loc = symbol_list.index(symbol)
            feature[loc] = 1
        else:
            feature[15] = 1

        # Degree
        degree = atom.GetDegree()  # 5
        if degree > 5:
            print("atom degree larger than 5. Please check before featurizing.")
            raise RuntimeError
        feature[16 + degree] = 1

        # Formal Charge
        charge = atom.GetFormalCharge()  # 1
        feature[22] = charge

        # radical electrons
        radelc = atom.GetNumRadicalElectrons()  # 1
        feature[23] = radelc

        # Hybridization
        hyb = atom.GetHybridization()  # 6 ( 5 + 1 other)
        hybridization_list = [rdkit.Chem.rdchem.HybridizationType.SP,
                              rdkit.Chem.rdchem.HybridizationType.SP2,
                              rdkit.Chem.rdchem.HybridizationType.SP3,
                              rdkit.Chem.rdchem.HybridizationType.SP3D,
                              rdkit.Chem.rdchem.HybridizationType.SP3D2]
        if hyb in hybridization_list:
            loc = hybridization_list.index(hyb)
            feature[loc + 24] = 1
        else:
            feature[29] = 1

        # aromaticity
        if atom.GetIsAromatic():  # 1
            feature[30] = 1

        # hydrogens
        hs = atom.GetNumImplicitHs()  # 5
        feature[31 + hs] = 1

        # chirality, chirality type
        if atom.HasProp('_ChiralityPossible'):
            # TODO what kind of error
            feature[36] = 1

            try:
                chi = atom.GetProp('_CIPCode')
                chi_list = ['R', 'S']
                loc = chi_list.index(chi)
                feature[37 + loc] = 1
            except KeyError:
                feature[37] = 0
                feature[38] = 0
        return feature

    def get_bond_features(self, bond):
        feature = np.zeros(10)

        # bond type
        type = bond.GetBondType()
        bond_type_list = [rdkit.Chem.rdchem.BondType.SINGLE,
                          rdkit.Chem.rdchem.BondType.DOUBLE,
                          rdkit.Chem.rdchem.BondType.TRIPLE,
                          rdkit.Chem.rdchem.BondType.AROMATIC]
        if type in bond_type_list:
            loc = bond_type_list.index(type)
            feature[0 + loc] = 1
        else:
            print("Wrong type of bond. Please check before feturization.")
            raise RuntimeError

        # conjugation
        conj = bond.GetIsConjugated()
        feature[4] = conj

        # ring
        ring = bond.IsInRing()
        feature[5] = ring

        # stereo
        stereo = bond.GetStereo()
        stereo_list = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                       rdkit.Chem.rdchem.BondStereo.STEREOANY,
                       rdkit.Chem.rdchem.BondStereo.STEREOZ,
                       rdkit.Chem.rdchem.BondStereo.STEREOE]
        if stereo in stereo_list:
            loc = stereo_list.index(stereo)
            feature[6 + loc] = 1
        else:
            print("Wrong stereo type of bond. Please check before featurization.")
            raise RuntimeError

        return feature

    def smiles2graph(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if (mol is None):
            return None
        src = []
        dst = []
        atom_feature = []
        bond_feature = []

        try:
            for atom in mol.GetAtoms():
                one_atom_feature = self.get_atom_features(atom)
                atom_feature.append(one_atom_feature)

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                one_bond_feature = self.get_bond_features(bond)
                src.append(i)
                dst.append(j)
                bond_feature.append(one_bond_feature)
                src.append(j)
                dst.append(i)
                bond_feature.append(one_bond_feature)

            src = torch.tensor(src).long()
            dst = torch.tensor(dst).long()
            atom_feature = np.array(atom_feature)
            bond_feature = np.array(bond_feature)
            atom_feature = torch.tensor(atom_feature).float()
            bond_feature = torch.tensor(bond_feature).float()
            graph_cur_smile = dgl.graph((src, dst), num_nodes=len(mol.GetAtoms()))
            graph_cur_smile.ndata['x'] = atom_feature
            graph_cur_smile.edata['x'] = bond_feature
            return graph_cur_smile

        except RuntimeError:
            return None

    def featurize_atoms(self, mol):
        feats = []
        for atom in mol.GetAtoms():
            feats.append(atom.GetAtomicNum())
        return {'atomic': torch.tensor(feats).reshape(-1).to(torch.int64)}

    def featurize_bonds(self, mol):
        feats = []
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        for bond in mol.GetBonds():
            btype = bond_types.index(bond.GetBondType())
            # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
            feats.extend([btype, btype])
        return {'type': torch.tensor(feats).reshape(-1).to(torch.int64)}
