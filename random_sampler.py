import json
import os
import pandas as pd
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for Random Sampler")
    parser.add_argument('--root', default='/mnt/hdd1/caoye/OOD/dataset/drugood_all',
                        type=str, help='root of input data')
    parser.add_argument('--save_path', default='/mnt/hdd1/caoye/OOD/code/OODv2/processed', type=str,
                        help='')
    args = parser.parse_args()
    return args


def load_data(args):
    """
    Load raw data
    """
    with open(os.path.join(f'{args.root}/{args.dataset}.json')) as f:
        info = json.load(f)
    data_train = pd.DataFrame(info['split']['train'])
    data_val = pd.DataFrame(info['split']['ood_val'])
    data_test = pd.DataFrame(info['split']['ood_test'])
    data = dict()
    data['train'], data['valid'], data['test'] = data_train, data_val, data_test
    return data

def check_sampling():
    datasets1 = ['potency', 'ki', 'ic50', 'ec50']
    datasets2 = ['assay', 'size', 'scaffold']
    for dataset1 in datasets1:
        for dataset2 in datasets2:
            data_train = pd.read_csv(
                f'/mnt/hdd1/caoye/OOD/code/OODv2/processed/sbap_core_{dataset1}_{dataset2}_train.csv')
            data_val = pd.read_csv(
                f'/mnt/hdd1/caoye/OOD/code/OODv2/processed/sbap_core_{dataset1}_{dataset2}_valid.csv')
            data_test = pd.read_csv(
                f'/mnt/hdd1/caoye/OOD/code/OODv2/processed/sbap_core_{dataset1}_{dataset2}_test.csv')
            data_train_neg = data_train[['smiles_neg', 'protein_neg', 'cls_label_neg']]
            data_train_pos = data_train[['smiles_pos', 'protein_pos', 'cls_label_pos']]
            data_train_neg = data_train_neg.rename(
                columns={'smiles_neg': 'smiles', 'protein_neg': 'protein', 'cls_label_neg': 'cls_label'})
            data_train_pos = data_train_pos.rename(
                columns={'smiles_pos': 'smiles', 'protein_pos': 'protein', 'cls_label_pos': 'cls_label'})
            data_train = pd.concat([data_train_pos, data_train_neg])

            data_val_neg = data_val[['smiles_neg', 'protein_neg', 'cls_label_neg']]
            data_val_pos = data_val[['smiles_pos', 'protein_pos', 'cls_label_pos']]
            data_val_neg = data_val_neg.rename(
                columns={'smiles_neg': 'smiles', 'protein_neg': 'protein', 'cls_label_neg': 'cls_label'})
            data_val_pos = data_val_pos.rename(
                columns={'smiles_pos': 'smiles', 'protein_pos': 'protein', 'cls_label_pos': 'cls_label'})
            data_val = pd.concat([data_val_pos, data_val_neg])

            data_test_neg = data_test[['smiles_neg', 'protein_neg', 'cls_label_neg']]
            data_test_pos = data_test[['smiles_pos', 'protein_pos', 'cls_label_pos']]
            data_test_neg = data_test_neg.rename(
                columns={'smiles_neg': 'smiles', 'protein_neg': 'protein', 'cls_label_neg': 'cls_label'})
            data_test_pos = data_test_pos.rename(
                columns={'smiles_pos': 'smiles', 'protein_pos': 'protein', 'cls_label_pos': 'cls_label'})
            data_test = pd.concat([data_test_pos, data_test_neg])

            print(dataset1, dataset2, set(zip(data_train['smiles'].tolist(), data_train['protein'].tolist()))
                                          & set(zip(data_val['smiles'].tolist(), data_val['protein'].tolist())),
                  set(zip(data_val['smiles'].tolist(), data_val['protein'].tolist()))
                      & set(zip(data_test['smiles'].tolist(), data_test['protein'].tolist())),
                  set(zip(data_train['smiles'].tolist(), data_train['protein'].tolist()))
                      & set(zip(data_test['smiles'].tolist(), data_test['protein'].tolist())))


def df_diff(subdf, df, columns):
    data = pd.concat([df, subdf]).reset_index().drop(['index'], axis=1)
    return data.drop_duplicates(subset=columns, keep=False)


def random_choose(smiles_set, protein, exist_pairs):
    while 1:
        mol = random.choice(smiles_set)
        if (protein, mol) not in exist_pairs:
            return mol


def pos_neg_pair(data, args):
    sequence = []
    smiles = []
    for dataset in ['train', 'valid', 'test']:
        sequence += data[dataset]['protein'].tolist()
        smiles += data[dataset]['smiles'].tolist()
    exist_pairs = list(zip(sequence, smiles))

    for dataset in ['train', 'valid', 'test']:
        data_tmp = data[dataset]
        smiles_set = list(set(data_tmp['smiles'].tolist()))
        data_pos = data_tmp[data_tmp['cls_label'] == 1][['smiles', 'protein', 'cls_label']]
        data_pos = data_pos.rename(
            columns={'smiles': 'smiles_pos', 'cls_label': 'cls_label_pos'})
        data_neg = data_tmp[data_tmp['cls_label'] == 0][['smiles', 'protein', 'cls_label']]
        data_neg = data_neg.rename(
            columns={'smiles': 'smiles_neg', 'cls_label': 'cls_label_neg'})
        data_neg_pos = pd.merge(data_pos, data_neg, how='left', on='protein')
        data_neg_pos['protein_neg'] = data_neg_pos['protein'].tolist()
        data_neg_pos = data_neg_pos.rename(columns={'protein': 'protein_pos'})
        data_neg_pos = data_neg_pos.drop_duplicates(subset=['smiles_pos', 'protein_pos'], keep='first')
        data_neg_pos = data_neg_pos.drop_duplicates(subset=['smiles_neg', 'protein_neg'], keep='first')
        data_neg_pos = data_neg_pos.dropna()

        data_pos_matched = data_neg_pos[['smiles_pos', 'protein_pos', 'cls_label_pos']]
        data_pos = data_pos.rename(columns={'protein': 'protein_pos'})
        data_pos_remain = df_diff(data_pos_matched, data_pos, ['smiles_pos', 'protein_pos'])
        data_pos_remain = data_pos_remain.rename(columns={'smiles_pos': 'smiles'})

        data_neg_matched = data_neg_pos[['smiles_neg', 'protein_neg', 'cls_label_neg']]
        data_neg = data_neg.rename(columns={'protein': 'protein_neg'})
        data_neg_remain = df_diff(data_neg_matched, data_neg, ['smiles_neg', 'protein_neg'])
        data_neg_remain = data_neg_remain.rename(columns={'smiles_neg': 'smiles'})

        data_neg_pos1 = pd.merge(data_pos_remain, data_neg_remain, how='left', on='smiles')
        data_neg_pos1['smiles_neg'] = data_neg_pos1['smiles'].tolist()
        data_neg_pos1 = data_neg_pos1.rename(columns={'smiles': 'smiles_pos'})
        data_neg_pos1 = data_neg_pos1.drop_duplicates(subset=['smiles_pos', 'protein_pos'], keep='first')
        data_neg_pos1 = data_neg_pos1.drop_duplicates(subset=['smiles_neg', 'protein_neg'], keep='first')
        data_neg_pos1 = data_neg_pos1.dropna()

        data_pos_matched = pd.concat([data_pos_matched, data_neg_pos1[['smiles_pos', 'protein_pos', 'cls_label_pos']]])
        data_neg_pos = pd.concat([data_neg_pos, data_neg_pos1])

        data_pos_remain = df_diff(data_pos_matched, data_pos, ['smiles_pos', 'protein_pos'])
        for index, row in data_pos_remain.iterrows():
            data_pos_remain.loc[index, 'protein_neg'] = data_pos_remain.loc[index, 'protein_pos']
            mol = random_choose(smiles_set, data_pos_remain.loc[index, 'protein_pos'], exist_pairs)
            data_pos_remain.loc[index, 'smiles_neg'] = mol
            # data_pos_remain.loc[index, 'cls_label_neg'] = 0
            exist_pairs += [(data_pos_remain.loc[index, 'protein_pos'], mol)]

        data_neg_pos = pd.concat([data_neg_pos, data_pos_remain])
        data_neg_pos['cls_label_neg'] = 0
        data_neg_pos.to_csv(f'{args.save_path}/{args.dataset}_{dataset}.csv', index=False)


def main(args):
    datasets = ['potency', 'ki', 'ic50', 'ec50']
    domains = ['assay', 'size', 'scaffold']

    for dataset in datasets:
        for domain in domains:
            args.dataset = f'sbap_core_{dataset}_{domain}'
            print(args.dataset)
            data = load_data(args)
            pos_neg_pair(data, args)

    check_sampling()


if __name__ == '__main__':
    args = parse_args()
    main(args)
