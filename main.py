import argparse
import os
import torch.nn
from model import Mymodel
from dataset import DrugOOD
from collator import MoleDataset, Collator_fn
import copy
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
from torch.utils import data
from tqdm import tqdm
import json
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef
import torch.nn as nn
import random


# 对于蛋白质和药物需要分别跑causal 和 非causal的证明在protein上找casual的是有意义的吗

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # dgl.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("--DEBUG", type=bool, default=1)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--repeat', type=int, default=5, help='')
    # dataset
    parser.add_argument('--dataset', default='sbap_core_ic50_size', type=str, help='name of data set')
    parser.add_argument('--root', default='/media/data2/caoye/OOD/code/OODv2_2.10/processed',
                        type=str, help='root of input data')
    parser.add_argument('--save_graph', default='/media/data2/caoye/OOD/code/OODv2_2.10/processed/graph', type=str,
                        help='path of graphs')
    # Training parameters
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                        help='mini-batch size (default: 128), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='wd', help='initial weight decay', dest='wd')
    parser.add_argument("--patience", type=int, default=10)
    # Model setting
    parser.add_argument('--readout', default='last', type=str)
    parser.add_argument('--layer_norm', default=0, type=bool)
    # parser.add_argument('--emb_size', default=512, type=int)
    parser.add_argument('--hidden_size', default=192, type=int)
    parser.add_argument('--base_res', default=1, type=bool, help='residual connection in base GNN')
    parser.add_argument('--residual', default=1, type=bool,
                        help='residual connection for causal graph and shortcut graph learning')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument("--graph_pool", type=str, default='mean', choices=['sum', 'max', 'mean', 'wsum', 'set2set'],
                        help='Pooling graph node')
    parser.add_argument("--layer_num", type=int, default=3, help='Number of layer in base GNN.')
    parser.add_argument('--GNNConv', type=str, default='GCN', choices=['GCN', 'GAT', 'SAGE', 'GIN'])
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--node_attn', type=bool, default=1,
                        help='Whether divide molecule graph to causal and shortcut graph')
    parser.add_argument('--prot_attn', type=bool, default=1,
                        help='')
    parser.add_argument('--causal_inference', type=bool, default=0, help='')
    parser.add_argument('--interaction', type=str, default='concat', choices=['concat', 'bilinear'],
                        help='The way to get interaction features of molecule and protein')
    parser.add_argument('--max_protein_len', type=int, default=1000, help='Max length of protein sequence')
    parser.add_argument('--num_attn_layer', type=int, default=2, help='')
    parser.add_argument('--num_lstm_layer', type=int, default=2, help='')
    parser.add_argument('--num_attn_head_prot', type=int, default=12, help='')
    # Loss
    parser.add_argument('--task', type=str, default='DTI', choices=['DTI', 'DDI'])
    parser.add_argument('--distribution', type=str, default='uniform', choices=['uniform', 'normal'])
    parser.add_argument('--with_random', type=int, default=1, help='interaction way')
    parser.add_argument('--cat_or_add', type=str, default='cat', choices=['cat', 'add'])
    parser.add_argument('--shortcut_loss', type=str, default='KL', choices=['MSE', 'KL'], help='shortcut graph loss')
    parser.add_argument("--lam1", type=float, default=0.1, help='the weight of causal loss')
    parser.add_argument("--lam2", type=float, default=0.1, help='the weight of inv loss')
    parser.add_argument("--lam3", type=float, default=0.1, help='the weight of inv loss')
    parser.add_argument("--margin", type=float, default=0.8, help='parameter of MarginRankLoss')
    # save path
    parser.add_argument("--save_path", type=str, default='./results')
    args = parser.parse_args()
    return args


def calcu_metric(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    precision = tpr / (tpr + fpr + 0.00001)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    optim_thred = thresholds[1:][np.argmax(f1[1:])]
    print("optimal threshold: " + str(optim_thred))
    y_pred = np.where(y_pred >= optim_thred, 1, 0)

    confusion = confusion_matrix(y_true, y_pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    acc = (TP + TN) / float(TP + TN + FP + FN)
    sensitity = TP / float(TP + FN)
    specificity = TN / float(TN + FP)

    mcc = matthews_corrcoef(y_true, y_pred)
    precision = TP / (TP + FP)  # equal to: precision_score(y_true, y_pred)
    recall = TP / (TP + FN)  # equal to: recall_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)  # equal to: f1_score(y_true, y_pred)

    return round(acc, 4), round(sensitity, 4), round(specificity, 4), round(mcc, 4), \
           round(precision, 4), round(recall, 4), round(f1, 4)


def get_au_aupr(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)
    return round(auc, 4), round(aupr, 4)


def validate(args, data_generator, model):
    model.eval()

    y_pred = []
    y_label = []

    with torch.no_grad():
        for i, (graph, protein, prot_mask, label) in enumerate(data_generator):
            graph, protein, prot_mask, label = graph.to(args.device), protein.to(args.device), prot_mask.to(
                args.device), torch.FloatTensor(np.array(label)).to(args.device)

            pred = model(graph, protein, prot_mask)

            y_pred.append(pred)
            y_label.append(label)

    y_pred = torch.cat(y_pred, dim=0)
    y_label = torch.cat(y_label, dim=0)
    return y_pred.flatten(), y_label.flatten()


def training(args, model, train_dataloader, opt, loss_margin):
    model.train()
    loss_fct = nn.BCELoss()
    total_loss = []
    for i, (graph, protein, prot_mask, label) in enumerate(train_dataloader):
        graph, protein, prot_mask, label = graph.to(args.device), protein.to(args.device), prot_mask.to(args.device), \
                                           torch.FloatTensor(np.array(label)).to(args.device)
        target = torch.ones(len(label)//2).to(args.device)

        if args.causal_inference:

            pred, c_pred, s_pred, interv_pred = model(graph, protein, prot_mask, training=True)
            pred, c_pred, s_pred, interv_pred = pred.view(-1), c_pred.view(-1), s_pred.view(-1), interv_pred.view(-1)

            if args.distribution == 'uniform':
                if args.shortcut_loss == 'KL':
                    target_s = torch.ones((len(s_pred), 2), dtype=torch.float).to(args.device) / 2  # binary classification
                if args.shortcut_loss == 'MSE':
                    target_s = torch.ones_like(s_pred, dtype=torch.float).to(args.device) / 2  # binary classification
            else:
                if args.shortcut_loss == 'KL':
                    target_s = torch.randn(len(s_pred), 2, dtype=torch.float32).to(args.device)
                    target_s = torch.softmax(target_s, dim=-1)
                if args.shortcut_loss == 'MSE':
                    target_s = torch.randn(len(s_pred), dtype=torch.float32).to(args.device)
                    target_s = torch.sigmoid(target_s)
            if args.shortcut_loss == 'KL':
                s_pred = torch.log(torch.stack((torch.sigmoid(s_pred), 1 - torch.sigmoid(s_pred)), dim=1))
                s_loss = F.kl_div(s_pred, target_s, reduction='batchmean')
            if args.shortcut_loss == 'MSE':
                s_loss = F.mse_loss(torch.sigmoid(s_pred), target_s, reduction='mean')

            loss_global = loss_margin(pred[:len(pred)//2], pred[len(pred)//2:], target)
            loss_inv = loss_fct(torch.sigmoid(interv_pred), label)
            pred = torch.sigmoid(pred)
            c_pred = torch.sigmoid(c_pred)
            loss_dist = F.kl_div(torch.log(torch.stack((pred, 1 - pred), dim=1)),
                                 torch.stack((c_pred, 1 - c_pred), dim=1), reduction='batchmean')
            print(round(loss_global.item(), 4), round(loss_inv.item(), 4), round(loss_dist.item()), round(s_loss.item(), 4))
            loss = loss_global + args.lam1 * s_loss + args.lam2 * loss_inv + args.lam3 * loss_dist
        else:
            pred = model(graph, protein, prot_mask, training=True).view(-1)
            pred_pos = pred[:len(pred)//2]
            pred_neg = pred[len(pred)//2:]
            loss = loss_margin(pred_pos, pred_neg, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss.append(loss)

    return torch.stack(total_loss).mean()


def main(args):
    print(args)
    print('[INFO] Preparing data ... ')
    save_path = f'{args.save_path}/{args.dataset}/{args.GNNConv}/' \
                f'lr{args.lr}_weight_decay{args.wd}_hsize{args.hidden_size}_layer{args.layer_num}' \
                f'_pool{args.graph_pool}_layer_norm{args.layer_norm}_sloss_{args.shortcut_loss}' \
                f'_lambda{args.lam1}_{args.lam2}_{args.lam3}_res{args.base_res}_{args.residual}_interaction{args.interaction}' \
                f'_causal{args.causal_inference}_margin{args.margin}_drop{args.dropout}_prot{args.num_attn_layer}_' \
                f'{args.num_attn_head_prot}_{args.num_lstm_layer}/'

    os.makedirs(f'{save_path}/model', exist_ok=True)
    args.save_graph = f'{args.save_graph}_{args.dataset}'
    print('Results are save in ', save_path)

    if os.path.exists(f'{save_path}/model/best_model_repeat{args.num_run}.pth'):
        print('This run has been finished!!!!!!!')
        return

    DrugOOD(args)
    training_set = MoleDataset(f'{args.save_graph}', 'train', args.DEBUG)
    validation_set = MoleDataset(f'{args.save_graph}', 'valid', args.DEBUG)
    testing_set = MoleDataset(f'{args.save_graph}', 'test', args.DEBUG)
    collate_fn = Collator_fn()
    training_generator = data.DataLoader(training_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                         drop_last=False, shuffle=True, collate_fn=collate_fn)
    validation_generator = data.DataLoader(validation_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                           drop_last=False, shuffle=False, collate_fn=collate_fn)
    testing_generator = data.DataLoader(testing_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                        drop_last=False, shuffle=False, collate_fn=collate_fn)

    # Build model
    if os.path.exists(f'{save_path}/model/model_repeat{args.num_run}.pth'):
        model = torch.load(f'{save_path}/model/model_repeat{args.num_run}.pth')
    else:
        model = Mymodel(args)
    model = model.to(args.device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)

    # loss
    loss_margin = nn.MarginRankingLoss(margin=args.margin)
    f_csv = open(f'./{save_path}/results.csv', 'a')
    f_csv.write('repeat,acc,sensitity,specificity,mcc,precision,recall,f1,auc,aupr\n')
    f_csv.close()

    max_auc = 0
    model_max = copy.deepcopy(model)

    stop_count = 0
    for epo in tqdm(range(args.epochs)):
        # training
        train_loss = training(args, model, training_generator, opt, loss_margin)

        # validation
        val_pred, val_true = validate(args, validation_generator, model)
        target = torch.ones(len(val_pred)//2).to(args.device)
        val_loss = loss_margin(val_pred[:len(val_pred)//2], val_pred[len(val_pred)//2:], target)
        val_auc, val_aupr = get_au_aupr(val_true.cpu().numpy(), torch.sigmoid(val_pred).cpu().numpy())
        print(
            f'Epoch {epo}: train loss {round(train_loss.item(), 4)}, val loss {round(val_loss.item(), 4)}, auc {val_auc}, aupr {val_aupr}')

        # early stopping
        if val_auc > max_auc:
            max_auc = val_auc
            model_max = copy.deepcopy(model)
            torch.save(model_max, f'{save_path}/model/model_repeat{args.num_run}.pth')
            stop_count = 0
        else:
            stop_count += 1

        if (stop_count > args.patience) or (epo == args.epochs - 1):
            torch.save(model_max, f'{save_path}/model/best_model_repeat{args.num_run}.pth')
            print('\nEarly Stopped!')
            break

    print('[INFO] Testing ...')
    test_pred, test_true = validate(args, testing_generator, model_max)
    test_pred = torch.sigmoid(test_pred).cpu().numpy()
    test_true = test_true.cpu().numpy()
    results = dict()
    results['y_pred'] = test_pred.tolist()
    results['y_true'] = test_true.tolist()
    json.dump(results, open(f'./{save_path}/repeat{args.num_run}_logits.json', 'w'))

    auc, aupr = get_au_aupr(test_true, test_pred)
    acc, sensitity, specificity, mcc, precision, recall, f1 = calcu_metric(test_true, test_pred)
    print(f'Testing AUC: {auc}, AUPR: {aupr}')
    ls = [args.num_run, acc, sensitity, specificity, mcc, precision, recall, f1, auc, aupr]
    f_csv = open(f'./{save_path}/results.csv', 'a')
    f_csv.write(','.join(map(str, ls)) + '\n')
    f_csv.close()
    print('Results are save in ', save_path)


if __name__ == '__main__':
    args = parse_args()
    args.num_run = 0
    print(f'This is repeat {args.num_run}')
    set_seeds(222 + 2 ^ args.num_run)
    main(args)
    print('Done~~~')
