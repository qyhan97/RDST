import sys
import os
import os.path as osp
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))
import torch
import copy
import numpy as np
import time
import math
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from Main.pargs import pargs
from Main.dataset import TreeDataset
from Main.word2vec import Embedding, collect_sentences, train_word2vec
from Main.model import Net
from Main.utils import create_log_dict_pretrain, write_log, write_json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data

def generateData(model, unlabel_loader, device, epoch, percentile, sample_alpha):

    model.eval()
    labeled_datalist = []
    true_pseudo_datalist = []
    false_pseudo_datalist = []

    true_outputs = []
    false_outputs = []

    wrong_count = 0
    wrong_T = 0
    wrong_F = 0
    total_T = 0
    total_F = 0
    list_F = []
    list_T = []

    len_F = 0
    len_T = 0
    for data in unlabel_loader:

        data = data.to(device)
        logits, pred = model(data)
        y_prob_max = pred.max(1).values
        y_pred = pred.max(1).indices
        for index in y_pred:
            if index == 1:
                len_F +=1
            if index == 0:
                len_T +=1
    list_unlabel_num = [len_T, len_F]
    dist_unlabel = (torch.tensor(list_unlabel_num, dtype=torch.float) / (len_F + len_T)).to(device)
    dist_target = (torch.ones([1, 2], dtype=torch.float) / 2).to(device)
    r = (dist_target / dist_unlabel) ** args.DA_t
    for data in unlabel_loader:
        data = data.to(device)
        logits, pred = model(data)
        dist_aligned = torch.softmax(logits.detach(), dim=-1) * r
        dist_aligned_softmax = torch.softmax(dist_aligned, dim=-1)
        y_prob_max = dist_aligned_softmax.max(1).values
        y_pred = dist_aligned_softmax.max(1).indices
        
        for index, prob in enumerate(y_prob_max):
            if y_pred[index] == 1:
                if data[index].y[0] != 1:
                    wrong_F += 1
                    list_F.append(1)
                if data[index].y[0] == 1:
                    list_F.append(0)

                onedata = Data(x=data[index].x, y=torch.tensor([y_pred[index]]).to(device), edge_index=data[index].edge_index)
                false_pseudo_datalist.append(onedata)
                false_outputs.append(y_prob_max[index])

            if y_pred[index] == 0:
                if data[index].y[0] != 0:
                    wrong_T += 1
                    list_T.append(1)
                if data[index].y[0] == 0:
                    list_T.append(0)
                onedata = Data(x=data[index].x, y=torch.tensor([y_pred[index]]).to(device), edge_index=data[index].edge_index)
                true_pseudo_datalist.append(onedata)
                true_outputs.append(y_prob_max[index])
            if y_pred[index] != data[index].y[0]:
                wrong_count += 1
                
            if data[index].y[0] == 0:
                total_T += 1
            if data[index].y[0] == 1:
                total_F += 1

    true_outputs = torch.tensor(true_outputs)
    false_outputs = torch.tensor(false_outputs)

    true_sorted_indices = torch.argsort(true_outputs, descending=True)
    true_sorted_indices = true_sorted_indices.tolist()

    false_sorted_indices = torch.argsort(false_outputs, descending=True)
    false_sorted_indices = false_sorted_indices.tolist()

    all_outputs = torch.cat((true_outputs, false_outputs), dim=0)
    all_sorted_output, _ = torch.sort(all_outputs, descending=True)
    all_sorted_output = all_sorted_output.tolist()
    length_all = len(all_sorted_output)
    length_threshold = int(length_all * percentile)
    threshold = all_sorted_output[length_threshold - 1]

    length_true = len(true_sorted_indices)
    length_false = len(false_sorted_indices)
    max_TF = max(length_true, length_false)
    miu_T = (length_false/max_TF) ** sample_alpha
    miu_F = (length_true/max_TF) ** sample_alpha

    T_threshold_index = 0
    F_threshold_index = 0

    for i, index in enumerate(false_sorted_indices):
        if false_outputs[index] < threshold:
            F_threshold_index = i
            break
        elif i == (len(false_sorted_indices) - 1):
            F_threshold_index = i + 1

    for i, index in enumerate(true_sorted_indices):
        if true_outputs[index] < threshold:
            T_threshold_index = i
            break
        elif i == (len(true_sorted_indices) - 1):
            T_threshold_index = i + 1

    select_false = int(F_threshold_index * miu_F)
    select_true = int(T_threshold_index * miu_T)

    false_select_indices = false_sorted_indices[:select_false]
    true_select_indices = true_sorted_indices[:select_true]

    unlabeled_latalist_len1 = len(false_pseudo_datalist)+len(true_pseudo_datalist)

    true_select = 0
    wrong_T_added = 0

    for i in true_select_indices:
        labeled_datalist.append(true_pseudo_datalist[i])
        true_select += 1
        if list_T[i] == 1:
            wrong_T_added +=1

    false_select = 0
    wrong_F_added = 0
    for i in false_select_indices:
        labeled_datalist.append(false_pseudo_datalist[i])
        false_select += 1
        if list_F[i] == 1:
            wrong_F_added +=1

    sum_select = false_select + true_select
    num_batches = math.ceil((sum_select + 2 * args.k) / args.batch_size)
    label_batch = math.ceil(2 * args.k / num_batches)

    wrong_added_sum = wrong_T_added+wrong_F_added
    log_info2 = f'  epoch:{epoch}, unlabeled set number:{unlabeled_latalist_len1}\n' \
                + f' T selected: {true_select}, F selected:{false_select}, sum selected:{sum_select} \n ' \
                + f' total_T:{total_T}, total_F:{total_F} \n ' \
                + f' pseudo_T:{length_true}, pseudo_F:{length_false} \n' \
                + f' wrong_T:{wrong_T}, wrong_F:{wrong_F}, wrong_sum:{wrong_count}\n' \
                + f' wrong_T_added:{wrong_T_added}, wrong_F_added:{wrong_F_added}, wrong_added_sum:{wrong_added_sum}\n'
    return labeled_datalist, label_batch, log_info2

def pre_train(unsup_train_loader, model, optimizer, device):
    model.train()
    loss_all = 0

    for data in unsup_train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        loss = model.unsup_loss(data)
        loss.backward()
        optimizer.step()

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(unsup_train_loader.dataset)


def fine_tuning(train_loader, model, optimizer, device):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        logits, out = model(data)
        CEloss = F.nll_loss(out, data.y.long().view(-1))
        h = torch.softmax(logits, dim=-1)
        h_mean = torch.mean(h, dim=0)
        p = (torch.ones([1, 2], dtype=torch.float) / 2).to(device)
        equal_loss = torch.sum(p * torch.log(p / h_mean))
        loss = CEloss + lambda_eq * equal_loss
        loss.backward()

        loss_all += loss.item() * data.num_graphs

        optimizer.step()
    return loss_all / len(train_loader.dataset)

def fine_tuning2(labeled_loader, pseudo_loader, model, optimizer, device):
    model.train()
    total_loss = 0

    for data_l, data_u in zip(labeled_loader, pseudo_loader):
        data_l = data_l.to(device)
        data_u = data_u.to(device)
        optimizer.zero_grad()
        logits_l, out_l = model(data_l)
        logits_u, out_u = model(data_u)
        logits = torch.cat((logits_l, logits_u), dim=0)
        h = torch.softmax(logits, dim=-1)
        h_mean = torch.mean(h, dim=0)
        p = (torch.ones([1, 2], dtype=torch.float) / 2).to(device)
        equal_loss = torch.sum(p * torch.log(p / h_mean))

        CEloss_l = F.nll_loss(out_l, data_l.y.long().view(-1))
        CEloss_u = F.nll_loss(out_u, data_u.y.long().view(-1))

        loss = CEloss_l + lambda_u * CEloss_u + lambda_eq * equal_loss

        loss.backward()

        total_loss += loss.item() * (data_l.num_graphs+ data_u.num_graphs)

        optimizer.step()
    return total_loss / (len(labeled_loader.dataset)+len(pseudo_loader.dataset))

def test(model, dataloader, num_classes, device):
    model.eval()
    error = 0

    y_true = []
    y_pred = []

    for data in dataloader:
        data = data.to(device)
        _, pred = model(data)
        error += F.nll_loss(pred, data.y.long().view(-1)).item() * data.num_graphs
        y_true += data.y.tolist()
        y_pred += pred.max(1).indices.tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = round(accuracy_score(y_true, y_pred), 4)
    precs = []
    recs = []
    f1s = []
    for label in range(num_classes):
        precs.append(round(precision_score(y_true == label, y_pred == label, labels=True), 4))
        recs.append(round(recall_score(y_true == label, y_pred == label, labels=True), 4))
        f1s.append(round(f1_score(y_true == label, y_pred == label, labels=True), 4))
    micro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)

    macro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    return error / len(dataloader.dataset), acc, precs, recs, f1s, \
           [micro_p, micro_r, micro_f1], [macro_p, macro_r, macro_f1]

def test2(model, labeled_loader, pseudo_loader, num_classes, device):
    model.eval()
    error = 0

    y_true = []
    y_pred = []
    for data_l, data_u in zip(labeled_loader, pseudo_loader):
        data_l = data_l.to(device)
        data_u = data_u.to(device)
        _, out_l = model(data_l)
        _, out_u = model(data_u)

        CEloss_l = F.nll_loss(out_l, data_l.y.long().view(-1))
        CEloss_u = F.nll_loss(out_u, data_u.y.long().view(-1))

        loss = CEloss_l + lambda_u * CEloss_u
        error += loss.item() * (data_l.num_graphs + data_u.num_graphs)
        y = torch.cat((data_l.y.long(), data_u.y.long()))
        out = torch.cat((out_l,out_u))

        y_true += y.tolist()
        y_pred += out.max(1).indices.tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = round(accuracy_score(y_true, y_pred), 4)
    precs = []
    recs = []
    f1s = []
    for label in range(num_classes):
        precs.append(round(precision_score(y_true == label, y_pred == label, labels=True), 4))
        recs.append(round(recall_score(y_true == label, y_pred == label, labels=True), 4))
        f1s.append(round(f1_score(y_true == label, y_pred == label, labels=True), 4))
    micro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)

    macro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    return error / (len(labeled_loader.dataset)+len(pseudo_loader.dataset)), acc, precs, recs, f1s, \
           [micro_p, micro_r, micro_f1], [macro_p, macro_r, macro_f1]

def test_and_log(model, val_loader, test_loader, num_classes, device, epoch, lr, loss, train_acc, ft_log_record):
    val_error, val_acc, val_precs, val_recs, val_f1s, val_micro_metric, val_macro_metric = \
        test(model, val_loader, num_classes, device)
    test_error, test_acc, test_precs, test_recs, test_f1s, test_micro_metric, test_macro_metric = \
        test(model, test_loader, num_classes, device)

    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Val ERROR: {:.7f}, Test ERROR: {:.7f}\n  Train ACC: {:.4f}, Validation ACC: {:.4f}, Test ACC: {:.4f}\n' \
                   .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc) \
               + f'  Test PREC: {test_precs}, Test REC: {test_recs}, Test F1: {test_f1s}\n' \
               + f'  Test Micro Metric(PREC, REC, F1):{test_micro_metric}, Test Macro Metric(PREC, REC, F1):{test_macro_metric}'


    ft_log_record['val accs'].append(val_acc)
    ft_log_record['test accs'].append(test_acc)
    ft_log_record['test precs'].append(test_precs)
    ft_log_record['test recs'].append(test_recs)
    ft_log_record['test f1s'].append(test_f1s)
    ft_log_record['test micro metric'].append(test_micro_metric)
    ft_log_record['test macro metric'].append(test_macro_metric)
    return val_error, val_acc, log_info, ft_log_record

if __name__ == '__main__':
    args = pargs()

    unsup_train_size = args.unsup_train_size
    dataset = args.dataset
    unsup_dataset = args.unsup_dataset
    vector_size = args.vector_size
    device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    ft_runs = args.ft_runs
    num_gc_layers = args.layers
    k = args.k
    delta = args.delta
    weight_decay = args.weight_decay
    lambda_eq = args.lambda_eq
    lambda_u = args.lambda_u

    word_embedding = 'tfidf' if 'tfidf' in dataset else 'word2vec'
    lang = 'ch' if 'Weibo' in dataset else 'en'
    tokenize_mode = args.tokenize_mode

    batch_size = args.batch_size
    epochs = args.epochs
    ft_epochs = args.ft_epochs
    diff_lr = args.diff_lr
    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, f'dataset_{k}')
    train_path = osp.join(label_dataset_path, 'train')
    val_path = osp.join(label_dataset_path, 'val')
    test_path = osp.join(label_dataset_path, 'test')
    unlabeled_path = osp.join(label_dataset_path, 'unlabeled')
    unlabel_dataset_path = osp.join(dirname, '..', 'Data', unsup_dataset, 'dataset')
    model_path = osp.join(dirname, '..', 'Model',
                          f'w2v_{dataset}_{tokenize_mode}_{unsup_train_size}_{vector_size}.model')

    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'{log_name}.json')
    weight_path = osp.join(dirname, '..', 'Model', f'{log_name}.pt')

    log = open(log_path, 'w')
    log_dict = create_log_dict_pretrain(args)

    word2vec = Embedding(model_path, lang, tokenize_mode) if word_embedding == 'word2vec' else None
    final_test_acc = []
    best_acc = []
    for run in range(runs):
        write_log(log, f'run:{run}')
        log_record = {
            'run': run,
            'record': []
        }
        unlabel_dataset = TreeDataset(unlabeled_path, word_embedding, word2vec)
        unsup_train_loader = DataLoader(unlabel_dataset, batch_size=batch_size, shuffle=True)
        num_classes = unlabel_dataset.num_classes
        num_features = unlabel_dataset.num_features

        model = Net(num_features, args.hid_feats, num_gc_layers, num_classes).to(device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        for epoch in range(1, epochs + 1):
            pretrain_loss = pre_train(unsup_train_loader, model, optimizer, device)

            log_info = 'Epoch: {:03d}, Loss: {:.7f}'.format(epoch, pretrain_loss)
            write_log(log, log_info)

        torch.save(model.state_dict(), weight_path)
        write_log(log, '')
        # finetune
        ft_lr = args.ft_lr
        write_log(log, f'k:{k}')
        ft_log_record = {'k': k, 'val accs': [], 'test accs': [], 'test precs': [], 'test recs': [],
                         'test f1s': [], 'test micro metric': [], 'test macro metric': []}

        train_dataset = TreeDataset(train_path, word_embedding, word2vec)
        val_dataset = TreeDataset(val_path, word_embedding, word2vec)
        test_dataset = TreeDataset(test_path, word_embedding, word2vec)
        unlabeled_dataset = TreeDataset(unlabeled_path, word_embedding, word2vec)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

        model.load_state_dict(torch.load(weight_path))
        optimizer = Adam(model.parameters(), lr=ft_lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

        val_error, val_acc, log_info, ft_log_record = test_and_log(model, val_loader, test_loader, num_classes,
                                                                   device, 0, args.ft_lr, 0, 0, ft_log_record)
        write_log(log, log_info)
        bestacc = 0
        best_val_acc = 0
        st_best_acc = []
        st_mean_acc = []

        for epoch in range(1, ft_epochs + 1):
            ft_lr = scheduler.optimizer.param_groups[0]['lr']
            _ = fine_tuning(train_loader, model, optimizer, device)

            train_error, train_acc, _, _, _, _, _ = test(model, train_loader, num_classes, device)
            val_error, val_acc, log_info, ft_log_record = test_and_log(model, val_loader, test_loader, num_classes,
                                                                       device,
                                                                       epoch, ft_lr, train_error, train_acc,
                                                                       ft_log_record)
            write_log(log, log_info)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
            if ft_log_record['test accs'][-1] > bestacc:
                bestacc = ft_log_record['test accs'][-1]

            scheduler.step(val_error)

        st_mean_acc.append(round(np.mean(ft_log_record['test accs'][-10:]), 3))
        st_best_acc.append(bestacc)
        iter = 0
        alpha = 0
        end = int(1 / delta)
        while iter < end:
            iter += 1
            bestacc = 0
            best_val_acc = 0
            alpha = alpha + delta
            if iter == 1:
                model.load_state_dict(best_model_state)
            else:
                model.load_state_dict(best_st_model_state)

            pseudo_dataset, labeled_batchsize, log_info2 = generateData(model, unlabeled_loader, device, epoch, alpha, args.alpha)
            write_log(log, log_info2)
            labeled_loader = DataLoader(train_dataset, batch_size=labeled_batchsize, shuffle=True)
            pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size-labeled_batchsize, shuffle=True)
            model.load_state_dict(torch.load(weight_path))
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

            val_error, val_acc, log_info, ft_log_record = test_and_log(model, val_loader, test_loader, num_classes,
                                                                       device, ft_epochs * iter, args.ft_lr, 0, 0,
                                                                       ft_log_record)
            write_log(log, log_info)
            for epoch in range(1 + iter * ft_epochs, 1 + (iter + 1) * ft_epochs):
                ft_lr = scheduler.optimizer.param_groups[0]['lr']
                _ = fine_tuning2(labeled_loader, pseudo_loader, model, optimizer, device)
                train_error, train_acc, _, _, _, _, _ = test2(model, labeled_loader, pseudo_loader, num_classes, device)
                val_error, val_acc, log_info, ft_log_record = test_and_log(model, val_loader, test_loader, num_classes,
                                                                           device,
                                                                           epoch, ft_lr, train_error, train_acc,
                                                                           ft_log_record)
                write_log(log, log_info)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_st_model_state = copy.deepcopy(model.state_dict())

                if ft_log_record['test accs'][-1] > bestacc:
                    bestacc = ft_log_record['test accs'][-1]

                if not diff_lr:
                    scheduler.step(val_error)
            st_best_acc.append(bestacc)
            st_mean_acc.append(round(np.mean(ft_log_record['test accs'][-10:]), 3))

        log_info4 = f'best_acc_list:{st_best_acc}\n'
        log_info5 = f'mean_acc_list:{st_mean_acc}\n'
        write_log(log, log_info4)
        write_log(log, log_info5)
        best_acc.append(bestacc)
        final_test_acc.append(round(np.mean(ft_log_record['test accs'][-10:]), 3))
        ft_log_record['mean acc'] = round(np.mean(ft_log_record['test accs'][-10:]), 3)
        log_record['record'].append(ft_log_record)
        write_log(log, '')

        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)
    final_acc = round(np.mean(final_test_acc), 4)
    final_best_acc = round(np.mean(best_acc), 4)
    print('final_mean_acc:', final_acc)
    print('final_mean_acc_list:', final_test_acc)
    print('final_best_acc:', final_best_acc)
    print('final_best_acc_list:', best_acc)
    log_info3 = f'final_main_acc:{final_acc}, final_main_acc_list:{final_test_acc}\n' \
                + f'final_best_acc:{final_best_acc}, final_best_acc_list:{best_acc}'
    write_log(log, log_info3)
