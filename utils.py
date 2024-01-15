import json
import os
import shutil
import jieba
import nltk
import re
import numpy as np
from nltk.tokenize import MWETokenizer

mwe_tokenizer = MWETokenizer([('<', '@', 'user', '>'), ('<', 'url', '>')], separator='')

def word_tokenizer(sentence, lang='en', mode='naive'):
    if lang == 'en':
        if mode == 'nltk':
            return mwe_tokenizer.tokenize(nltk.word_tokenize(sentence))
        elif mode == 'naive':
            return sentence.split()
    if lang == 'ch':
        if mode == 'jieba':
            return jieba.lcut(sentence)
        elif mode == 'naive':
            return sentence

def write_json(dict, path):
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(dict, file_obj, indent=4, ensure_ascii=False)

def clean_comment(comment_text):
    match_res = re.match('回复@.*?:', comment_text)
    if match_res:
        return comment_text[len(match_res.group()):]
    else:
        return comment_text

def write_post(post_list, path):
    for post in post_list:
        write_json(post[1], os.path.join(path, f'{post[0]}.json'))

def write_log(log, str):
    log.write(f'{str}\n')
    log.flush()

def write_npz(post_list, path):
    for post in post_list:
        np.savez(os.path.join(path, post[0]),post[1])

def create_log_dict_pretrain(args):
    log_dict = {}
    log_dict['dataset'] = args.dataset
    log_dict['unsup_dataset'] = args.unsup_dataset
    log_dict['vector_size'] = args.vector_size
    log_dict['unsup_train_size'] = args.unsup_train_size
    log_dict['runs'] = args.runs
    log_dict['ft_runs'] = args.ft_runs

    log_dict['batch_size'] = args.batch_size
    log_dict['tddroprate'] = args.tddroprate
    log_dict['budroprate'] = args.budroprate
    log_dict['hid_feats'] = args.hid_feats
    log_dict['out_feats'] = args.out_feats

    log_dict['lr'] = args.lr
    log_dict['ft_lr'] = args.ft_lr
    log_dict['epochs'] = args.epochs
    log_dict['ft_epochs'] = args.ft_epochs
    log_dict['weight_decay'] = args.weight_decay
    log_dict['k'] = args.k
    log_dict['droprate'] = args.aug_rate
    log_dict['aug1'] = args.aug1
    log_dict['aug2'] = args.aug2

    log_dict['record'] = []
    return log_dict

def create_log_dict(args):
    log_dict = {}
    log_dict['dataset'] = args.dataset
    log_dict['unsup_dataset'] = args.unsup_dataset
    log_dict['tokenize_mode'] = args.tokenize_mode
    log_dict['unsup_train_size'] = args.unsup_train_size

    log_dict['vector_size'] = args.vector_size
    log_dict['runs'] = args.runs

    log_dict['model'] = args.model
    log_dict['batch_size'] = args.batch_size
    log_dict['tddroprate'] = args.tddroprate
    log_dict['budroprate'] = args.budroprate
    log_dict['hid_feats'] = args.hid_feats
    log_dict['out_feats'] = args.out_feats

    log_dict['diff_lr'] = args.diff_lr
    log_dict['lr'] = args.lr
    log_dict['epochs'] = args.epochs
    log_dict['weight_decay'] = args.weight_decay
    log_dict['threshold'] = args.threshold
    log_dict['k'] = args.k

    log_dict['record'] = []
    return log_dict

def dataset_makedirs(dataset_path):
    train_path = os.path.join(dataset_path, 'train', 'raw')
    val_path = os.path.join(dataset_path, 'val', 'raw')
    test_path = os.path.join(dataset_path, 'test', 'raw')
    unlabeled_path = os.path.join(dataset_path, 'unlabeled', 'raw')

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(train_path)
    os.makedirs(val_path)
    os.makedirs(test_path)
    os.makedirs(unlabeled_path)

    os.makedirs(os.path.join(dataset_path, 'train', 'processed'))
    os.makedirs(os.path.join(dataset_path, 'val', 'processed'))
    os.makedirs(os.path.join(dataset_path, 'test', 'processed'))
    os.makedirs(os.path.join(dataset_path, 'unlabeled', 'processed'))

    return train_path, val_path, test_path, unlabeled_path
