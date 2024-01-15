import os
import os.path as osp
import json
import random
import sys
import numpy as np
import torch
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))
from Main.pargs import pargs
from Main.utils import write_post, dataset_makedirs
seed = 1234

def sort_dataset(label_source_path, label_dataset_path, k_shot):
    train_path, val_path, test_path, unlabeled_path = dataset_makedirs(label_dataset_path)

    label_file_paths = []
    for filename in os.listdir(label_source_path):
        label_file_paths.append(os.path.join(label_source_path, filename))

    true_post = []
    false_post = []
    
    for filepath in label_file_paths:
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        if post['source']['label'] == 0:
            true_post.append((post['source']['tweet id'], post))
        elif post['source']['label'] == 1:
            false_post.append((post['source']['tweet id'], post))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

    random.shuffle(true_post)
    random.shuffle(false_post)
    
    train_post = []
    val_post = []
    unlabeled_post = []
    test_post = []

    train_post.extend(true_post[:k_shot])
    train_post.extend(false_post[:k_shot])
    
    val_post.extend(true_post[int(0.6*len(true_post)):int(0.8*len(true_post))])
    val_post.extend(false_post[int(0.6*len(false_post)):int(0.8*len(false_post))])
    
    unlabeled_post.extend(true_post[k_shot:int(0.6*len(true_post))])
    unlabeled_post.extend(false_post[k_shot:int(0.6*len(false_post))])
    
    test_post.extend(true_post[int(0.8*len(true_post)):])
    test_post.extend(false_post[int(0.8*len(false_post)):])
    
    random.shuffle(train_post)
    random.shuffle(val_post)
    random.shuffle(unlabeled_post)
    random.shuffle(test_post)

    print('train:',len(train_post))
    print('val:',len(val_post))
    print('unlabeled:',len(unlabeled_post))
    print('test:',len(test_post))
    

    write_post(train_post, train_path)
    write_post(val_post, val_path)
    write_post(unlabeled_post, unlabeled_path)
    write_post(test_post, test_path)

if __name__ == '__main__':
    args = pargs()
    k = args.k
    dataset = args.dataset
    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, f'dataset_{k}')
    sort_dataset(label_source_path, label_dataset_path, k)

