import argparse


def pargs():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Weibo')
    parser.add_argument('--tokenize_mode', type=str, default='jieba')
    parser.add_argument('--vector_size', type=int, help='word embedding size', default=200)
    parser.add_argument('--unsup_train_size', type=int, help='word embedding unlabel data train size', default=100000)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--ft_runs', type=int, default=1)

    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--model', type=str, default='bigcn')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tddroprate', type=float, default=0.2)
    parser.add_argument('--budroprate', type=float, default=0.2)
    parser.add_argument('--hid_feats', type=int, default=128)
    parser.add_argument('--out_feats', type=int, default=128)
    parser.add_argument('--layers' , type=int, default=2)
    parser.add_argument('--diff_lr', type=str2bool, default=False)

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--ft_lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--ft_epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--k', type=int, default=10000)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--DA_t', type=float, default=0.5)
    parser.add_argument('--lambda_eq', type=float, default=1.4)
    parser.add_argument('--lambda_u', type=float, default=0.8)
    args = parser.parse_args()
    return args
