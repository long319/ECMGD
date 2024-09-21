
import argparse
import torch
def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="Device: cuda:num or cpu")
    parser.add_argument("--path", type=str, default='D:\\code\\Dataset\\Multi-view\\', help="Path of datasets")
    parser.add_argument("--path_Iso", type=str, default='D:\\code\\Dataset\\HeteGraph\\', help="Path of datasets")
    parser.add_argument("--path_Miss", type=str, default='D:\\code\\Dataset\\IncompleMulti\\', help="Path of datasets")

    parser.add_argument("--dataset", type=str, default="BDGP", help="Name of datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")

    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.01, help='weight for residual link')
    parser.add_argument('--layers', type=int, default=2, help='weight for residual link')
    parser.add_argument('--Miss_rate', type=float, default=0, help='Missing rate')

    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

    parser.add_argument("--n_repeated", type=int, default=5, help="Number of repeated times. Default is 10.")
    parser.add_argument("--save_results", action='store_true', default=True, help="xx")
    parser.add_argument("--save_all", action='store_true', default=True, help="xx")
    parser.add_argument("--save_loss", action='store_true', default= True, help="xx")
    parser.add_argument("--save_ACC", action='store_true', default=True, help="xx")
    parser.add_argument("--save_F1", action='store_true', default=True, help="xx")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio of labeled samples")
    parser.add_argument("--Iso_ratio", type=float, default=0.2, help="Ratio of labeled samples")

    parser.add_argument("--num_epoch", type=int, default=250, help="Number of training epochs. Default is 200.")

    # parser.add_argument("--knns", type=int, default=10, help="Number of k nearest neighbors")
    # parser.add_argument("--common_neighbors", type=int, default=2,
    #                     help="Number of common neighbors (when using pruning strategy 2)")
    # parser.add_argument("--pr1", action='store_true', default=True, help="Using prunning strategy 1 or not")
    # parser.add_argument("--pr2", action='store_true', default=True, help="Using prunning strategy 2 or not")

    args = parser.parse_args()

    return args