

import os
import warnings
import random
import torch
import configparser
import numpy as np
from args import parameter_parser
from utils import tab_printer
from train import train
from scipy.io import savemat
from plot_function import draw_tsne
import time
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parameter_parser()
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)

    all_ACC = []
    all_F1 = []
    all_TIME = []
    for i in range(args.n_repeated):
        ACC, P, R, F1, cost_time, Loss_list, ACC_list, F1_list = train(args, device)
        all_ACC.append(ACC)
        all_F1.append(F1)
        all_TIME.append(cost_time)
    print("====================")
    print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    print("F1 : {:.2f} ({:.2f})".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
    print("====================")
    if args.save_results:
        experiment_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        results_direction = './results/' + args.dataset + '_results.txt'
        fp = open(results_direction, "a+", encoding="utf-8")
        fp.write(format(experiment_time))
        fp.write("\ndataset_name: {}\n".format(args.dataset))
        fp.write("alpha: {}  |  ".format(args.alpha))
        fp.write("layers: {}  |  ".format(args.layers))
        fp.write("hidden_channels: {}  |  ".format(args.hidden_channels))
        fp.write("ratio: {}  |  ".format(args.ratio))
        fp.write("epochs: {}  |  ".format(args.num_epoch))
        fp.write("lr: {}  |  ".format(args.lr))
        fp.write("wd: {}\n".format(args.weight_decay))
        fp.write("Miss_rate: {}  |  ".format(args.Miss_rate))
        # fp.write("alpha: {}\n".format(args.alpha))
        # fp.write("layer: {}\n".format(str_layers))
        fp.write("ACC:  {:.2f} ({:.2f})\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
        fp.write("F1 :  {:.2f} ({:.2f})\n".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
        fp.write("Time:  {:.2f} ({:.2f})\n\n".format(np.mean(all_TIME), np.std(all_TIME)))
        fp.close()
