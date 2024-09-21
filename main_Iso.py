


import os
import warnings
import random
import torch
import configparser
import numpy as np
from args import parameter_parser
from utils import tab_printer
from train import  train_IsoGraph
from scipy.io import savemat
import time
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parameter_parser()
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    args.device = device
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    # tab_printer(args)
    # for args.ratio in [0.2]:
    all_ACC = []
    all_MiF1 = []
    all_MaF1 = []
    all_TIME = []
    for i in range(args.n_repeated):
        torch.cuda.empty_cache()
        test_acc, test_micro_f1, test_macro_f1, cost_time, Loss_list, ACC_list, Micro_F1_list, Macro_F1_list = train_IsoGraph(
            args, device)
        all_ACC.append(test_acc)
        all_MiF1.append(test_micro_f1)
        all_MaF1.append(test_macro_f1)
        all_TIME.append(cost_time)

    print("-----------------------")
    print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    print("MaF1 : {:.2f} ({:.2f})".format(np.mean(all_MaF1) * 100, np.std(all_MaF1) * 100))
    print("MiF1: {:.2f} ({:.2f})".format(np.mean(all_MiF1) * 100, np.std(all_MiF1) * 100))
    print("Time : {:.2f} ({:.2f})".format(np.mean(all_TIME), np.std(all_TIME)))
    print("-----------------------")
    if args.save_results:
        experiment_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        results_direction = './results/' + args.dataset + '_results.txt'
        fp = open(results_direction, "a+", encoding="utf-8")
        fp.write(format(experiment_time))
        fp.write("\ndataset_name: {}\n".format(args.dataset))
        fp.write("alpha: {}  |  ".format(args.alpha))
        fp.write("hidden_channels: {}  |  ".format(args.hidden_channels))
        fp.write("ratio: {}  |  ".format(args.Iso_ratio))
        fp.write("epochs: {}  |  ".format(args.num_epoch))
        fp.write("lr: {}  |  ".format(args.lr))
        fp.write("wd: {}\n".format(args.weight_decay))
        # fp.write("lambda: {}  |  ".format(args.Lambda))
        # fp.write("alpha: {}\n".format(args.alpha))
        # fp.write("layer: {}\n".format(str_layers))
        fp.write("ACC:  {:.2f} ({:.2f})\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
        fp.write("MaF1 :  {:.2f} ({:.2f})\n".format(np.mean(all_MaF1) * 100, np.std(all_MaF1) * 100))
        fp.write("MiF1 :  {:.2f} ({:.2f})\n".format(np.mean(all_MiF1) * 100, np.std(all_MiF1) * 100))
        fp.write("Time:  {:.2f} ({:.2f})\n\n".format(np.mean(all_TIME), np.std(all_TIME)))
        fp.close()

    # if args.save_all:
    #     if args.save_loss:
    #         fp2 = open("results/loss/" + str(args.dataset) + ".txt", "a+", encoding="utf-8")
    #         fp2.seek(0)
    #         fp2.truncate()
    #         for i in range(len(Loss_list)):
    #             fp2.write(str(Loss_list[i]) + '\n')
    #         fp2.close()
    #
    #     if args.save_ACC:
    #         fp3 = open("results/MIF1/" + str(args.dataset) + ".txt", "a+", encoding="utf-8")
    #         fp3.seek(0)
    #         fp3.truncate()
    #         for i in range(len(Micro_F1_list)):
    #             fp3.write(str(Micro_F1_list[i]) + '\n')
    #         fp3.close()
    #
    #     if args.save_F1:
    #         fp4 = open("results/MAF1/" + str(args.dataset) + ".txt", "a+", encoding="utf-8")
    #         fp4.seek(0)
    #         fp4.truncate()
    #         for i in range(len(Macro_F1_list)):
    #             fp4.write(str(Macro_F1_list[i]) + '\n')
    #         fp4.close()