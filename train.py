

import warnings
import time
import random
from plot_function import permute_adj
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from args import parameter_parser
from utils import tab_printer, get_evaluation_results, compute_renormalized_adj
from Dataloader import load_data
from Dataloader_IsoGraph import load_Iso_data
from torch.autograd import Variable
from model import ECMGD
import scipy.sparse as ss
from plot_function import draw_tsne
import scipy.sparse as sp
from sklearn.metrics import f1_score
import seaborn as sns
def train(args, device):
    feature_list, labels, idx_labeled, idx_unlabeled = load_data(args, device)
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    N = feature_list[0].shape[0]
    num_view = len(feature_list)
    # print(N,num_view,num_classes)
    input_dims = []
    for i in range(num_view): # multiview data { data includes features and ... }
        input_dims.append(feature_list[i].shape[1])

    model = ECMGD(input_dims, args.hidden_channels, num_classes, num_view, args).to(device)
    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Best_Acc = 0.
    Loss_list = []
    ACC_list = []
    F1_list = []
    begin_time = time.time()
    with tqdm(total=args.num_epoch, desc="training", position=0) as pbar:
        for epoch in range(args.num_epoch):
            #encoder
            model.train()
            output = model(feature_list)
            output = F.log_softmax(output, dim=1)
            optimizer.zero_grad()
            loss = loss_function(output[idx_labeled], labels[idx_labeled])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                output = model(feature_list)
                pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
                ACC, _, _, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])
                if ACC > Best_Acc:
                    Best_Acc = ACC
                    bestre = output
                pbar.set_postfix({'Loss_ce': '{:.6f}'.format(loss.item()), 'ACC': '{:.2f}'.format(ACC * 100)})
                pbar.update(1)
                Loss_list.append(float(loss.item()))
                ACC_list.append(ACC)
                F1_list.append(F1)

    # draw_tsne(args.dataset, output, labels)
    cost_time = time.time() - begin_time
    model.eval()
    output = model(feature_list)
    # Aff = permute_adj(S[0].cpu().detach().numpy(), labels.cpu().detach().numpy(), 10)
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(S.cpu().detach().numpy(), cmap=plt.cm.bwr, annot=False, fmt=".2f", xticklabels=False, yticklabels=False, square=True, cbar=True,vmax=0.23,vmin=0.28)
    # plt.savefig('D:\\latex_appendix\\ACMM\\PMatrix\\'+args.dataset+'_Viewall_'+str(args.num_epoch)+'.svg', format='svg', dpi=600, bbox_inches='tight')

    # plt.savefig('D:\\latex_appendix\\ACMM\\PMatrix\\'+args.dataset+'_View_'+ str(k)+'_'+str(args.num_epoch)+'.jpg', format='jpg', bbox_inches='tight')
    print("Evaluating the model")
    pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
    ACC, P, R, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])
    print("------------------------")
    print("ratio = ",args.ratio)
    print("ACC:   {:.2f}".format(ACC * 100))
    print("F1 :   {:.2f}".format(F1 * 100))
    print("------------------------")

    return ACC, P, R, F1, cost_time, Loss_list, ACC_list, F1_list


def train_IsoGraph(args, device):
    (features,
        g,
        labels,
        train_mask,
        val_mask,
        test_mask,
    ) = load_Iso_data(args)

    # if hasattr(torch, "BoolTensor"):
    #     train_mask = train_mask.bool()
    #     val_mask = val_mask.bool()
    #     test_mask = test_mask.bool()
    num_classes = len(np.unique(labels))
    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels.flatten()-1).long().to(device)

    train_mask =  train_mask.to(device)
    val_mask =val_mask.to(device)
    test_mask = test_mask.to(device)
    N = features.shape[0]
    # input_dim = features.shape[1]
    num_metapath = len(g)
    for i in range(num_metapath):
        g[i] = ss.coo_matrix(g[i])
        g[i] = g[i] + g[i].T.multiply( g[i].T > g[i]) - g[i].multiply( g[i].T > g[i])
        adj_ = ss.eye(g[i].shape[0]) + g[i]
        rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
        print("mean_degree:", rowsum.mean())
        degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
        # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
        g[i] = (adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)).todense()
        g[i] = torch.from_numpy(g[i]).float().to(device)
    #pre process
    feature_list = []
    input_dims = []

    for i in range(num_metapath):
        # Asqure = torch.spmm(g[i],g[i])
        Asqure = g[i]
        representation = torch.spmm(Asqure,features)
        feature_list.append(representation)
        input_dims.append(representation.shape[1])

    #MODEL
    model = ECMGD(input_dims, args.hidden_channels, num_classes, num_metapath, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    loss_function = torch.nn.NLLLoss()

    Loss_list = []
    ACC_list = []
    Micro_F1_list = []
    Macro_F1_list = []
    best_val_acc = 0.0
    begin_time = time.time()

    with tqdm(total=args.num_epoch, desc="training",position=0) as pbar:
        for epoch in range(args.num_epoch):
            result = model(feature_list)
            output = F.log_softmax(result, dim=1)
            loss = loss_function(output[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                pred_labels = torch.argmax(output, 1)
                # val_loss = loss_function(output[val_mask], labels[val_mask])
                val_acc, val_micro_f1, val_macro_f1 = score(pred_labels[val_mask], labels[val_mask])
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_output = output
                pbar.set_postfix({'Lossce': '{:.6f}'.format((loss).item()),
                                  'Val_acc': '{:.2f}'.format(val_acc * 100),
                                  'Val_macro_f1': '{:.4f}'.format(val_macro_f1 * 100),
                                  'Val_micro_f1': '{:.2f}'.format(val_micro_f1 * 100)
                                  })
                pbar.update(1)
                Loss_list.append(float((loss).item()))
                ACC_list.append(val_acc)
                Micro_F1_list.append(val_micro_f1)
                Macro_F1_list.append(val_macro_f1)
    cost_time = time.time() - begin_time
    model.eval()
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(S.cpu().detach().numpy(), cmap=plt.cm.bwr, annot=False, fmt=".2f", xticklabels=False, yticklabels=False, square=True, cbar=True,vmax=0.32,vmin=0.34)
    # plt.savefig('D:\\latex_appendix\\ACMM\\PMatrix\\'+args.dataset+'_Viewall_'+str(args.num_epoch)+'.svg', format='svg', dpi=600, bbox_inches='tight')
    # print(output.shape)
    # z = model(features, g)
    print("Evaluating the model")
    pred_labels = torch.argmax(best_output, 1)
    test_acc, test_micro_f1, test_macro_f1 = score(pred_labels[test_mask], labels[test_mask])

    print("------------------------")
    print("ratio: {:.2f}".format(args.Iso_ratio))
    # print("Layer:  {}".format(args.num_layers))
    print("hidden_channels:   {}".format(args.hidden_channels))
    print("ACC:   {:.2f}".format(test_acc * 100))
    print("Macro_f1 :   {:.2f}".format(test_macro_f1 * 100))
    print("Micro_f1 :   {:.2f}".format(test_micro_f1 * 100))
    print("------------------------")

    return test_acc, test_micro_f1, test_macro_f1, cost_time, Loss_list, ACC_list, Micro_F1_list, Macro_F1_list

def score(prediction, labels):
    prediction = prediction.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

def evaluateT(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1