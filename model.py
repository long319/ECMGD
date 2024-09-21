
import math,os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from torch_sparse import SparseTensor, matmul
# from torch_geometric.utils import degree
import torch
import sys


def full_attention_conv(qs, ks, vs, output_attn=False):
    qs = qs / torch.norm(qs, p=2) # [N, H, M]
    ks = ks / torch.norm(ks, p=2) # [L, H, M]
    # numerator
    attention_num = torch.sigmoid(torch.einsum("ij,kj->ik", qs, ks))  # [N, L, H]\

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    attention_normalizer = torch.einsum("nl,l->n", attention_num, all_ones)
    attention_normalizer = attention_normalizer.unsqueeze(0).repeat(ks.shape[0], 1)

    # compute attention and attentive aggregated results
    attention = attention_num / attention_normalizer
    attn_output = torch.einsum("ij,jk->ik", attention, vs)  # [N, H, D]
    if output_attn:
        return attn_output, attention_num
    else:
        return attn_output

class IntraDiffCov(nn.Module):
    def __init__(self, in_channels, out_channels, view):
        super(IntraDiffCov, self).__init__()
        self.Wklist = nn.ModuleList()
        self.Wqlist = nn.ModuleList()
        self.Wvlist = nn.ModuleList()
        for v in range(view):
            self.Wklist.append(nn.Linear(in_channels, out_channels))
            self.Wqlist.append(nn.Linear(in_channels, out_channels))
            self.Wvlist.append(nn.Linear(in_channels, out_channels))
        self.out_channels = out_channels
        self.views = view
    def reset_parameters(self):
        for Wk in self.Wklist:
            Wk.reset_parameters()
        for Wq in self.Wqlist:
            Wq.reset_parameters()
        for Wv in self.Wvlist:
            Wv.reset_parameters()

    def forward(self, latent_feature):
        # feature transformation
        intra_view = []
        attention_view = []
        for v in range(self.views):
            Key = self.Wklist[v](latent_feature[v,:,:])
            Query = self.Wqlist[v](latent_feature[v,:,:])
            Value = self.Wvlist[v](latent_feature[v,:,:])
            attention_output = full_attention_conv(Query,Key,Value) # [N, H, D]
            intra_view.append(attention_output)
            # attention_view.append(attention)
        intra_view = torch.stack(intra_view, dim=0)
        return intra_view

class InterDiffCov(nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(InterDiffCov, self).__init__()
        self.args = args
        self.WQ = nn.Linear(in_channels, out_channels)
        self.WK = nn.Linear(in_channels, out_channels)
        self.WV = nn.Linear(in_channels, out_channels)

    def projection_p(self, P):
        dim = P.shape[0]
        if self.args.dataset in ['ACM','DBLP','IMDB','amazon','ALIBABA','YELP']:
            device = str(self.args.device)
        else:
            device ='cuda:' + self.args.device
        one = torch.ones(dim, 1).to(device)

        relu = torch.nn.ReLU()
        '''
        Projection
        '''
        P_1 = relu(P)

        support_1 = torch.mm(torch.mm(P_1, one) - one, one.t()) / dim
        P_2 = P_1 - support_1

        support_2 = torch.mm(one, torch.mm(one.t(), P_2) - one.t()) / dim
        P_3 = P_2 - support_2

        return P_3

    def softmax_projection(self, x):
        sm_0 = torch.nn.Softmax(dim=0)
        sm_1 = torch.nn.Softmax(dim=1)

        x_0 = sm_0(x)
        x_1 = sm_1(x)

        proj_x = (x_0 + x_1) / 2
        return proj_x

    def multi_projection(self, x):
        proj_x = self.softmax_projection(x)
        for i in range(10):
            proj_x = self.projection_p(proj_x)

        return proj_x

    def forward(self, latent_feature):
        Q = self.WQ(latent_feature)
        K = self.WK(latent_feature)
        V = self.WV(latent_feature)
        Q = Q / torch.norm(Q, p='fro', dim=(1, 2), keepdim=True)
        K = K / torch.norm(Q, p='fro', dim=(1, 2), keepdim=True)
        V = V / torch.norm(Q, p='fro', dim=(1, 2), keepdim=True)

        P = torch.sigmoid(torch.einsum('ivd,jvd->ij', Q, K))
        P = self.multi_projection(P)
        final = torch.einsum('ij,ivd->jvd', P, V)
        return final

class ECMGD(nn.Module):
    '''
    CrossFormer model class
    x: input node features [N, Dv]
    return y_hat predicted logits [N, C]
    '''
    def __init__(self, in_channels_list, hidden_channels, out_channels, view, args):
        super(ECMGD, self).__init__()
        self.view = len(in_channels_list)
        self.Intraconvs = nn.ModuleList()
        self.Intercovs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for v in range(self.view):
            self.fcs.append(nn.Linear(in_channels_list[v], hidden_channels))
        # self.alpha = args.alpha
        self.alpha = nn.Parameter(torch.FloatTensor([-3]), requires_grad=True)
        self.layers = args.layers
        for i in range(args.layers):
            self.Intraconvs.append(IntraDiffCov(hidden_channels, hidden_channels, view))
            self.Intercovs.append(InterDiffCov(args, hidden_channels, hidden_channels))
        self.fcs.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = args.dropout
        self.activation = F.relu


    def reset_parameters(self):
        for conv in self.Intraconvs:
            conv.reset_parameters()
        for conv in self.Interconvs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x_list):
        # input MLP layer
        alpha = F.sigmoid(self.alpha)
        x_latent = []
        for v in range(self.view):
            x_latent.append(self.fcs[v](x_list[v]))
        x_latent = torch.stack(x_latent, dim=0)

        for i in range(self.layers):
            Intra = self.Intraconvs[i](x_latent)
            Inter = self.Intercovs[i](x_latent)
            x_latent = (1 - 2 * alpha) * x_latent + alpha * Intra + alpha * Inter


        final = torch.sum(x_latent, dim=0)
        # output MLP layer
        x_out = self.fcs[-1](final)
        return x_out

