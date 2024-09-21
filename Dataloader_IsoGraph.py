
import scipy.io as sio
import scipy.sparse as ss
import random
import torch
import numpy as np

def load_Iso_data(args):
    """Load dataset whose file format is .mat
        Parameters
        ----------
        datasetPath: String
                    Path of dataset
        datasetName: String
                    Name of dataset
        ratio: float
                    train set ratio
        shuffle_seed: int
                    split train val test

        Ruturen
        ----------
        original_feature: ndarray
                        feature information of node(for classification)
        original_adj_list: list(ndarray)
                        n*n adj
        original_labels: ndarray
                        labels information of node(for classification)
        """
    data = sio.loadmat(args.path_Iso + args.dataset + '.mat')
    if  args.dataset == 'YELP':
        original_feature = data['X']
        original_adj_list = [data['BSB'], data['BLB'], data['BUB']]
        original_labels = data['Y']
    else:
        original_feature = data['X']
        original_adj_list = []
        for adj in data['adj'][0]:
            if ss.isspmatrix(adj):
                adj = adj.A
            original_adj_list.append(adj)
        original_labels = data['Y']
    # print(type(original_feature[0]),type(original_adj_list[0]),type(original_label),) print the type of returned information
    idx_train, idx_val, idx_test = generate_partition(original_labels, args.Iso_ratio, args.shuffle_seed)
    idx_train = torch.tensor(idx_train)
    idx_val = torch.tensor(idx_val)
    idx_test = torch.tensor(idx_test)
    return original_feature, original_adj_list, original_labels, idx_train, idx_val, idx_test

def generate_partition(labels, ratio, seed):
    """generate partition for train val test 6:2:2
            Parameters
            ----------
            labels: ndarray
                    dimension:(n,1)
            ratio: float
                    supervision rate
            seed: int
                    for index random
            Ruturen
            ----------
            indices_train, indices_val, indices_test: list(int)
            """
    sum = {}
    labels = labels.flatten()
    labels = labels - min(set(labels))
    each_class_num = count_each_class_num(labels)
    indices_list = []
    indices_train = []
    indices_val = []
    indices_test = []
    sum[-1] = 0
    for num in range(len(each_class_num)):
        sum[num] = sum[num-1] + each_class_num[num]
        indices_list.append(list(range(sum[num-1] , sum[num]))) # 该类的index列表
        if seed >= 0:
            random.seed(seed)
            random.shuffle(indices_list[num]) #一定要加一个seed
        indices_train += indices_list[num][0:int(each_class_num[num]*ratio)] # 前ratio 为train，然后val和test对半分。
        temp = (1.0 - ratio)/2.0 + ratio
        indices_val += indices_list[num][int(each_class_num[num]*ratio):int(each_class_num[num]*temp)]
        indices_test += indices_list[num][int(each_class_num[num]*temp):-1]
    return indices_train, indices_val, indices_test

def count_each_class_num(labels):
    '''Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict

if __name__ == '__main__':
    # datasets = ["ACM", "ALIBABA", "DBLP", "IMDB","YELP"]
    #
    datasets = ['IMDB']
    for dataset in datasets:
        print("Information of", dataset, ":")
        original_feature, original_adj_list, original_labels, idx_train, idx_val, idx_test = load_data("D:/code/Hetegraph/datasethete/", dataset, 0.2, 42)
        print(original_feature, original_adj_list, original_labels, idx_train, idx_val, idx_test )