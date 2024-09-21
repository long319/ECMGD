
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as  np

def draw_tsne(dataset_name, output_, labels):
    X_tsne = TSNE(n_components=2, learning_rate=100, random_state=42).fit_transform(output_.cpu().detach().numpy())
    plt.figure(figsize=(8, 6))
    # plt.title('Dataset : ' + dataset_name + '   (Label rate : 20 nodes per class)')
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels.cpu().detach().numpy(), s=8,cmap='rainbow')
    handles, _ = scatter.legend_elements(prop='colors')
    # labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    plt.axis('off')
    # plt.legend(handles, labels, loc='upper right')
    # plt.colorbar(ticks=range(10))
    plt.savefig('./results/svg/' + dataset_name + '.svg', format='svg', transparent=True)
    plt.show()
    
def permute_adj(affinity, labels, n_class):
    new_ind = []
    for i in range(n_class):
        ind = np.where(labels == i)[0].tolist()
        new_ind += ind
    return affinity[new_ind, :][:, new_ind]
# def visulization(P):
