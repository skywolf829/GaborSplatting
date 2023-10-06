import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def viz_PCA(data, color=None):
    # data: [N, M]
    pca = PCA(n_components=2)
    x = pca.fit_transform(data)
    plt.scatter(x[:,0], x[:,1], c=color, s=0.1)
    plt.show()


def viz_TSNE(data, color=None):
    # data: [N, M]
    x = TSNE(n_components=2, learning_rate='auto',
            init='random').fit_transform(data)
    plt.scatter(x[:,0], x[:,1], c=color, s=0.1)
    plt.show()