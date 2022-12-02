import json

from print_util import print_dict
from preproc_util import preprocess_dataset
from CONST import RAW_DATASET_PATH, COL_NAMES
from kneed import KneeLocator

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_clusters(X_preproc, y, cluster_labels, n_clusters):
    colors = {0:'pink', 1:'mediumpurple', 2:'cornflowerblue', \
              3:'mediumseagreen', 4:'skyblue', 5:'bisque', \
              6:'orange', 7:'silver', 8:'lightcoral', 9:'coral', \
              10:'dodgerblue', 11:'crimson', 12:'slategray', 13:'brown'}
    #pca = PCA(2)
    #df = pca.fit_transform(X_preproc)
 
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=3).fit_transform(X_preproc)
    

    for coordinate, label, cluster_label in zip(X_embedded, y, cluster_labels):
        if label == 1:
            m = "*"
        else:
            m = "s"
        plt.scatter(x=coordinate[0], y=coordinate[1], c=colors[cluster_label], marker=m)
    plt.show()

     
def create_clusters(X_preproc, y, n=None):
    def find_elbowpoint():
        sse = []
        for n_clusters in range(2, 20):
            kmeans_model = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
            kmeans_model.fit(X_preproc)
            sse.append(kmeans_model.inertia_)
    
        # Finding the elbow point = 14
        kl = KneeLocator(range(2, 20), sse, curve="convex", direction="decreasing").elbow
        return kl

    if n is None:
        n = find_elbowpoint()
    else:
        n = n
    print("elbow point=", n)
    kmeans_model = KMeans(init='k-means++', n_clusters=n, n_init=10)
    kmeans_model.fit(X_preproc)
    cluster_labels = kmeans_model.labels_
    return X_preproc, y, cluster_labels, n


def driven_func():
    X, y = preprocess_dataset(RAW_DATASET_PATH, COL_NAMES)

    x_1, y_1 = [], []
    for _x, _y in zip(X, y):
        if _y == 0:
            x_1.append(_x)
            y_1.append(_y)

    x_1 = np.array(x_1)
    y_1 = np.array(y_1)
    X_preproc, y, cluster_labels, n = create_clusters(x_1, y_1)
    plot_clusters(X_preproc, y, cluster_labels, n)

if __name__ == "__main__":
    driven_func()
