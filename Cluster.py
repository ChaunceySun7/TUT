print(__doc__)

from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans,DBSCAN, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import load_digits,fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


#读取手写数字辨识数据
def read_data_1():
    digits = load_digits()
    data = scale(digits.data)
    n_samples, n_features = data.shape
    n_digits = len(np.unique(digits.target))
    labels_true = digits.target

    print("n_digits: %d, \t n_samples %d, \t n_features %d"
          % (n_digits, n_samples, n_features))
    return data,labels_true,n_digits

#读取20newsgroups数据
def read_data_2():
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                 shuffle=True, random_state=42)
    print("%d documents" % len(dataset.data))
    print("%d categories" % len(dataset.target_names))
    print()

    labels_true = dataset.target
    true_k = np.unique(labels_true).shape[0]
    hasher = HashingVectorizer(stop_words='english', alternate_sign=False,
                               norm=None, binary=False)
    vectorizer = make_pipeline(hasher, TfidfTransformer())
    X = vectorizer.fit_transform(dataset.data)

    return X,labels_true,true_k

def score(labels_true,labels_pred):
    result_NMI = metrics.normalized_mutual_info_score(labels_true,labels_pred)
    result_Homo = metrics.homogeneity_score(labels_true,labels_pred)
    result_Com = metrics.completeness_score(labels_true,labels_pred)

    print("NMI score:",result_NMI)
    print("Homogeneity score:",result_Homo)
    print("Completeness score",result_Com)

def K_means(X,Y,n_clusters):
    print("k_means")
    label_pred = KMeans(init='k-means++',n_clusters=n_clusters,n_init=10).fit_predict(X)
    score(Y,label_pred)

def Affinity_Propagation(X,Y):
    print("Affinity_Propagation")
    label_pred = AffinityPropagation().fit_predict(X)
    score(Y,label_pred)

def Mean_Shift(X,Y):
    print("Mean_Shift")
    bandwidth = estimate_bandwidth(X,quantile=0.2,n_samples=500)
    label_pred = MeanShift(bandwidth=bandwidth,bin_seeding=True).fit_predict(X)
    score(Y,label_pred)

def Spectral_Clustering(X,Y,gamma,n_clusters):
    print("Spectral_Clustering")
    label_pred = SpectralClustering(n_clusters=n_clusters,gamma=gamma).fit_predict(X)
    score(Y,label_pred)

def Agglomerative_Clustering(X,Y,linkage,n_clusters):
    print("Agglomerative_Clustering: " + linkage)
    labels_pred = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage).fit_predict(X)
    score(Y,labels_pred)

def DBSCAN_(X,Y,eps,min_samples):
    print("DBSCAN")
    labels_pred = DBSCAN(eps=eps,min_samples=min_samples).fit_predict(X)
    score(Y,labels_pred)

def Gaussian_Mixture(X,Y,cov_type,n_components):
    print("Gaussian_Mixture: " + cov_type)
    gmm = GaussianMixture(n_components=n_components,covariance_type=cov_type).fit(X)
    labels_pred = gmm.predict(X)
    score(Y,labels_pred)

if __name__ == '__main__':
    #读取数据
    X1,Y1,n_digits= read_data_1()
    X2, Y2, true_k = read_data_2()
    linkages = ['ward', 'average', 'complete', 'single']
    cov_types = ['spherical','diag','tied','full']

    print("第一个数据集:")
    #K_means(X1,Y1,n_digits)
    #Affinity_Propagation(X1,Y1)
    #Mean_Shift(X1,Y1)
    #Spectral_Clustering(X1,Y1,gamma=0.06,n_clusters=n_digits)
    # for linkage in linkages:
    #     Agglomerative_Clustering(X1,Y1,linkage,n_digits)
    #DBSCAN_(X1,Y1,eps=0.3,min_samples=1)
    # for cov_type in cov_types:
    #     Gaussian_Mixture(X1,Y1,cov_type,n_digits)



    print("第二个数据集:")
    #K_means(X2,Y2,true_k)
    #Affinity_Propagation(X2, Y2)
    Mean_Shift(X2,Y2)  #有问题
    #Spectral_Clustering(X2,Y2,gamma=0.06,n_clusters=true_k)
    # for linkage in linkages:
    #     Agglomerative_Clustering(X2,Y2,linkage,true_k)  # 有问题
    #DBSCAN_(X2, Y2, eps=0.3, min_samples=1)
    # for cov_type in cov_types:
    #     Gaussian_Mixture(X2,Y2,cov_type,true_k)   #有问题






