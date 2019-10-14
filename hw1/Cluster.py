print(__doc__)

from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans,DBSCAN, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import load_digits,fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import sys
import logging
from sklearn import manifold
from optparse import OptionParser

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

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

#读取20newsgroups数据
def read_data_2():

    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # parse commandline arguments
    op = OptionParser()
    op.add_option("--lsa",
                  dest="n_components", type="int",
                  help="Preprocess documents with latent semantic analysis.")
    op.add_option("--no-minibatch",
                  action="store_false", dest="minibatch", default=True,
                  help="Use ordinary k-means algorithm (in batch mode).")
    op.add_option("--no-idf",
                  action="store_false", dest="use_idf", default=True,
                  help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing",
                  action="store_true", default=False,
                  help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=10000,
                  help="Maximum number of features (dimensions)"
                       " to extract from text.")
    op.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print progress reports inside k-means algorithm.")

    print(__doc__)
    op.print_help()

    argv = [] if is_interactive() else sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    # Load some categories from the training set
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

    t0 = time()
    if opts.use_hashing:
        if opts.use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english', alternate_sign=False,
                                       norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features,
                                           stop_words='english',
                                           alternate_sign=False, norm='l2',
                                           binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                     min_df=2, stop_words='english',
                                     use_idf=opts.use_idf)
    X = vectorizer.fit_transform(dataset.data)

    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()

    if opts.n_components:
        print("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(opts.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        print("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        print()

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
    X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
    print("Agglomerative_Clustering: " + linkage)
    labels_pred = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage).fit_predict(X_red)
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
    K_means(X1,Y1,n_digits)
    # NMI score:0.698
    # Homogeneity score:0.679
    # Completeness score:0.718

    Affinity_Propagation(X1,Y1)
    # NMI score:0.655
    # Homogeneity score:0.932
    # Completeness score:0.460

    Mean_Shift(X1,Y1)
    # NMI score:0.048
    # Homogeneity score:0.009
    # Completeness score:0.257

    Spectral_Clustering(X1,Y1,gamma=0.06,n_clusters=n_digits)
    # NMI score:0.043
    # Homogeneity score:0.007
    # Completeness score:0.254

    for linkage in linkages:
        Agglomerative_Clustering(X1,Y1,linkage,n_digits)
    # NMI score:
    #         'ward' : 0.800
    #         'average' : 0.041
    #         'complete' : 0.065
    #         'single' : 0.040
    # Homogeneity score:
    #         'ward' : 0.758
    #         'average' : 0.007
    #         'complete' : 0.017
    #         'single' : 0.006
    # Completeness score:
    #         'ward' : 0.836
    #         'average' : 0.238
    #         'complete' : 0.250
    #         'single' : 0.276

    DBSCAN_(X1,Y1,eps=0.3,min_samples=1)
    # NMI score:0.554
    # Homogeneity score:1.000
    # Completeness score:0.307

    for cov_type in cov_types:
        Gaussian_Mixture(X1,Y1,cov_type,n_digits)
    # NMI score:
    #         'spherical' : 0.662
    #         'diag' : 0.620
    #         'tied' : 0.710
    #         'full' : 0.529
    # Homogeneity score:
    #         'spherical' : 0.645
    #         'diag' : 0.560
    #         'tied' : 0.673
    #         'full' : 0.482
    # Completeness score:
    #         'spherical' : 0.679
    #         'diag' : 0.685
    #         'tied' : 0.750
    #         'full' : 0.582

    print("第二个数据集:")
    K_means(X2,Y2,true_k)
    # NMI score:0.485
    # Homogeneity score:0.446
    # Completeness score:0.527

    Affinity_Propagation(X2, Y2)
    # NMI score:0.411
    # Homogeneity score:0.885
    # Completeness score:0.191

    Mean_Shift(X2.toarray(),Y2) 
    # NMI score:0.0
    # Homogeneity score:0.0
    # Completeness score:1.0

    Spectral_Clustering(X2,Y2,gamma=0.06,n_clusters=true_k)
    # NMI score:0.244
    # Homogeneity score:0.180
    # Completeness score:0.330

    for linkage in linkages:
         Agglomerative_Clustering(X2.toarray(),Y2,linkage,true_k)
    # NMI score:
    #         'ward' : 0.624
    #         'average' : 0.613
    #         'complete' : 0.573
    #         'single' : 0.012
    # Homogeneity score:
    #         'ward' : 0.595
    #         'average' : 0.603
    #         'complete' : 0.541
    #         'single' : 0.001
    # Completeness score:
    #         'ward' : 0.654
    #         'average' : 0.623
    #         'complete' : 0.607
    #         'single' : 0.141

    DBSCAN_(X2, Y2, eps=0.3, min_samples=1)
    # NMI score:0.411
    # Homogeneity score:1.000
    # Completeness score:0.169

    for cov_type in cov_types:
         Gaussian_Mixture(X2.toarray(),Y2,cov_type,true_k)
    # NMI score:
    #         'spherical' : 0.456
    #         'diag' : 0.622
    #         'tied' : 0.562
    #         'full' : 0.508
    # Homogeneity score:
    #         'spherical' : 0.409
    #         'diag' : 0.587
    #         'tied' : 0.535
    #         'full' : 0.476
    # Completeness score:
    #         'spherical' : 0.508
    #         'diag' : 0.659
    #         'tied' : 0.591
    #         'full' : 0.542






