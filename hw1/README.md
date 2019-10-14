# Homework1 of DM 
## 1.load_digits数据集
  sklearn.datasets.load_digits()

### 数据集数据
  n_digits :10
  
  n_samples :1797
  
  n_features :64
  

### 性能评价指标
#### Normalized Mutual Information (NMI) 归一化互信息
  metrics.normalized_mutual_info_score(labels_true, labels_pred)
  
  用于度量两个聚类结果的相似程度
#### Homogeneity 同质性
  metrics.homogeneity_score(labels_true, labels_pred)
  
  每个群集只包含单个类的成员
####  Completeness 完整性
  metrics.completeness_score(labels_true, labels_pred)
  
  给定类的所有成员都分配给同一个群集
### 各类算法在load_digits数据集上的运行结果
  
  |  算法   |  NMI  |  Homogeneity |  Completeness |
  |  ----   | ----  |   ----       |    -----      |
  | K-means  | 0.698 |  0.679  | 0.718 |
  | AffinityPropagation  | 0.655 |  0.932  |  0.460 |
  | Mean-Shift  | 0.048 |  0.009  |  0.257  |
  | Spectral Clustering  | 0.043 |  0.007  |  0.254  |
  | Agglomerative Clustering (Ward) | 0.800 |    0.758    |  0.836 |
  | Agglomerative Clustering (average)  | 0.041|  0.007 |  0.238  |
  | Agglomerative Clustering (complete)  | 0.065 |  0.017  |   0.250  |
  | Agglomerative Clustering (single)  | 0.040 |  0.006  |   0.276  |
  | DBSCAN  | 0.554 |  1.000  |   0.307  |
  | Gaussian Mixture(spherical)  | 0.662 |  0.645  |   0.679  |
  | Gaussian Mixture(diag)  | 0.620 |  0.560  |   0.685  |
  | Gaussian Mixture(tied)  | 0.710 |  0.673  |   0.750  |
  | Gaussian Mixture(full)  | 0.529 |  0.482  |   0.582  |
  
  
  ### 实验结论
    从以上的实验结果可以看出来，其中 Agglomerative Clustering (Ward)算法的综合结果最好，NMI的分数最高，
    
    同时完整性的结果也最高。
    
    DBSCAN算法的同质性分数最高，K-means算法、AffinityPropagation算法、Gaussian Mixture算法的结果比较适中
    
    其中Gaussian Mixture中tied方法的综合结果最好。
    
    当时有几个算法运行的结果分数很低，不知是算法的问题还是数据加载的问题，比如 Mean-Shift算法、
    
    Spectral Clustering等三个分数都很低。
    
    综上来看，Agglomerative Clustering (Ward)算法、K-means算法、AffinityPropagation算法
    
    以及Gaussian Mixture算法在第一个数据集的运行结果都不错。
    
## 2.20newsgroups数据集
   sklearn.datasets.fetch_20newsgroups()
   
### 数据集数据
   categories : 4
   
   categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space']
   
   documents : 3387
### 性能评价指标
#### Normalized Mutual Information (NMI) 归一化互信息
  metrics.normalized_mutual_info_score(labels_true, labels_pred)
  
  用于度量两个聚类结果的相似程度
#### Homogeneity 同质性
  metrics.homogeneity_score(labels_true, labels_pred)
  
  每个群集只包含单个类的成员
####  Completeness 完整性
  metrics.completeness_score(labels_true, labels_pred)
  
  给定类的所有成员都分配给同一个群集
    
### 各类算法在20newsgroups数据集上的运行结果

  |  算法   |  NMI  |  Homogeneity |  Completeness |
  |  ----   | ----  |   ----       |    -----      |
  | K-means  | 0.485 |  0.446  | 0.527 |
  | AffinityPropagation  | 0.411 |  0.885  |  0.191 |
  | Mean-Shift  | 0.0 |  0.0  |  1.0  |
  | Spectral Clustering  | 0.244 |  0.180  |  0.330  |
  | Agglomerative Clustering (Ward) | 0.624 |    0.595    |  0.654 |
  | Agglomerative Clustering (average)  | 0.613|  0.603 |  0.623  |
  | Agglomerative Clustering (complete)  | 0.573 |  0.541  |   0.607  |
  | Agglomerative Clustering (single)  | 0.012 |  0.001  |   0.141  |
  | DBSCAN  | 0.411 |  1.000  |   0.169  |
  | Gaussian Mixture(spherical)  | 0.456 |  0.409  |   0.508  |
  | Gaussian Mixture(diag)  | 0.622 |  0.587  |   0.659  |
  | Gaussian Mixture(tied)  | 0.562 |  0.535  |   0.591  |
  | Gaussian Mixture(full)  | 0.508 |  0.476  |   0.542  |
  
### 实验结论
    从以上的实验结果可以看出来，其中Agglomerative Clustering (Ward)算法的综合结果来看最好，NMI分数最高。
    
    DBSCAN算法的同质性分数最高，Gaussian Mixture(diag)算法的完整性分数最高。
    
    K-means算法、AffinityPropagation算法结果比较适中
    
    Agglomerative Clustering算法中除了single方法之外，其他三个方法的结果综合来看都不错，
    
    其中Ward方法是其中结果最好的
    
    Gaussian Mixture算法中四个方法结果都不错，其中diag方法综合来看是四个中最优的
    
    当时有几个算法运行的结果分数很低，不知是算法的问题还是数据加载的问题，比如 Mean-Shift算法、
    
    Agglomerative Clustering (single）算法、Spectral Clustering算法 等三个分数都很低。
    
    综上来看，Agglomerative Clustering 算法除掉 Agglomerative Clustering (single)、K-means算法
    
    、AffinityPropagation算法
    
    以及Gaussian Mixture算法在第二个数据集的运行结果都不错。
  
