# Homework1 of DM 
## 1.MNIST数据集
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
### 各类算法在MNIST数据集上的运行结果
  
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
  
  
  
