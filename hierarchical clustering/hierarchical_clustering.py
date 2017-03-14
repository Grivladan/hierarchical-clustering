from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

import numpy as np

np.set_printoptions(precision=5, suppress=True)

# generate two clusters: a with 100 points, b with 50:
np.random.seed(500)  # for repeatability of this tutorial
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[10])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[5])
X = np.concatenate((a, b),)
print(X)  # 150 samples with 2 dimensions
plt.scatter(X[:,0], X[:,1])

Z = linkage(X, 'ward')

c, coph_dists = cophenet(Z, pdist(X))
print(c)

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram ward method')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)

Z = linkage(X, 'single')

c, coph_dists = cophenet(Z, pdist(X))
print(c)

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram single linkage method')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)

Z = linkage(X, 'complete')

c, coph_dists = cophenet(Z, pdist(X))
print(c)

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram complete linkage method')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)

Z = linkage(X, 'average')

c, coph_dists = cophenet(Z, pdist(X))
print(c)

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram average linkage method')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()