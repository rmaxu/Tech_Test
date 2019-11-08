# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:12:32 2019

@author: rmuh
"""

#Algorithm Comparison

import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from sklearn import metrics

np.random.seed(0)

#load make_moons dataset
X, y = make_moons(n_samples=1500, noise=0.09)

#Scale the variables
X = StandardScaler().fit_transform(X)

#Scatter plot
plt.figure(figsize = (12,8))
plt.scatter(X[:,0], X[:,1], c=y)
plt.title('Original')
plt.show()

#------ K-means
t0 = time.time()
kmeans = cluster.KMeans(n_clusters=2).fit(X)
t1 = time.time()
print("K-means time :", t1-t0)

#Plot k-means results
plt.figure(figsize = (12,8))
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.title('K-means')
plt.show()

#----- Affinity Propagation
t0 = time.time()
ap = cluster.AffinityPropagation(preference=-300, damping = .75).fit(X)
t1 = time.time()
print("AP time :", t1-t0)

#Plot AP results
plt.figure(figsize = (12,8))
plt.scatter(X[:,0], X[:,1], c=ap.labels_)
plt.title('Affinity Propagation')
plt.show()

#------ Spectral Clustering
t0 = time.time()
sc = cluster.SpectralClustering(n_clusters=2, 
                                affinity="nearest_neighbors").fit(X)
t1 = time.time()
print("SC time :", t1-t0)

#Plot AP results
plt.figure(figsize = (12,8))
plt.scatter(X[:,0], X[:,1], c=sc.labels_)
plt.title('Spectral Clustering')
plt.show()


#------ DBSCAN
t0 = time.time()
dbscan = cluster.DBSCAN(eps=0.25, min_samples=7).fit(X)
t1 = time.time()
print("DBSCAN time :", t1-t0)

#Plot DBSCAN results
plt.figure(figsize = (12,8))
plt.scatter(X[:,0], X[:,1], c=dbscan.labels_)
plt.title('DBSCAN')
plt.show()

#------ Hierarchical (Agglomerative) Clustering
t0 = time.time()
hclust = cluster.AgglomerativeClustering(linkage='single').fit(X)
t1 = time.time()
print("Hierarchical Clustering time :", t1-t0)

#Plot Hierarchical Clustering results
plt.figure(figsize = (12,8))
plt.scatter(X[:,0], X[:,1], c=hclust.labels_)
plt.title('Hierarchical Clustering')
plt.show()

#------------------- Performance Evaluation --------------------------------

#Adjusted Rand index
#Rand Index measures the similarity of the two assignments
print("Adjusted Rand Index :")
print("K-means ", metrics.adjusted_rand_score(y, kmeans.labels_))
print("AP ", metrics.adjusted_rand_score(y, ap.labels_))
print("SC ", metrics.adjusted_rand_score(y, sc.labels_))
print("DBSCAN ", metrics.adjusted_rand_score(y, dbscan.labels_))
print("HClustering ", metrics.adjusted_rand_score(y, hclust.labels_))

#Mutual Information
#Measures the agreement of the two assignments

print("Adjusted Mutual Information :")
print("K-means ", metrics.adjusted_mutual_info_score(y, kmeans.labels_))
print("AP ", metrics.adjusted_mutual_info_score(y, ap.labels_))
print("SC ", metrics.adjusted_mutual_info_score(y, sc.labels_))
print("DBSCAN ", metrics.adjusted_mutual_info_score(y, dbscan.labels_))
print("HClustering ", metrics.adjusted_mutual_info_score(y, hclust.labels_))
