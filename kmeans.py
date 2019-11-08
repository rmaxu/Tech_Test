# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:46:31 2019

@author: rmuh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#K-means algorithm (Lloyd's algorithm)

class kmeans :
    def __init__(self, k = 3, tol = 0.0001, max_iter=300) :
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self, X) :
        n_samples = X.shape[0]
        #Randomly initialize k centers
        random_state = np.random.mtrand._rand  
        seeds = random_state.permutation(n_samples)[:self.k]
        self.centers = X[seeds]
        
        #Iterate until reach max_iter
        for i in range(self.max_iter) :
            #Auxiliary dictionary to store the points of each group
            groups = {}
            for j in range(self.k) :
                groups[j] = []
            
            #Cluster assignment step
            for entry in X :
                distances = [np.linalg.norm(entry - center) ** 2 
                             for center in self.centers]
                label = distances.index(min(distances))
                groups[label].append(entry)
            
            #Store the actual centers
            prev_centers = self.centers.copy()
            
            #Centers update step
            for l in groups :
                self.centers[l] = np.average(groups[l], axis = 0)
            
            #Compute the difference between the centers
            x = np.ravel(prev_centers - self.centers, order='K')
            dif = np.dot(x, x)
            
            #Stop if the difference between the centers is <= than tolerance
            if dif <= self.tol :        
                break
            
        #Store the array of labels
        self.labels_ = []
        for entry in X :
            distances = [np.linalg.norm(entry - center) ** 2 
                         for center in self.centers]
            label = distances.index(min(distances))
            self.labels_.append(label)
            

#load make_moons dataset
X, y = make_moons(n_samples=1500, noise=0.05)

#Scale the variables
X = StandardScaler().fit_transform(X)

#Scatter plot
#plt.scatter(X[:,0], X[:,1], c=y)
#plt.show()

#------ K-means
clustering = kmeans(k=3)
clustering.fit(X)


#Plot k-means results
plt.figure(figsize = (12,8))
plt.scatter(X[:,0], X[:,1], c=clustering.labels_)
plt.title('K-means')
plt.show()

from sklearn import cluster

#------ K-means (sklearn)
kmeans_sk = cluster.KMeans(n_clusters=3,init='random',
                        algorithm='full',n_init=1).fit(X)


#Plot k-means (sklearn) results
plt.figure(figsize = (12,8))
plt.scatter(X[:,0], X[:,1], c=kmeans_sk.labels_)
plt.title('K-means (sklearn)')
plt.show()

#------------------- Performance Evaluation --------------------------------

#Adjusted Rand index
print("Adjusted Rand Index :")
print("K-means (sklearn)", metrics.adjusted_rand_score(y, kmeans_sk.labels_))
print("K-means ", 
      metrics.adjusted_rand_score(y, clustering.labels_))

#Mutual Information
print("Adjusted Mutual Information :")
print("K-means (sklearn)", 
      metrics.adjusted_mutual_info_score(y, kmeans_sk.labels_))
print("K-means ", 
      metrics.adjusted_mutual_info_score(y, clustering.labels_))
