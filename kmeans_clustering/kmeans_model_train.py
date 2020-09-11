import numpy as np
import os
import csv

from sklearn.cluster import KMeans

cluster_size=10
centerfile=open('cluster_centers.csv','a')

X = np.genfromtxt(os.environ['FEATURES_FILE'], delimiter=',',dtype=None, encoding='utf-8')
X = np.delete(X,(0),axis=0)
names = X[:,1]
X = np.delete(X,(0,1),axis=1)

size = len(X)
print(size)

kmeans_cluster = KMeans(n_clusters=cluster_size)
kmeans_fit = kmeans_cluster.fit(X)
train_error = kmeans_fit.inertia_/float(size)
print(train_error)

for i in range(size):
    with open('labels.csv','a') as file:
        wr = csv.writer(file)
        row = [names[i]]
        for value in X[i]:
            row.append(float(value))
        row.append(int(kmeans_fit.labels_[i])+1)
        wr.writerow(row)

with open('cluster_centers.csv','a') as file:
    wr = csv.writer(file)
    for i in range(len(kmeans_fit.cluster_centers_)):
        row = [i+1]
        for value in kmeans_fit.cluster_centers_[i]:
            row.append(value)
        wr.writerow(row)