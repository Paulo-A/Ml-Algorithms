##############################################################
##############################################################
# This is a script for cluster size for the kmeans algorithm #
# Note:                                                      #
#   Feature file must be of the following form:              #
#       Line0: label of the features                         #
#       Col0: id of the examples                             #
#       Col1-n: value of features                            #
##############################################################
##############################################################
import numpy as np
import os

from sklearn.cluster import KMeans


X = np.genfromtxt(os.environ['FEATURES_FILE'], delimiter=',',dtype=None, encoding='utf-8')
X = np.delete(X,(0),axis=0)
names = X[:,0]
X = np.delete(X,(0),axis=1)

size = len(X)

file=open('training_errors.csv','a')

cluster_size = 2
while cluster_size<=60:
    kmeans_cluster = KMeans(n_clusters=cluster_size)
    kmeans_fit = kmeans_cluster.fit(X)
    train_error = kmeans_fit.inertia_/float(size)

    print(train_error,cluster_size)
    np.savetxt(
        file,
        np.c_[train_error,cluster_size],
        delimiter=","
    )
    cluster_size+=1

train_error = np.delete(train_error,(0))
file.close()