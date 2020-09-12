import numpy as np
import os
import csv
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


X = np.genfromtxt(os.environ['FEATURES_FILE'], delimiter=',',dtype=None, encoding='utf-8')

X = np.delete(X,(0),axis=0)
names = X[:,0]
X = np.delete(X,(0),axis=1).astype(float)


size = len(X)

y = np.genfromtxt(os.environ['TARGETS_FILE'], delimiter=',',dtype=None, encoding='utf-8')
y = np.delete(y,(0),axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
regr = LogisticRegression()
regr.fit(X_train, y_train)
dump(regr, os.environ['MODEL_FILE'])