##############################################################
##############################################################
# This is a script for training a svm model                  #
# Note:                                                      #
#   Feature file must be of the following form:              #
#       Line0: label of the features                         #
#       Col0: id of the examples                             #
#       Col1-n: value of features                            #
##############################################################
##############################################################

import numpy as np
import os
import csv
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC


X = np.genfromtxt(os.environ['FEATURES_FILE'], delimiter=',',dtype=None, encoding='utf-8')

X = np.delete(X,(0),axis=0)
names = X[:,0]
X = np.delete(X,(0),axis=1).astype(float)

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

size = len(X)

y = np.genfromtxt(os.environ['TARGETS_FILE'], delimiter=',',dtype=None, encoding='utf-8')
y = np.delete(y,(0),axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.5)

regr = SVC()
regr_cal = CalibratedClassifierCV(regr,  method='sigmoid')
regr_cal.fit(X_train, y_train)

dump(regr_cal, os.environ['MODEL_FILE'])

print('Train score =', regr_cal.score(X_train, y_train))

print('Test score =', regr_cal.score(X_test, y_test))