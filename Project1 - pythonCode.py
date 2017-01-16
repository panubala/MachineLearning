#%matplotlib inline
from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model as sklin
import sklearn.metrics as skmet
import sklearn.cross_validation as skcv
import sklearn.grid_search as skgs
import sklearn.kernel_ridge as kr
import sklearn.svm as svm
import sklearn.preprocessing as pre
from sklearn.preprocessing import FunctionTransformer

#import our train data
my_data = np.genfromtxt('train.csv', delimiter=',')
my_data = my_data[1:,1:] #delete first row and column
Y = my_data[:,0]
X = my_data[:,1:]
print(X.shape,Y.shape)

#import our test data and delete the names
my_test_data = np.genfromtxt('test.csv', delimiter=',')
my_test_data = my_test_data[1:,1:]
Xtest = my_test_data
Xtest.shape

poly=pre.PolynomialFeatures(degree=3)
X=poly.fit_transform(X)
Xtest=poly.fit_transform(Xtest)

#our score function, error between predicted and given value
def score(gtruth, pred):
    diff = gtruth - pred
    return np.sqrt(np.mean(np.square(diff)))

i=10    
param_grid = {'alpha': np.linspace(150, 190, 40)} #degree 3

def f (x,y): return -score(x,y)
#neg_scorefun = skmet.make_scorer(lambda x, y: score(x,y))
neg_scorefun = skmet.make_scorer(f)


#Ridge Regression
#withoout a kernel function
grid_search = skgs.GridSearchCV(sklin.Ridge(), param_grid, scoring=neg_scorefun, cv=i)

#with a kernel function
#grid_search = skgs.GridSearchCV(kr.KernelRidge(kernel='poly', gamma=0.1, degree=3), param_grid, scoring=neg_scorefun, cv=i, error_score=0)

print(grid_search.fit(X,Y))
best=grid_search.best_estimator_

print ('\n',best)
print('\ni =',i)
print('Best: ',-grid_search.best_score_)

Ypred = best.predict(Xtest)
np.savetxt('result_validate_5.csv',Ypred)