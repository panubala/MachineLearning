
# coding: utf-8

# In[1]:

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
import pandas as pd


# In[95]:

#import our train data
my_data = np.genfromtxt('train.csv', delimiter=',')
my_data = my_data[1:,1:] #delete first row and column
Y = my_data[:,0]
X = my_data[:,1:]
#print(Y)
print(X.shape,Y.shape)


# In[96]:

#import our test data and delete the names
my_test_data = np.genfromtxt('test.csv', delimiter=',')
my_test_data = my_test_data[1:,1:]
Xtest = my_test_data
#print(Xtest)
print(Xtest.shape)


# In[22]:

#---Preprocessing--
poly=pre.PolynomialFeatures(degree=3)
X=poly.fit_transform(X)
Xtest=poly.fit_transform(Xtest)


# In[97]:

#our score function: error between predicted and given value
def score(gtruth, pred):
    diff = gtruth - pred
    return np.sqrt(np.mean(np.square(diff)))


# In[105]:

i=10
param_grid = {'alpha': np.linspace(2^-3, 2^3, 20)}

neg_scorefun = skmet.make_scorer(lambda x, y: -score(x,y))
#def f (x,y): return -score(x,y)
#neg_scorefun = skmet.make_scorer(f)


#-----------------------------------------------------------------------------------------
#Linear Ridge Regression
#grid_search = skgs.GridSearchCV(sklin.Ridge(), param_grid, scoring=neg_scorefun, cv=i)

#Kernelfunctions: 'linear', 'rbf' (gaussian), 'poly'

#Kernel Ridge
#grid_search = skgs.GridSearchCV(kr.KernelRidge(kernel='poly', gamma=0.1, degree=3), param_grid, scoring=neg_scorefun, cv=i, error_score=0)

#Support Vector Machines
#param_grid_SVM={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}
param_grid_SVM={'C': np.linspace(1.13,1.2,10), 'gamma': [0.002]} #specific for Project 2

#C-Support Vector Classification.
grid_search = skgs.GridSearchCV(svm.SVC(kernel='rbf'), scoring=neg_scorefun, param_grid=param_grid_SVM, cv=i)

#Epsilon-Support Vector Regression.
#grid_search = skgs.GridSearchCV(svm.SVR(kernel='rbf'), scoring=neg_scorefun, param_grid=param_grid_SVM, cv=i)

#Lasso
#grid_search=skgs.GridSearchCV(sklin.Lasso(), param_grid, scoring=neg_scorefun, cv=i)

#LassoCV
#grid_search=sklin.LassoCV(alphas=np.linspace(2^-3, 2^3, 20), cv=i)

print(X.shape,Y.shape)
grid_search.fit(X,Y)
best=grid_search.best_estimator_

print ('\n',best)
print('\ni =',i)
print('Best Score: ',-grid_search.best_score_)


# In[106]:

#here: type is int
Ypred = np.int_(best.predict(Xtest))
print(Ypred)


# In[107]:

#import our sample file (here: input type is int)
my_sample_data = np.genfromtxt('sample.csv', delimiter=',',dtype=int)[1:,:] #withouth first line
Ypred=np.atleast_2d(Ypred).T
my_sample_data[:,1:]=Ypred
#print(my_sample_data)
print(my_sample_data.shape)


# In[108]:

#Add column names and export the file
header=['Id','y']
sample_File = pd.DataFrame(my_sample_data, columns=header)
sample_File.to_csv('submission.csv', index=False, header=True, sep=',')


# In[109]:

print(sample_File)


# In[ ]:



