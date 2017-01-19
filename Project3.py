
# coding: utf-8

# In[1]:

from __future__ import print_function

import numpy as np

import pandas as pd
import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


# In[41]:

train_file=pd.read_hdf("train.h5","train")
train_file=train_file.as_matrix()
y_train = train_file[:-10000,0]
X_train = train_file[:-10000,1:]
y_val = train_file[-10000:,0]
X_val = train_file[-10000:,1:]
X_train.astype(theano.config.floatX)
y_train = np.int32(y_train)

test_file=pd.read_hdf("test.h5","test")
test_file=test_file.as_matrix()
X_test = test_file


# In[42]:

print(X_train.dtype)
print(y_train,y_train.dtype)


# In[43]:

#y_train = y_train.reshape((y_train.shape[0],1))
#y_val = y_val.reshape((y_val.shape[0],1))
print(y_train.shape, y_val.shape)


# In[44]:

print(X_train.shape,y_train.shape)
print(X_val.shape,y_val.shape)
print(X_test.shape)


# In[45]:

test_file=pd.read_hdf("test.h5","test")
test_file=test_file.as_matrix()
X_test=test_file
#X_test=astype(theano.config.floatX)
print(X_test.shape, X_test.dtype)


# In[46]:

def build_mlp1():
    l_in = lasagne.layers.InputLayer(shape=(None,X_train.shape[1]))

    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=80,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=60,
            nonlinearity=lasagne.nonlinearities.rectify)

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=5,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    return l_out


# In[47]:

def build_mlp2():
    l_in = lasagne.layers.InputLayer(shape=(None,X_train.shape[1]))

    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    
    l_hid1 = lasagne.layers.DenseLayer(
        l_in_drop, num_units=250,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    l_hid2 = lasagne.layers.DenseLayer(
        l_hid1_drop, num_units=170,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    l_hid3 = lasagne.layers.DenseLayer(
        l_hid2_drop, num_units=110,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid3_drop = lasagne.layers.DropoutLayer(l_hid3, p=0.5)

    l_hid4 = lasagne.layers.DenseLayer(
        l_hid3_drop, num_units=90,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid4_drop = lasagne.layers.DropoutLayer(l_hid4, p=0.5)

    l_hid5 = lasagne.layers.DenseLayer(
        l_hid4_drop, num_units=70,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid5_drop = lasagne.layers.DropoutLayer(l_hid5, p=0.5)

    l_hid6 = lasagne.layers.DenseLayer(
        l_hid5_drop, num_units=50,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid6_drop = lasagne.layers.DropoutLayer(l_hid6, p=0.5)

    l_hid7 = lasagne.layers.DenseLayer(
        l_hid6_drop, num_units=110,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid7_drop = lasagne.layers.DropoutLayer(l_hid7, p=0.7)

    l_hid8 = lasagne.layers.DenseLayer(
        l_hid7_drop, num_units=50,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid8_drop = lasagne.layers.DropoutLayer(l_hid8, p=0.5)

    l_hid9 = lasagne.layers.DenseLayer(
        l_hid8_drop, num_units=20,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_hid9_drop = lasagne.layers.DropoutLayer(l_hid9, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_hid9_drop, num_units=5,
        nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

def build_mlp():
	l_in = lasagne.layers.InputLayer(shape=(None,X_train.shape[1]))

	l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.1)

	l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=90,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform())

	l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.2)

	l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=70,
            nonlinearity=lasagne.nonlinearities.tanh)

	l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.3)

	l_hid3 = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=40,
            nonlinearity=lasagne.nonlinearities.tanh)

	l_hid3_drop = lasagne.layers.DropoutLayer(l_hid3, p=0.5)

	l_hid4 = lasagne.layers.DenseLayer(
            l_hid3_drop, num_units=20,
            nonlinearity=lasagne.nonlinearities.tanh)

	l_hid4_drop = lasagne.layers.DropoutLayer(l_hid4, p=0.6)

	l_out = lasagne.layers.DenseLayer(
            l_hid4_drop, num_units=5,
            nonlinearity=lasagne.nonlinearities.softmax)

	return l_out


# In[48]:

input_var = T.matrix('inputs')
target_var = T.lvector('targets')

network = build_mlp2()
print(lasagne.layers.get_output(network, input_var))
print(lasagne.layers.get_all_params(network, trainable=True))


# In[50]:

prediction = lasagne.layers.get_output(network, input_var)
#print(lasagne.layers.get_output(network))
loss = T.mean(T.nnet.categorical_crossentropy(prediction, target_var))
params = lasagne.layers.get_all_params(network, trainable=True)
updates_sgd = lasagne.updates.sgd(loss, params, learning_rate=0.0001)
updates = lasagne.updates.apply_momentum(updates_sgd, params, momentum=0.9)
#print(updates)


# In[51]:

test_prediction = lasagne.layers.get_output(network, input_var, deterministic=True)
test_loss = T.mean(T.nnet.categorical_crossentropy(test_prediction, target_var))

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)


# In[52]:

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, allow_input_downcast=True, updates=updates)


# In[53]:

val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True, on_unused_input='ignore')

predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1),on_unused_input='ignore')
# In[1]:

num_epochs=50000
for epoch in range(num_epochs):
#        # In each epoch, we do a full pass over the training data:
        train_err = 0
        start_time = time.time()
        train_err = train_fn(X_train, y_train)

        val_err = 0
        val_acc = 0
        err, acc = val_fn(X_val, y_val)

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t:",train_err)

        print("  validation loss:\t\t:",err)
        print("  validation accuracy:\t\t: %",acc * 100)


# In[ ]:
Ypred = predict_fn(X_test)

print(Ypred)

my_sample_data = np.genfromtxt('sample.csv', delimiter=',',dtype=int)[1:,:] #withouth first line
Ypred=np.atleast_2d(Ypred)
my_sample_data[:,1:]=Ypred.T
#print(my_sample_data)
print(my_sample_data.shape)

header=['Id','y']
sample_File = pd.DataFrame(my_sample_data, columns=header)
sample_File.to_csv('submission.csv', index=False, header=True, sep=',')


# In[ ]:



