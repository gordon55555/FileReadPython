import numpy as np
import sys
import cPickle, gzip, numpy
#import theano
#import theano.tensor as T
import types
import scipy
import copy
import os
import time
import numpy
import pylab
#from PIL import Image
#import theano
#import theano.tensor as T
f0 = open(r"/home/wuxupeng/PythonProjects/FullResult/data_batch_1", 'rb')

get_x=cPickle.load(f0)
#k=get_x.get_topological_view()
f0.close()

#a= get_x[2][0][1].reshape((3,32,32))
print get_x.readline()
'''
pylab.gray()
pylab.imshow(a[0])
pylab.show()
pylab.imshow(a[1])
pylab.show()
pylab.imshow(a[2])
pylab.show()
print get_x[2][1]

##############################33
trainx=get_x.get_data()[0]
trainy=get_x.get_data()[1]
norm_train_x = np.zeros((40000,32*32*3),dtype=trainx.dtype)
norm_valid_x = np.zeros((10000,32*32*3),dtype=trainx.dtype)
print type(norm_valid_x)

norm_train_y = np.zeros(40000,dtype=numpy.int64)
norm_valid_y = np.zeros(10000,dtype=numpy.int64)
norm_test_y = np.zeros(10000,dtype=numpy.int64)

for i in xrange(40000):
    for j in xrange (32*32*3):
        norm_train_x[i,j] = trainx[i,j]
    norm_train_y[i] = trainy[i]
for i in xrange(10000):
    for j in xrange (32*32*3):
        norm_valid_x[i,j] = trainx[i+40000,j]
    norm_valid_y[i] = trainy[i+40000]
f0 = open(r"/home/wuxupeng/PythonProjects/FullResult/pylearn2data/cifar10_preprocessed_test_GordonNorm.pkl", 'rb')
test=cPickle.load(f0)
f0.close()

for i in xrange(10000):
    norm_test_y[i] = test.get_data()[1][i]
result = ((norm_train_x,norm_train_y),(norm_valid_x,norm_valid_y),(test.get_data()[0],norm_test_y))
f1 = file(r'/home/wuxupeng/PythonProjects/FullResult/cifar10_0-1.pkl', 'wb')
cPickle.dump(result, f1)
f1.close()

######################################################################################
trainSet = np.zeros((k.shape[0],32*32*3),dtype=k.dtype)
t=0
for i in xrange(k.shape[0]):
   for j in xrange(3):
      for m in xrange(32):
         for n in xrange(32):
            trainSet[i][t] =k[i][m][n][j] 
            t = t+ 1    
   t=0



labels=[]
f1 = open(r"/home/wuxupeng/PythonProjects/FullResult/pylearn2data/cifar10/cifar-10-batches-py/data_batch_1", 'rb')
get_x1=cPickle.load(f1)
f1.close()
labels=[]
for i in xrange(10000):
      labels.append(get_x1['labels'][i])

f1 = open(r"/home/wuxupeng/PythonProjects/FullResult/pylearn2data/cifar10/cifar-10-batches-py/data_batch_2", 'rb')
get_x1=cPickle.load(f1)
f1.close()
labels=[]
for i in xrange(10000):
      labels.append(get_x1['labels'][i])

f1 = open(r"/home/wuxupeng/PythonProjects/FullResult/pylearn2data/cifar10/cifar-10-batches-py/data_batch_3", 'rb')
get_x1=cPickle.load(f1)
f1.close()
labels=[]
for i in xrange(10000):
      labels.append(get_x1['labels'][i])

f1 = open(r"/home/wuxupeng/PythonProjects/FullResult/pylearn2data/cifar10/cifar-10-batches-py/data_batch_4", 'rb')
get_x1=cPickle.load(f1)
f1.close()
labels=[]
for i in xrange(10000):
      labels.append(get_x1['labels'][i])

f1 = open(r"/home/wuxupeng/PythonProjects/FullResult/pylearn2data/cifar10/cifar-10-batches-py/data_batch_5", 'rb')
get_x1=cPickle.load(f1)
f1.close()
labels=[]
for i in xrange(10000):
      labels.append(get_x1['labels'][i])

a=[trainSet,labels]
f1 = file(r'/home/wuxupeng/PythonProjects/FullResult/pylearn2data/cifar10_preprocessed_train1_Standardize.pkl', 'wb')
cPickle.dump(a, f1)
f1.close()
'''
