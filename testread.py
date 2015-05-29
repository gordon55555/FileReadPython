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
#import pylab
from PIL import Image
import theano
import theano.tensor as T

'''
get_x=cPickle.load(f0)
k=get_x.get_topological_view()
f0.close()
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
f1 = file(r'/home/wuxupeng/PythonProjects/FullResult/pylearn2data/train5_GordonNorm.pkl', 'wb')
cPickle.dump(a, f1)
f1.close()
'''
