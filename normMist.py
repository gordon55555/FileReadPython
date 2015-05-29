
import numpy as np
import sys
import cPickle, gzip, numpy
import theano
import theano.tensor as T
import types
import threading,time
import scipy
import copy
#import theano
#import theano.tensor as T

import numpy
#import pylab
#from PIL import Image 
import os
import math
f=open(r"/home/wuxupeng/PythonProjects/FullResult/mnist.pkl", 'rb')
train1_set, valid1_set, test1_set=cPickle.load(f)
f.close()
train_set_x = train1_set[0]
train_set_y = train1_set[1]
print type(train_set_y)
print type(train_set_y[0])
'''
valid_set_x = valid1_set[0]
valid_set_y = valid1_set[1]

test_set_x = test1_set[0]
test_set_y = test1_set[1]

#print train_set_x[100]

print "train"
average_train = numpy.mean(train_set_x,axis=0)
minus_train = train_set_x - average_train

norm_train = train_set_x - average_train

for i in xrange(minus_train.shape[0]):
    for j in xrange(minus_train.shape[1]):
        minus_train[i,j] = minus_train[i,j] **2
sum_train = numpy.sum(minus_train, axis=0)
sqre_train = numpy.sqrt(sum_train)
fai_train = (1.0/float(minus_train.shape[0]) ) * sqre_train
for i in xrange(norm_train.shape[0]):
    for j in xrange(norm_train.shape[1]):
        if(fai_train[j] == 0.0):
            norm_train[i,j] = 0.0
        else:
            norm_train[i,j] = norm_train[i,j] / fai_train[j]
print norm_train

print "vaild"
average_valid = numpy.mean(valid_set_x,axis=0)
minus_valid = valid_set_x - average_valid
norm_valid = valid_set_x - average_valid
for i in xrange(minus_valid.shape[0]):
    for j in xrange(minus_valid.shape[1]):
        minus_valid[i,j] = minus_valid[i,j] **2
sum_valid = numpy.sum(minus_valid, axis=0)
sqre_valid = numpy.sqrt(sum_valid)
fai_valid = (1.0/float(minus_valid.shape[0]) ) * sqre_valid
for i in xrange(norm_valid.shape[0]):
    for j in xrange(norm_valid.shape[1]):
        if(fai_valid[j] == 0.0):
            norm_valid[i,j] = 0.0
        else:
            norm_valid[i,j] = norm_valid[i,j] / fai_valid[j]

        
print "test"
average_test = numpy.mean(test_set_x,axis=0)
minus_test = test_set_x - average_test
norm_test = test_set_x - average_test
for i in xrange(minus_test.shape[0]):
    for j in xrange(minus_test.shape[1]):
        minus_test[i,j] = minus_test[i,j] **2
sum_test = numpy.sum(minus_test, axis=0)
sqre_test = numpy.sqrt(sum_test)
fai_test = (1.0/float(minus_test.shape[0]) ) * sqre_test
for i in xrange(norm_test.shape[0]):
    for j in xrange(norm_test.shape[1]):
        if(fai_test[j] == 0.0):
            norm_test[i,j] = 0.0
        else:
            norm_test[i,j] = norm_test[i,j] / fai_test[j]


result = ((norm_train,train_set_y),(norm_valid,valid_set_y),(norm_test,test_set_y))
f1 = file(r'/home/wuxupeng/PythonProjects/FullResult/norm_mnist.pkl', 'wb')
cPickle.dump(result, f1)
f1.close()
'''
