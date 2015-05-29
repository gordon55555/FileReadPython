import cPickle
import os
import numpy
import pylab
import random
def part10():
    f = open(r'/home/wuxupeng/PythonProjects/FullResult/mnist.pkl','rb')
    data=cPickle.load(f)
    f.close()
    print 'The mnist consist by trainSet, ValidSet and testSet .'
    print 'The number is below respectively.'
    print data[0][0].shape, len(data[0][0]),len(data[1][0]),len(data[2][0]) 

    dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

    for x in data[0][1]:
       dict[x]=dict[x]+1
    print dict, sum(dict.values()), type(data[0][0])

    part10={}
    for x in xrange( 10):
       part10.setdefault(x,numpy.zeros((dict[x],784)))

    point= {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    for x in xrange(len(data[0][1])):
        i = data[0][1][x]
        part10[i][point[i]] =data[0][0][x]
        point[i] += 1
    
    print "part10 is a dictionary, just only have the train Set."
    '''
    a= part10[4][100].reshape((28,28))
    pylab.gray()
    pylab.imshow(a)
    pylab.show()
    '''
 
    f1 = file(r'/home/wuxupeng/PythonProjects/FullResult/mnistPart10.pkl', 'wb')
    cPickle.dump(part10, f1)
    f1.close()
    print "The pre-deal is finished!"

def match_data():
    print 'begin:Dynamic to make the match data.'
    f = open(r'/home/wuxupeng/PythonProjects/FullResult/mnist.pkl','rb')
    sourceData = cPickle.load(f)
    f.close()
    f1= open(r'/home/wuxupeng/PythonProjects/FullResult/mnistPart10.pkl', 'rb')    
    part10 = cPickle.load(f1)
    f1.close()

    dealTrainData = numpy.zeros(sourceData[0][0].shape)
    
    for i in xrange(50000):
        y = sourceData[0][1][i]
        rand = random.randint(0,part10[y].shape[0]-1)
        dealTrainData[i] = part10[y][rand]
         
    print 'end to make the match data'

def match_norm():
    print 'begin:Dynamic to make the match data.'
    f = open(r'/home/wuxupeng/PythonProjects/FullResult/mnist.pkl','rb')
    sourceData = cPickle.load(f)
    f.close()
    f1= open(r'/home/wuxupeng/PythonProjects/FullResult/mnistPart10.pkl', 'rb')
    part10 = cPickle.load(f1)
    f1.close()

    dealTrainData = numpy.zeros(sourceData[0][0].shape)
    part={}
    for x in xrange( 10):
       part.setdefault(x,numpy.zeros((1,784)))
    
    part[0][0] = part10[0][12]
    part[1][0] = part10[1][14]
    part[2][0] = part10[2][36]
    part[3][0] = part10[3][18]
    part[4][0] = part10[4][29]
    part[5][0] = part10[5][29]
    part[6][0] = part10[6][17]
    part[7][0] = part10[7][26]
    part[8][0] = part10[8][4]
    part[9][0] = part10[9][18]

    for i in xrange(50000):
        y = sourceData[0][1][i]
        dealTrainData[i] = part[y][0]

    print 'end to make the match data'
    return part
def chose_num():
    f = open(r'/home/wuxupeng/PythonProjects/FullResult/mnistPart10.pkl','rb')
    part10=cPickle.load(f)
    f.close()
    for i in xrange(1000):
        print 'picture num: '+str(i)
        a= part10[6][i].reshape((28,28))
        pylab.gray()
        pylab.imshow(a)
        pylab.show()
if __name__ == '__main__':
    chose_num()

