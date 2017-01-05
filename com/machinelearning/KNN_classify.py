'''this is a handwriting recognition classifier achieved in  KNN algorithm'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
from os import listdir
#  classify for one test sample (structure: vector & matrix & ndarrays)
def classify0(inX,dataset,labels,k):     #inX: features of one test_sample,it is a array
    datasetsize = dataset.shape[0]         # dataset: trainSet that manifested into the form of array
    # print type(datasetsize)              # labels  : labels  i.e the class that every train sample belogs to
    diffMat = tile(inX,(datasetsize,1))-dataset
    # print type(diffMat)
    sqDifMat = diffMat**2
    sqDistance = sqDifMat.sum(axis = 1)
    # print sqDistance,type(sqDistance)   #[ 2.21  2.    0.    0.01] ,<type 'numpy.ndarray'>
    distances = sqDistance**0.5
    sortedDistIndicies = distances.argsort()    #[2 3 1 0] <type 'numpy.ndarray'>
    # print sortedDistIndicies,type(sortedDistIndicies)
    classCount = {}
    # print type(classCount)
    for i in xrange(k):
        voteLabel = labels[sortedDistIndicies[i]]
        # print voteLabel
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    # print classCount #{'A': 1, 'B': 2}
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
    # print sortedClassCount,type(sortedClassCount)   #[('B', 2), ('A', 1)] <type 'list'>
    return sortedClassCount[0][0]

# processing the raw text data, change a 32*32 txt file into a array[1,1024]  (one sample)
def img2vector(filename):
    fr = open(filename)
    returnVect = zeros((1,1024))
    for i in xrange(32):
        line = fr.readline()
        for j in xrange(32):
            returnVect[0,32*i+j] = int(line[j])
    return returnVect

# the main program
def handwritingClassTest():
    filedir = "MyProject/dataSet/KNN_data/trainDigits"
    # filedir = "F:\PycharmWorkspace/algorithms/dataset/t1"
    testdir = "MyProject/dataSet/KNN_data/testDigits"
    # testdir = "F:\PycharmWorkspace/algorithms/dataset/t2"
    trainset = listdir(filedir)
    # print trainset,type(trainset)    #<type list>    ['0_0.txt', '0_1.txt', '0_10.txt', '0_100.txt', '0_101.txt',...]
    size = len(trainset)
    xMat = zeros((size,1024))
    labels = []
    for i in range(size):                   #processing trainSet ==> matrix
        name = trainset[i]
        # print name,type(name)
        returnVect = img2vector(filedir+"/"+name)
        #   <==>   returnVect = img2vector(filedir+"/%s" % name)
        # print filedir+"/"+name
        xMat[i,:] = returnVect[:]
        #  <==>   xMat[i,:] = returnVect
        labels.append(int(name.strip().split('_')[0]))
    testset = listdir(testdir)
    testsize = len(testset)
    errcount = 0.0
    for i in range(testsize):                    #classify each test sample
        testname = testset[i]
        test_label = int(testname.strip().split('_')[0])
        testVect = img2vector(testdir+"/"+testname)
        classres = classify0(testVect,xMat,labels,3)
        # print classres
        print "the classifier came back with : %d,the real class is %d" %(classres,test_label)
        if(classres != test_label): errcount += 1.0
    error_rate = errcount/float(testsize)
    # print errcount,testsize
    print "the total number of errors is: %d" % errcount
    print "the error rate is : %f" % (error_rate)
if __name__ == "__main__":
    handwritingClassTest()