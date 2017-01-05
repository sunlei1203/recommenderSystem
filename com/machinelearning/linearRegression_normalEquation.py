'''dataSet form：<1.0   x1  y>
using norm equation achieve linear regression
dataset : linearregression/normal_0.txt
'''
from numpy import *
import matplotlib.pyplot as plt
def loadData(filename):
    feature_num = len(open(filename).readline().split('\t'))-1
    x_input_mat = []
    y_label_mat = []
    for line in open(filename):
        temp_x = []
        para = line.strip().split('\t')
        for i in xrange(feature_num):
            temp_x.append(float(para[i]))
        x_input_mat.append(temp_x)
        y_label_mat.append(float(para[-1]))
    return x_input_mat,y_label_mat
def calculate(x,y):
    xMat = mat(x)
    yMat = mat(y).T
    xTx = xMat.T*xMat
    if linalg.det(xTx)==0.0:
        print "the matrix is sigular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

if __name__ =="__main__":
    trainDir = "F:\PycharmWorkspace/MyProject/dataSet/linearregression/normal_0.txt"
    x_mat,y_mat = loadData(trainDir)
    ws = calculate(x_mat,y_mat)
    print ws
    xMat = mat(x_mat)
    yMat = mat(y_mat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()
