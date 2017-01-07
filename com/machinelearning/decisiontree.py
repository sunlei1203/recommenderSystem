from math import log
import operator
def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels
# calcute shannonEnt for a dataSet, the last elment  of each list of dataSet is the class tag
def calcShannonEnt(dataSet):
    size = len(dataSet)
    classLabels = {}
    shannonEnt = 0.0
    for featVec in dataSet:
        feature = featVec[-1]
        if feature not in classLabels.keys():
            classLabels[feature] = 0
        classLabels[feature] += 1
    for key in classLabels:    # key is : each class tag
        prob = float(classLabels[key])/size
        ent = -(log(prob,2))
        shannonEnt += prob*ent
    return shannonEnt
# according to the feature(axis) and value(one uniq value of this feature) split the dataSet; warning: just one value (one feature may need call this function several times)
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        feature = featVec[axis]
        if feature == value:
            reducedVec = featVec[:axis]
            reducedVec.extend(featVec[axis+1:])
            retDataSet.append(reducedVec)
    return retDataSet
# def splitDataSet(dataSet, axis, value):
#     retDataSet = []
#     for featVec in dataSet:
#         if featVec[axis] == value:
#             reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
#             reducedFeatVec.extend(featVec[axis+1:])
#             retDataSet.append(reducedFeatVec)
#     return retDataSet
def chooseBestFeatureToSplit(dataSet):
    feature_num = len(dataSet[0])-1
    # print "feature num :"+str(feature_num)
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    best_feature = -1
    # feature_dict = {}
    # for feature in dataSet:
    #     for i in range(feature_num):
    #         feature_dict[i].append(feature[i])
    # for i in range(len(feature_dict)):
    #     feature_dict[i] = set(feature_dict[i])
    # for key in feature_dict:
    #     feature_shannon = 0
    #     for i in feature_dict[key]:
    #         retDataSet = splitDataSet(dataSet,key,i)
    #         shannonent = calcShannonEnt(retDataSet)
    #         feature_shannon += shannonent
    #     infoGain = feature_shannon - baseEntropy
    #     if bestInfoGain< infoGain:
    #         bestInfoGain = infoGain
    #         best_feature = key
    for i in range(feature_num):  #for each feature of the dataSet do something
        featureList = [example[i] for example in dataSet]
        uniqfeature = set(featureList)
        feature_shannon = 0.0
        for feature in uniqfeature:        # for each values(split dataSet into several branches) of this feature (split the dataSet) do something
            retDataSet = splitDataSet(dataSet,i,feature)
            # shannonent = calcShannonEnt(retDataSet)
            prob = len(retDataSet)/float(len(dataSet))   # the shannonEnt of each branch splited by each value of this feature is added with probability rather than simply add
            feature_shannon += prob * calcShannonEnt(retDataSet)
        infoGain =  baseEntropy - feature_shannon      #   !!!
        # print "baseEntropy :"+ str(baseEntropy)+"feature_shannon :"+ str(feature_shannon)+"infoGain :" +str(infoGain)
        if bestInfoGain < infoGain:
            bestInfoGain = infoGain
            best_feature = i
        # print "bestInfoGain :" +str(bestInfoGain)
    return best_feature

def majorityCnt(classList):
    classCount = {}
    for label in classList:
        if label not in classCount.keys(): classCount[label] = 0
        classCount[label] += 1
    sortedindex = sorted(classCount.iteritems(),key = operator.itemgetterm(1),reverse=True)
    return sortedindex[0][0]
# create decision tree using recursive(iterator)
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]  # the last element of all list of dataSet i.e all class tag of this dataSet form the 'classList',if all the class tag is th same,then return
    # print "classList :"+str(classList)
    # if there is only one class tag in dataSet
    if classList.count(classList[0]) == len(classList):return classList[0]
    # if len(set(classList))==1: return classList[0]

    # if there is no feature in dataSet,just class tag column ,return the class with more;(or the feature is ran out)
    if len(dataSet[0])== 1:
        return majorityCnt(classList)

    # the best feature for the current dataSet
    bestFeature = chooseBestFeatureToSplit(dataSet)
    # print "bestFeature: " +str(bestFeature)
    # the tags (or values) of this best feature
    # the tag of feature is mean : the name(meaning) of this feature??  for example: 'if it has flippers??'
    best_feature_label = labels[bestFeature]
    # myTree contains the best feature query sequence structure
    myTree = {best_feature_label:{}}    # {best_feature_tag1:{value1:{ a class tag or another feature tag},value2:{...}...}}
    # print "myTree :"+str(myTree)

    # del the feature that had been  experienced(used)
    del(labels[bestFeature])

    # get the values og this best feature
    feature_values = [example[bestFeature] for example in dataSet]
    uniq_values = set(feature_values)
    # print "uniq_values :"  + str(uniq_values)
    for value in uniq_values:
        subLabels = labels[:]
        # print "subLabels: " + str(subLabels)
        one_branch_dataSet = splitDataSet(dataSet, bestFeature,value)
        myTree[best_feature_label][value] = createTree(one_branch_dataSet,subLabels)
    return myTree
# classify a test sample
def classify(inputTree,featureLabels,testVec):
    # print featureLabels
    feature = str(inputTree.keys()[0])
    # print feature,type(feature)
    featureindex = featureLabels.index(feature)   #find the index of the feature
    nextdict = inputTree[feature]
    for key in nextdict.keys():
        if testVec[featureindex] == key:
            if type(nextdict[key]).__name__=='dict':
                classLabel = classify(nextdict[key],featureLabels,testVec)
            else: classLabel = nextdict[key]
    return classLabel
if __name__ == "__main__":
    dataSet,labels = createDataSet()
    old_labels = []
    old_labels.extend(labels[:])
    # print old_labels
    # print labels,type(labels)
    # print dataSet
    # dataSet[0][-1] = 'maybe'
    # print dataSet
    # shannonEnt = calcShannonEnt(dataSet)
    # print shannonEnt
    # mat = splitDataSet(dataSet,0,1)
    # print mat
    # feature = chooseBestFeatureToSplit(dataSet)
    # print feature
    myTree = createTree(dataSet,labels)
    # print old_labels
    print myTree    #{'no surfacing': {0: 'no', 1: {'no surfacing': {0: 'no', 1: 'yes'}}}}
    classLabel = classify(myTree,old_labels,[1,0])   #warning: annotate the del setence in def  createTree, or elst some elements of labels were deleted
    print classLabel
    # print old_labels