from math import log
from operator import itemgetter
import operator
def calcShannonEnt(dataSet):
    '''
    计算香农熵
    :param dataSet:
    :return:
    '''
    numEntires = len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]  #提取label
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntires  #所有类标签的发生频率计算类别出现的频率
        shannonEnt -=prob*log(prob,2)               #计算香农熵 熵越高，混合的数据越多。
    return shannonEnt
def splitDataSet(dataSet,axis,value):
    '''

    :param dataset:待划分的数据集
    :param axis:划分数据集的特征
    :param value:返回特征的值
    :return: retDataSet 根据axis分类后除特征值后的结果
    '''
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''

    :param dataSet:
例子 [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no'], [1, 1, 'maybe']]
    :return:信息增益最大的特征索引值
    '''
    numFeatures=len(dataSet[0])-1  #特征数量   len[[1, 1, 'yes']=3   num=2   减去label-'yes'
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList= [example[i] for example in dataSet]  #[1, 1, 'yes']——> 1,1,'yes'
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature
def majorityCnt(classList):
    '''
    函数说明:统计classList中出现此处最多的元素(类标签)
    :param classList: 类标签列表
    :return: sortedClassCount[0][0] - 出现此处最多的元素(类标签)
    '''
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
def createTree(dataSet,labels,featLabels):
    labels=labels[:]
    classList=[example[-1] for example in dataSet] #所有类别
    print('classList',classList)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0])==1 or len(labels)==0:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)   #从剩余所有特征中选取最优索引
    bestFeatLabel=labels[bestFeat]               #最优特征是什么
    featLabels.append(bestFeatLabel)
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])                        #删除选择出的最优特征
    featValues=[example[bestFeat] for example in dataSet]  #选择类别中的特征
    uniqueVals = set(featValues)
    print('1:',bestFeat,'2:',bestFeatLabel,'3:',featValues,'4',featLabels)
    for value in uniqueVals:
        subLabels=labels[:]  #为了确保每次递归选择特征时，原始labels不变。
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels,featLabels)
        #myTree {'no surfacing': {0: 'no'}}
        print('myTree',myTree)
    return myTree

def createDetaSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no'],[1,1,'maybe']]
    labels=['no surfacing','flippers']
    return dataSet,labels
'''
dataSet,labels=createDetaSet()
a=calcShannonEnt(dataSet)
print(dataSet,a)
feature0=splitDataSet(dataSet,0,0)
print('\n以元素第一个数字是否为0作区分',feature0)
feature1=splitDataSet(dataSet,0,1)
feature10=splitDataSet(dataSet,0,0)
print('\n以元素第一个数字是否为1作区分',feature1)
print('\n再以元素第二个数字是否为0作区分',feature10)
bestFeature=chooseBestFeatureToSplit(dataSet)
print(bestFeature)
'''
# dataSet,labels=createDetaSet()
# print('--',dataSet,labels)
# featLabels=[]
# myTree=createTree(dataSet,labels,featLabels)
#
# print('-------result-------',myTree)