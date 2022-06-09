import numpy as np
import operator
import collections
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
def classify0(inX,dataSet,labels,k):
    '''

    :param inX:用于分类的输入向量(测试集)
    :param dataSet:输入的训练样本集(训练集)
    :param labels: 标签
    :param k: 前k个近邻
    :return:sortedClassCount[0][0] - 分类结果
    '''
    distance=np.sum((inX-dataSet)**2,axis=1)**0.5
    print(inX-dataSet,distance)
    tmp=distance.argsort()[:k] #从小到大排列前k个序号
    print(distance.argsort())
    k_labels=[labels[index] for index in tmp]  #找到序号对应的label
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label

group,label=createDataSet()
print('----',group,label)
real_label=classify0([0.5,0.4],group,label,1)
print(real_label)