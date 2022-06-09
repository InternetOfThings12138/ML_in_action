import numpy as np
import os
import collections
def classify0(inX,dataSet,labels,k):

    # dataSetSize=dataSet.shape[0]#行数
    # # 在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
    # diffMat=np.tile(inX,(dataSetSize,1)) -dataSet #
    # sqDiffMat=diffMat**2
    # #行相加
    # sqDistances=sqDiffMat.sum(axis=1)
    # distance=sqDistances**0.5
    # #排序 从小到大
    # sortedDistIndex=distance.argsort()
    # #定一个记录类别次数的字典
    # classCount={}
    # for i in range(k):
    #     #取出前k个元素的类别
    #     voteLabel=labels[sortedDistIndex[i]]
    #     # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
    #     classCount[voteLabel]=classCount.get(voteLabel,0)+1
    # # key=operator.itemgetter(1)根据字典的值进行排序
    # # key=operator.itemgetter(0)根据字典的键进行排序
    # # reverse降序排序字典
    # sortedclassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # return sortedclassCount[0][0]


    #格式相同的直接处理方法 不需要np.tile()
    distance=np.sum((inX-dataSet)**2,axis=1)**0.5
    tmp=distance.argsort()[:k]
    k_labels=[labels[index] for index in tmp]
    label=collections.Counter(k_labels).most_common(1)[0][0]
    return label

def img2Vector(filename):
    '''
    
    将32*32转换为1*1024的向量——>为了使用之前的分类器。
    :param filename:
    :return:
    '''
    #创建1*1024向量
    returnVect=np.zeros((1,1024))
    #打开文件
    fr = open(filename)
    #按行读取 32行
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            #每行有32列
            returnVect[0,32*i+j]=(lineStr[j])
    return returnVect
def handwritingClassTest():
    hwLabels=[]
    trainingFileList=os.listdir('trainingDigits')  #训练集
    m=len(trainingFileList) #文件夹长度
    trainingMat=np.zeros((m,32*32))
    for i in range(m):
        train_fileNameStr=trainingFileList[i]
        #获取所分类数字标签
        classNumber = int(train_fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        #每个1*1024的数据存储在矩阵中
        trainingMat[i,:]=img2Vector(r'trainingDigits/%s'%(train_fileNameStr))
    testFileList=os.listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        test_fileNameStr=testFileList[i]
        classNumber = int(test_fileNameStr.split('_')[0])
        vectorUnderTest=img2Vector(r'testDigits/%s'%(test_fileNameStr))
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest))


if __name__ == '__main__':
	handwritingClassTest()