import numpy as np
from KNN import classify0
from Show import showdatas

def file2matrix(filename):
    """
    函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力
    Parameters:
    	filename - 文件名
    Returns:
    	returnMat - 特征矩阵
    	classLabelVector - 分类Label向量
    """
    #打开文件，指定编码。
    fr=open(filename,'r',encoding='utf-8')
    #读取文件所有内容  输出结果 ['40920\t8.326976\t0.953952\tlargeDoses\n',
    arrayOlines= fr.readlines()
    #针对有BOM（Byte Order Mark，字节顺序标记，出现在文本文件头部）的文本，去掉BOM
    arrayOlines[0] = arrayOlines[0].lstrip('\ufeff')
    #文件行数
    numberoflines=len(arrayOlines)
    #创建numpy二维数组 number行 3列
    returnMat = np.zeros((numberoflines,3))
    # 返回的标签向量
    classLabelVector=[]
    #行索引值
    index=0

    for line in arrayOlines:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line=line.strip()
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine=line.split('\t')
        returnMat[index,]=listFromLine[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力

        #datingTestSet中进行分类
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        #datingTestSet2已经按1，2，3分类好
        #classLabelVector.append(int(listFromLine[-1]))

        index+=1
    return returnMat,classLabelVector
def autoNorm(dataSet):
    '''
    函数说明:对数据进行归一化
    :param dataSet: 特征矩阵
    :return: normDataSet -归一化后的特征矩阵
            ranges -数据范围
            minVals -数据最小值
    '''
    minVals=dataSet.min(0) #列中选取最小值      [0.       0.       0.001156] 1*3
    maxVals=dataSet.max(0)   #[9.1273000e+04 2.0919349e+01 1.6955170e+00]
    ranges=maxVals-minVals
    # shape(dataSet)返回dataSet的矩阵行列数
    normDataSet=np.zeros(np.shape(dataSet))
    #返回dataSet的行数 (1000, 3)  m=1000
    m= dataSet.shape[0]
    #原始值减去最小值
    normDataSet = dataSet-np.tile(minVals,(m,1))  #在方向重复m次，行1次
    #除以最大和最小值的差，得到归一化数据列
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals
def datingClassTest(hoRatio,k):
    '''
    函数说明:分类器测试函数
    取百分之十的数据作为测试数据，检测分类器的正确性
    :return:打印输出
    '''
    filename= 'datingTestSet.txt'
    datingDataMat,datingLabels=file2matrix(filename)
    #取所有数据的10%作为测试集
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    #10%测试数据的个数
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集，后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels
                                     [numTestVecs:m],k)
        #print('分类结果:%s\t真实类别:%d'%(classifierResult,datingLabels[i]))
        if classifierResult!=datingLabels[i]:
            errorCount+=1.0
    print('错误率:%f%%'%(errorCount/float(numTestVecs)))
def classifyPerson():
    '''
    函数说明:通过输入一个人的三维特征,进行分类输出
    :return:
    '''
    # 输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    # 三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    # 打开的文件名
    filename = "datingTestSet.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    #数据显示
    showdatas(datingDataMat,datingLabels)
    # 训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 生成NumPy数组,测试数据
    inArr = np.array([ffMiles, precentTats, iceCream])
    #测试数据归一化
    norminArr=(inArr-minVals)/ranges
    #返回结果
    classifierResult=classify0(norminArr,normMat,datingLabels,3)
    # 打印结果
    print("你可能%s这个人" % (resultList[classifierResult - 1]))

if __name__ == '__main__':
    hoRatio=0.1
    k=4
    datingClassTest(hoRatio,k)
    classifyPerson()

#print(datingDataMat,datingLabels)
