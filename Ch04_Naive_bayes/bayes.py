from numpy import *


def loadDataSet():
    """
    :return:初始化数组以及分类结果
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    """
    创建词汇表
    :param dataSet: 待分词数据集
    :return: 返回不重复的所有词
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    词集模型，每个词在文档出现一次
    根据词汇表，将句子转化为向量，即出现过的词数量+1
    :param vocabList: 词库
    :param inputSet:  输入句子
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  #  += 1 词袋模型，每个词可以出现多次
        else:
            print("the word: {} is not in my Vocabulary!".format(word))
    return returnVec


def trainNBO(trainMatrix, trainCategory):
    """

    :param trainMatrix: 待训练句向量矩阵
    :param trainCategory: 训练分类
    :return:
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pabusive = sum(trainCategory)/float(numTrainDocs)
    print("pabusive", pabusive)
    # 降低某个因子为0导致的影响
    p0num = ones(numWords)
    p1num = ones(numWords)
    p0denom = 2.0
    p1denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1num += trainMatrix[i]
            p1denom += sum(trainMatrix[i])
        else:
            p0num += trainMatrix[i]
            p0denom += sum(trainMatrix[i])
    # 防止下溢出，数字过小为0
    p1vect = log(p1num/p1denom)
    p0vect = log(p0num/p0denom)
    return p0vect, p1vect, pabusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    print("dataSet:", listOPosts, "\n", "分类结果：", listClasses)
    myVocabList = createVocabList(listOPosts)
    print("所有词展示：", myVocabList)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print("词向量矩阵", trainMat)
    p0V, p1V, pAb = trainNBO(array(trainMat), array(listClasses))
    print("训练结果：", p0V, p1V, pAb)
    testEntry = ["love", "my", "dalmation"]
    thisDoc1 = array(setOfWords2Vec(myVocabList, testEntry))
    print("测试词向量为：", thisDoc1)
    print(testEntry, "classified as: ", classifyNB(thisDoc1, p0V, p1V, pAb))
    testEntry = ["stupid", "garbage"]
    thisDoc2 = array(setOfWords2Vec(myVocabList, testEntry))
    print("测试词向量为：", thisDoc2)
    print(testEntry, "classified as:", classifyNB(thisDoc2, p0V, p1V, pAb))
    testEntry = ["stupid", "garbage","wsh"]
    thisDoc2 = array(setOfWords2Vec(myVocabList, testEntry))
    print("测试词向量为：", thisDoc2)
    print(testEntry, "classified as:", classifyNB(thisDoc2, p0V, p1V, pAb))


if __name__ == '__main__':

    # 数据处理测试
    # listOPosts, listClasses = loadDataSet()
    # myVocabList = createVocabList(listOPosts)
    # print(myVocabList)  # 不重复单词
    # print(lis tOPosts[0])
    # word1 = setOfWords2Vec(myVocabList, listOPosts[0])
    # print(word1)
    # word2 = setOfWords2Vec(myVocabList, listOPosts[3])
    # print(word2)

    #朴素bayes测试
    # listOPosts, listClasses = loadDataSet()
    # myVocabList = createVocabList(listOPosts)
    # trainMat = []
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print(trainMat)
    # p0V, p1V, pAb = trainNBO(trainMat, listClasses)
    # print(p0V, "\n", p1V, "\n", pAb)

    #测试算法
    testingNB()