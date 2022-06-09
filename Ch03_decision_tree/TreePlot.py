import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from DecisionTree import *
import operator
import pickle
import pdb

def plotNode(nodeText,centerPt,parentPt,nodeType):
    '''
函数说明:绘制结点
    :param nodeText: 结点名
    :param centerPt: 文本位置
    :param parentPt: 标注的箭头位置
    :param nodeType: 结点格式
    :return:
    '''
    arrow_args=dict(arrowstyle='<-')                                           #定义箭头格式
    font = FontProperties(fname=r'c:\windows\fonts\simhei.ttf',size=14)        #设置中文字体 #绘制结点
    createPlot.ax1.annotate(nodeText,xy=parentPt,xycoords='axes fraction',\
                            xytext=centerPt,textcoords='axes fraction',\
                            va='center',ha='center',bbox=nodeType,arrowprops=arrow_args,FontProperties=font)
def plotMidText(cntrPt,parentPt,txtString):
    '''
    标注有向边属性
    :param cntrPt: 计算标注位置
    :param parentPt: 计算标注位置
    :param txtString: 标注内容
    :return:
    '''
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString,va='center',ha='center',rotation=30)
def getNumLeafs(myTree):
    '''
函数说明:获取决策树叶子结点的数目
    :param myTree:
    :return:
    '''
    numLeafs=0
    firstStr=next(iter(myTree))
    secondDict=myTree[firstStr]                     #获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict': #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs +=getNumLeafs(secondDict[key])
        else:
            numLeafs +=1
    return numLeafs
def getTreeDepth(myTree):
    '''
    获取决策树层数
    :param myTree:
    :return: maxDepth
    '''
    maxDepth=0
    firstStr=next(iter(myTree))
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1+getTreeDepth(secondDict[key])
        else:   thisDepth=1
        if thisDepth >maxDepth:maxDepth = thisDepth
    return maxDepth
def plotTree(myTree,parentPt,nodeTxt):
    decisionNode = dict(boxstyle='sawtooth',fc='0.8')
    leafNode = dict(boxstyle='round4',fc='0.8')
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)  # 中心位置
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 /plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff=1.0;
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
def classify(inputTree,featLabels,testVec):
    '''
    使用决策树进行分类
    :param inputTree: 决策树
    :param featLabels: 存储选择的最优特征标签
    :param testVec:  测试集
    :return: classLabel 分类结果
    '''
    firstStr = next(iter(inputTree))  #no surfacing
    #pdb.set_trace()
    print('first',inputTree,firstStr)
    secondDict = inputTree[firstStr]  #{0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}
    featIndex = featLabels.index(firstStr)   # 0  找到决策树特征索引
    print('featIndex',featIndex)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__ =='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else: classLabel = secondDict[key]
    return classLabel
def storeTree(inputTree,filename):
    with open(filename,'wb') as fw:
        pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):
    fr=open(filename,'rb')
    return pickle.load(fr)
if __name__ == '__main__':
    dataSet, labels = createDetaSet()
    featLabels = []
    myTree= createTree(dataSet, labels, featLabels)
    #storeTree(myTree,'classifierStorage.txt')  决策树保存
    #myTree=grabTree('classifierStorage.txt')   决策树读取
    #createPlot(myTree)   决策树可视化
    Feature1=input('no surfacing? 1.是/0.否')
    Feature2=input('flippers? 1.是/0.否')
    testVec=[int(Feature1),int(Feature2)]
    result=classify(myTree,featLabels,testVec)
    print(result)



#myTree={'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# a=getNumLeafs(myTree)
# b=getTreeDepth(myTree)
# print(a,b)
