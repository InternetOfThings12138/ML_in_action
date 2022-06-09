from DecisionTree import *
from TreePlot import *
glass=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in glass.readlines()]
labels=[example[-1] for example in lenses]
lensesFeature=['age','prescript','astigmatic','tearRate']
featLabels=[]
lensesTree=createTree(lenses,lensesFeature,featLabels)
storeTree(lensesTree,'lensesTree.txt')
testTree=grabTree('lensesTree.txt')
createPlot(testTree)