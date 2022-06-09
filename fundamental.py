from numpy import *
import numpy as np
import collections

'''
#(np.random.random([4,4]))
randmat=mat(random.rand(4,4))
#(randmat,randmat.I)  #.I矩阵求逆
randMat=randmat*randmat.I
#(randMat) #结果应该是单位矩阵
#randMat-eye(4)  #eye(4) 创建4*4单位矩阵

a = [2,3,1,5,4,1,1,2]
a = np.array(a)
collections.Counter(a) #返回字典  数字从小到大排列以及对应出现次数
b.most_common(2)  #列表形式返回前两位  数量最多的元素和数量
'''

def dropout(x,level):
    if level < 0. or level >=1:
        raise ValueError('Dropout level must be in interval [0,1[.')
    retain_prob = 1.-level
    random_tensor = np.random.binomial(n=1,p=retain_prob,size=x.shape)
    print(retain_prob,random_tensor)

    x *= random_tensor
    print(x)

    x/=retain_prob
    print(x)
    x/=retain_prob
    return x
x=np.array([1,2,3,4,5,6,7,8,9,10],dtype=np.float32)
dropout(x,0.4)
