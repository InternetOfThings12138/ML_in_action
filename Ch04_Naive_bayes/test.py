import jieba
import random
str = "任性的90后boy"
aa = jieba.cut(str)
print("/".join(aa))
jieba.add_word("任性的")
bb = jieba.cut_for_search(str)
print("/".join(aa))
print([y for y in bb])
a = [1,2,3,4]
random.shuffle(a)
print(a)

