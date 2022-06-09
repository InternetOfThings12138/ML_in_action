import re
from bayes import *


def textParse(big_string):
    """
    解析为小写字符串列表，并去掉长度小于2的
    :param bigString:
    :return:
    """
    list_of_tokens = re.split(r"\W*", big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spamTest():
    """
    利用贝叶斯进行垃圾邮件分类器
    :return:
    """
    doc_list, class_list, full_text = [], [], []
    for i in range(1, 26):
        word_list = textParse(open(r'email/spam/%d.txt' % i, encoding='gbk').read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)  # 垃圾邮件标签为１
        word_list = textParse(open(r'email/ham/%d.txt' % i, encoding='gbk').read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = createVocabList(doc_list)
    training_set = list(range(50))  # 数据集仅有50封邮件
    test_set = []
    for i in range(10):  # 10封为测试集
        rand_index = int(random.uniform(0, len(training_set)))  # 随机选择部分测试集方法为留存交叉验证（hold_out cross validation）
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])
    train_mat, train_classes = [], []  # 构建训练集、测试集
    for doc_index in training_set:
        train_mat.append(setOfWords2Vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0V, p1V, pSpam = trainNBO(array(train_mat), array(train_classes))  # 训练模型
    error_count = 0
    for doc_index in test_set:
        word_vector = setOfWords2Vec(vocab_list, doc_list[doc_index])
        if classifyNB(array(word_vector), p0V, p1V, pSpam) != class_list[doc_index]:
            error_count += 1
    print("The error rate is:", float(error_count) / len(test_set))


if __name__ == "__main__":
    """
    
    regEx = re.compile(r"\W*")
    emailText = open("email/ham/6.txt").read()
    ListOfToken = regEx.split(emailText)
    print(ListOfToken)
    """
    spamTest()
