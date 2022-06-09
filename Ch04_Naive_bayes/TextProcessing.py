from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import os
import random
import jieba
"""
函数说明：中文文本处理
        将文件夹内部的所有txt文档分词并存储在data_list中，将txt上一级文件夹名称存储在class_list中

Parameters:
    folder_path - 文本存放的路径
    test_size - 测试集占比，默认占所有数据集的20%
    
Returns:
    all_words_list - 按词频降序排序的训练集列表
    train_data_list - 训练集列表
    test_data_list - 测试集列表
    train_class_list - 训练集标签列表
    test_class_list - 测试集标签列表
"""


def TextProcessing(folder_path, test_size=0.2):
    folder_list = os.listdir(folder_path)
    data_list, class_list = [], []
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        j = 1
        for file in files:
            if j > 100:
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as f:
                raw = f.read()
            word_cut = jieba.cut(raw, cut_all=False)
            word_list = list(word_cut)
            data_list.append(word_list)
            class_list.append(folder)
            j += 1
    data_class_list = list(zip(data_list, class_list))
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)
    all_words_list = list(all_words_list)
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def MakeWordsSet(words_file):
    """
    读取文件内容并去重
    :param words_file: 文件路径
    :return:  读取内容set集合
    """
    words_set = set()
    with open(words_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                words_set.add(word)
    return words_set


def words_dict(all_words_list, deleteN, stopwords_set=set()):
    """
    文本特征选取
    :param all_words_list:  训练集所有文本列表
    :param deleteN:   删除词频最高的N个词
    :param stopwords_set:  指定结束语
    :return:  特征集
    """
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words):
    """
    根据feature_words将文本向量化
    :param train_data_list: 训练集
    :param test_data_list: 测试集
    :param feature_words: 特征集
    :return:
        train_feature_list - 训练集向量化列表
        test_feature_list - 测试集向量化列表
    """
    # 出现在特征集中，则置1
    def text_features(text, feature_words):
        # set是一个无序且不重复的元素集合
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    # 返回结果
    return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    """
    函数说明：新闻分类器
    :param train_feature_list: 训练集向量化的特征文本
    :param test_feature_list: 测试集向量化的特征文本
    :param train_class_list: 训练集分类标签
    :param test_class_list: 测试集分类标签
    :return:  test_accuracy: 分类器精度
    """
    # fit(X,y) Fit Naive Bayes classifier according to X, y
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    # score(X,y) Returns the mean accuracy on the given test data and labels
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy


def main():
    # 文本预处理
    # 训练集存放地址
    folder_path = './SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
    # print(all_words_list)
    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)
    # 词频出现前100的删除
    # feature_words = words_dict(all_words_list, 100, stopwords_set)
    # print(feature_words)
    test_accuracy_list = []
    deleteNs = range(0, 1000, 20)
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list.append(test_accuracy)
    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accurecy')
    plt.show()
    # 经过测试450效果比较好
    feature_words = words_dict(all_words_list, 450, stopwords_set)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    test_accuracy_list.append(test_accuracy)
    ave = sum(test_accuracy_list) / len(test_accuracy_list)
    print('当删掉前450个高频词分类精度为：%.5f' % ave)


if __name__ == "__main__":
    main()