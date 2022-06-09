import numpy as np
import random


def loadDataSet():
    """
    读取文件信息
    :return:
    """
    data_mat, label_mat = [], []
    fr = open("../testSet.txt")
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(x):
    if x >= 0:
        return 1.0/(1+np.exp(-x))
    else:
        return np.exp(x)/(1+np.exp(x))


def grad_ascent(data_mat, class_labels):
    """
    梯度上升算法
    :param data_mat:2维数组
    :param class_labels:
    :return:
    """
    data_matrix = np.mat(data_mat)
    label_mat = np.mat(class_labels).T  # 矩阵转置
    m, n = np.shape(data_matrix)
    alpha = 0.001
    epochs = 500
    weights = np.ones((n, 1))  # 初始权重
    weights_array = np.array([])
    for k in range(epochs):
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)
        weights = weights + alpha * data_matrix.T * error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(epochs, n)
    return weights, weights_array


def stoc_grad_ascent0(data_matrix, class_labels):
    """
    随机梯度上升
    :param data_matrix:
    :param class_labels:
    :return:
    """
    m, n = np.shape(data_matrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(200):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * np.mat(data_matrix[i])
    return weights


def stoc_grad_ascent1(data_matrix, class_labels, num_iter=100):
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    weights_array = np.array([])
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+i+j) + 0.01  # 每次迭代调整更新率  常数项0.01防止alpha降低至0
            rand_index = int(random.uniform(0, len(data_index)))  # 随机更新系数
            h = sigmoid(sum(data_matrix[rand_index] * np.array(weights)))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * np.array(data_matrix[rand_index])
            weights_array = np.append(weights_array, np.array(weights).flatten(), axis=0)
        del data_index[rand_index]
    print(np.shape(weights_array))
    weights_array = weights_array.reshape(num_iter * m, n)
    return weights, weights_array


if __name__ == "__main__":
    data_arr, label_mat = loadDataSet()
    print(grad_ascent(data_arr, label_mat))
