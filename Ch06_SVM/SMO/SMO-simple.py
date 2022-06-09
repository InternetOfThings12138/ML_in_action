import numpy as np
import matplotlib.pyplot as plt
import random


def load_data_set(file_name):
    """
    读取数据
    :param file_name:文件路径
    :return: data_mat 数据矩阵    label_mat 标签矩阵
    """
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_jrand(i, m):
    """
    随机选择一个整数
    :param i: alpha的下标
    :param m: alpha数目
    :return:
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))  # 在(0,m)范围内随机一个与i不相等的数字
    return j


def clip_alpha(aj, H, L):
    '''
    优化alpha
    :param aj:alpha_j
    :param H: 上限
    :param L: 下限
    :return:
    '''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smo_simple(data_mat_in, class_labels, C, toler, max_iter):
    '''
    简化版smo算法
    :param data_mat_in: 数据矩阵
    :param class_lables: 数据标签
    :param C: 松弛变量
    :param toler: 容错率
    :param max_iter: 最大迭代次数
    :return: 无
    '''
    # 转换为numpy的mat存储
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).T
    # 初始化b参数 统计dataMatrix的维度
    b = 0
    m, n = np.shape(data_matrix)
    # 初始化alpha参数，设为0
    alphas = np.mat(np.zeros((m, 1)))
    # 初始化迭代次数
    iter_num = 0
    # 最多迭代matIter次
    while iter_num < max_iter:
        alpha_pairs_changed = 0  # 记录alpha是否已经优化
        for i in range(m):
            # 步骤1：计算误差Ei
            fXi = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b  # 预测类别。
            Ei = fXi - float(label_mat[i])  # 误差
            # 符合优化条件
            if ((label_mat[i] * Ei < -toler) and (alphas[i] < C)) or ((label_mat[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机选择另一个与alpha_i成对优化的alpha_j
                j = select_jrand(i, m)
                # 步骤1：计算误差Ej
                fXj = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T) + b)
                Ej = fXj - float(label_mat[j])
                # 保存更新前的alpha值
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 步骤2计算上下界L、H
                if label_mat[i] != label_mat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L==H')
                    continue
                # 步骤3：计算eta
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T - \
                      data_matrix[j,:] * data_matrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                # 步骤4：更新alpha_j
                alphas[j] -= label_mat[j] * (Ei - Ej) / eta
                # 步骤5：修建alpha_j
                alphas[j] = clip_alpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print('alpha_j变化太小')
                    continue
                # 步骤6：更新alpha_i
                alphas[i] += label_mat[j] * label_mat[i] * (alphaJold - alphas[i])
                # 步骤7：更新b_1 和b_2
                b1 = b - Ei - label_mat[i] * (alphas[i] - alphaIold) * data_matrix[i, :] * data_matrix[i, :].T - label_mat[
                    j] * (alphas[j] - alphaJold) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - Ej - label_mat[i] * (alphas[i] - alphaIold) * data_matrix[i, :] * data_matrix[j, :].T - label_mat[
                    j] * (alphas[j] - alphaJold) * data_matrix[j, :] * data_matrix[j, :].T
                # 步骤8： 根据b_1和b_2更新b
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 统计优化次数
                alpha_pairs_changed += 1
                # 打印统计信息
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, alpha_pairs_changed))
        # 更新迭代次数
        if alpha_pairs_changed == 0:
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数 %d" % iter_num)
    return b, alphas


def showClassifer(dataMat, labelMat, w, b):
    '''
    绘制分类结果
    :param dataMat:数据矩阵
    :param w: y=wx+b
    :param b:
    :return:
    '''
    # 绘制样本点
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.xlabel('[0]', fontsize=15, color='r')
    plt.ylabel('[1]', fontsize=15, color='r')
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)  # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)  # 负样本散点图
    # 绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()


def calc_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


if __name__ == '__main__':
    data, label = load_data_set('testSetRBF.txt')
    b, alphas = smo_simple(data, label, 0.6, 0.001, 400) # 迭代400次与4000次并没有明显差别
    w = calc_w(data, label, alphas)
    showClassifer(data, label, w, b)
