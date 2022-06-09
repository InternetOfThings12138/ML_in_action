import matplotlib.pyplot as plt
import numpy as np
import logRegres


def plot_best_fit(weights):
    data_mat, label_mat = logRegres.loadDataSet()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]
    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


if __name__ == "__main__":
    data_arr, label_mat = logRegres.loadDataSet()

    # grad_ascent 原始方法，迭代次数多，速度慢
    # weights, weights_array = logRegres.grad_ascent(data_arr, label_mat)
    # print(weights.getA())
    # plot_best_fit(weights.getA())  # .getA() 将自身矩阵变量转化为ndarray类型的变量。等价于np.asarray(self)

    # stoc_grad_ascent0 采用随机梯度上升，初始版，未对weights进行调整
    # weights, weights_array  = logRegres.stoc_grad_ascent0(data_arr, label_mat)
    # print(weights)
    # plot_best_fit(weights)

    # stoc_grad_ascent1 采用随机梯度上升，调整alpha，随机选择样本更新weights
    weights, weights_array = logRegres.stoc_grad_ascent1(data_arr, label_mat)
    print(weights)
    plot_best_fit(weights)

