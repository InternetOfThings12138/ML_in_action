import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import logRegres

def plotWeights(weights_array1, weights_array2):
    # 设置汉字格式为14号简体字
    font = FontProperties(fname=r"C:\Windows\Fonts\simkai.ttf", size=14)
    # 将fig画布分隔成1行1列，不共享x轴和y轴，fig画布的大小为（20, 10）
    # 当nrows=3，ncols=2时，代表fig画布被分为6个区域，axs[0][0]代表第一行第一个区域
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    # x1坐标轴的范围
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'改进的梯度上升算法，回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'w0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'w1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_title_text = axs[2][0].set_title(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'w2', FontProperties=font)
    plt.setp(axs2_title_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    # x2坐标轴的范围
    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'梯度上升算法，回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'w0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'w1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_title_text = axs[2][1].set_title(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'w2', FontProperties=font)
    plt.setp(axs2_title_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()


if __name__ == "__main__":
    # 测试简单梯度上升法
    # Gradient_Ascent_test()
    # 加载数据集
    data_mat, label_mat = logRegres.loadDataSet()
    # 训练权重
    weights2, weights_array2 = logRegres.grad_ascent(data_mat, label_mat)
    # 新方法训练权重
    weights1, weights_array1 = logRegres.stoc_grad_ascent1(np.array(data_mat), label_mat)
    # 绘制数据集中的y和x的散点图
    # plotBestFit(weights)
    # print(gradAscent(dataMat, labelMat))
    plotWeights(weights_array1, weights_array2)
