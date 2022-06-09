import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
from matplotlib.font_manager import FontProperties
def showdatas(datingDataMat,datingLabels):
    #datingDataMat共三列 1.飞行常客距离 2.玩游戏时间 3.冰激淋公斤数
    #设置汉字格式 simhei 黑体
    font = FontProperties(fname=r"C:\windows\fonts\simhei.ttf",size=14)
    fig=plt.figure(figsize=(13,8))
    #将fig画布分成nrows*ncols个区域  axs[0][0]表示第一行第一个区域  x,y轴不共享 大小为（13，8）
    #fig, axs = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(13,8))  axs[0][0]
    numberofLabels=len(datingLabels)
    LabelsColors=[]
    for i in datingLabels:
        if i==1:
            LabelsColors.append('black')
        if i==2:
            LabelsColors.append('orange')
        if i==3:
            LabelsColors.append('red')
    #[0][0]画出散点图   1,2数据画图，散点大小15，透明度0.5

    axs0=fig.add_subplot(221)
    axs0.scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=LabelsColors,s=15,alpha=.5)
    #设置标题
    axs0_title=axs0.set_title(u'飞行常客里程数&玩视频游戏时间比较',FontProperties=font)
    axs0_xlabel_title=axs0.set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_title=axs0.set_ylabel(u'玩视频游戏消耗时间占比',FontProperties=font)
    #设置线属性 （每一行标题）
    plt.setp(axs0_title,size=9,weight='bold',color='red')
    plt.setp(axs0_xlabel_title,size=7,weight='bold',color='black')
    plt.setp(axs0_ylabel_title,size=7,weight='bold',color='black')

    #[0][1] 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs1=plt.subplot(222)
    axs1.scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title = axs1.set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_title = axs1.set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_title= axs1.set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    #设置线属性 （每一行标题）
    plt.setp(axs1_title,size=9,weight='bold',color='red')
    plt.setp(axs1_xlabel_title,size=7,weight='bold',color='black')
    plt.setp(axs1_ylabel_title,size=7,weight='bold',color='black')
    #[1][0]画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs2=plt.subplot(223)
    axs2.scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs2_title = axs2.set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_title = axs2.set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_title = axs2.set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    #设置线属性 （每一行标题）
    plt.setp(axs2_title,size=9,weight='bold',color='red')
    plt.setp(axs2_xlabel_title,size=7,weight='bold',color='black')
    plt.setp(axs2_ylabel_title,size=7,weight='bold',color='black')
    #设置图例
    didntLike=mlines.Line2D([],[],color='black',marker='.',markersize=6,label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',markersize=6, label='largeDoses')
    #添加图例
    axs0.legend(handles=[didntLike,smallDoses,largeDoses])
    axs1.legend(handles=[didntLike,smallDoses,largeDoses])
    axs2.legend(handles=[didntLike,smallDoses,largeDoses])

    plt.show()