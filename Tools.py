import pandas as pd
# import matplotlib.pyplot as plt
import pickle
import numpy as np


# 从文件中读取表
def LoadData(dataLink):
    df = pd.read_csv(dataLink)
    return df


# # 绘制图片
# def ShowPic(xSet, ySet, label, title, xLabel, yLabel='F1'):
#     # 输出结果
#     lenElem = len(xSet)
#     for i in range(lenElem):
#         print(xSet[i], ySet[i])
#
#     # 绘制图形
#     plt.plot(xSet, ySet, label=label)
#     plt.title(title)
#     plt.xlabel(xLabel)
#     plt.ylabel(yLabel)
#     plt.legend(loc=4)
#     plt.grid(True, linewidth=0.3)
#     plt.show()


# 存储实例
def StoreObj(obj, storeLink):
    pickle.dump(obj, open(storeLink, 'wb'))


# 读取实例,
def LoadObj(dataLink):
    return pickle.load(open(dataLink, 'rb'))