import numpy as np
from scipy import linalg


# 定义膀臂类
class Arm_disjoint:
    # 膀臂初始化
    def __init__(self, id, d, alpha, feature):
        # 初始化属性
        self.itemID = id
        self.d = d
        self.alpha = alpha
        self.x = np.array(feature[:self.d]).reshape((self.d, 1))

        # Lines 8 - 10  初始化每个臂的矩阵、向量
        self.A = np.identity(self.d)
        self.b = np.zeros((self.d, 1))

    # 返回A[a]
    def getA(self):
        return self.A

    # 返回b[a]
    def getb(self):
        return self.b

    # 返回ID值
    def getID(self):
        return self.itemID

    # 返回p[t,a]
    def getP(self):
        # 记忆保留
        # print('x', x.shape, type(x))

        # 临时矩阵
        A_inv = linalg.inv(self.A)
        xT = np.transpose(self.x)

        # line 8
        thetaHat = np.dot(A_inv, self.b)

        # line 9
        s = np.dot(xT, np.dot(A_inv, self.x))
        p = np.dot(np.transpose(thetaHat), self.x) + self.alpha * np.sqrt(s)
        return p

    # 更新
    def update(self, reward):
        self.A += np.dot(self.x, np.transpose(self.x))
        self.b += reward * self.x




















