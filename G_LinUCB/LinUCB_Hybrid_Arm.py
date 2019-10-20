import numpy as np
from scipy import linalg


# 定义膀臂类
class Arm_hybrid:
    # 记忆体
    z = None

    # 膀臂初始化
    def __init__(self, id, d, k, alpha, feature):
        # 初始化属性
        self.id = id
        self.d = d
        self.k = k
        self.alpha = alpha

        # Lines 8 - 10  初始化每个臂的矩阵、向量
        self.A = np.identity(self.d)
        self.B = np.zeros((self.d, self.k))
        self.b = np.zeros((self.d, 1))
        self.x = np.array(feature[:self.d]).reshape((self.d, 1))

    # 返回A[a]
    def getA(self):
        return self.A

    # 返回B[a]
    def getB(self):
        return self.B

    # 返回b[a]
    def getb(self):
        return self.b

    # 返回ID值
    def getID(self):
        return self.id

    def getz(self):
        return self.z

    # 返回p[t,a]
    def getP(self, A0, betaHat, u):
        # 生成上下文并记忆保留
        z = np.outer(self.x, u).reshape(self.k, 1)
        self.z = z

        # 临时矩阵
        A_inv = linalg.inv(self.A)
        A0_inv = linalg.inv(A0)
        zT = np.transpose(z)
        xT = np.transpose(self.x)
        BT = np.transpose(self.B)

        # Lines 12 - 14
        thetaHat = np.dot(A_inv, self.b - np.dot(self.B, betaHat))
        s1 = np.dot(zT, np.dot(A0_inv, z))
        s2 = np.dot(zT, np.dot(A0_inv, np.dot(BT, np.dot(A_inv, self.x))))
        s3 = np.dot(xT, np.dot(A_inv, self.x))
        s4 = np.dot(xT, np.dot(A_inv, np.dot(self.B, np.dot(A0_inv, np.dot(BT, np.dot(A_inv, self.x))))))

        # Li line 13
        s = s1 - 2*s2 + s3 + s4
        # Li line 14
        p = np.dot(zT, betaHat) + np.dot(xT, thetaHat) + self.alpha*np.sqrt(s)

        return p

    # 更新
    def update(self, reward):
        self.A += np.dot(self.x, np.transpose(self.x))
        self.B += np.dot(self.x, np.transpose(self.z))
        self.b += reward * self.x




















