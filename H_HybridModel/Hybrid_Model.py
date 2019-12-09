from H_HybridModel.Hybrid_Arm import *
from H_HybridModel.fTool import FTool
import random


# 模型
class LinUCB_hybrid:
    betaHat = None

    # 模型初始化
    def __init__(self, alpha=2.1):
        # 初始化属性
        self.alpha = alpha                  # 1 + np.sqrt(np.log(2/delta)/2)
        self.d = 6                          # size of user features
        self.k = self.d * self.d            # size of art features
        self.A0 = np.identity(self.k)       # initialization of env features, Line 1
        self.b0 = np.zeros((self.k, 1))     # Line 2

        self.r1 = 40
        self.r0 = -5

        # 记录所有的膀臂
        self.arms = []

        # 初始化记忆体
        self.currentArm = None
        self.total = 0
        self.IDRecorder = {}
        self.resSet = None

        self.fTool = FTool()

    # 增膀臂
    def addArm(self, id, feature):
        # Add an arm to the system with unique ID id, with feature length fLen:
        self.arms.append(Arm_hybrid(id, self.d, self.k, self.alpha, feature))
        self.IDRecorder[id] = self.total
        self.total += 1

    # 减膀臂
    def removeArm(self, id):
        try:
            del self.arms[id]
            print('Removed arm: ', id)
        except KeyError:
            print('Attempted to remove non-existed arm id', id)

    # 选择推荐
    def recommend(self, user_features, K, userID, trueSet=None):
        # load user feature
        # print(np.array(user_features[0][0:self.d]))
        u = np.array(user_features[0][:self.d]).reshape((self.d, 1))

        # Line 5
        self.betaHat = np.dot(linalg.inv(self.A0), self.b0)

        # Line 16
        bestP = []
        resSet = []

        if trueSet is None:
            for i in range(0, self.total):
                resSet.append(i)
        else:
            for each in trueSet:
                x = self.IDRecorder.get(each)
                if x is not None:
                    resSet.append(self.IDRecorder[each])

            for i in range(5):
                r = random.randint(0, self.total-1)
                while r in trueSet:
                    r = random.randint(0, self.total - 1)
                resSet.append(r)

        for each in resSet:
            z = self.fTool.get(userID, self.arms[each].getID())
            bestP.append(self.arms[each].getP(self.A0, self.betaHat, u, z))


        bestP = [each[0][0] for each in bestP]
        self.currentArm = np.argpartition(bestP, -K)[-K:]
        res = [self.arms[resSet[each]].getID() for each in self.currentArm]
        self.resSet = resSet

        return res

    # 更新结果
    def update(self, reward, N):
        for i in range(N):
            # reward[i] = reward[i] * (self.r1 if reward[i] > 0 else self.r0)
            # lines 17-18
            z = self.arms[self.resSet[self.currentArm[i]]].getz()
            B = self.arms[self.resSet[self.currentArm[i]]].getB()
            A = self.arms[self.resSet[self.currentArm[i]]].getA()
            b = self.arms[self.resSet[self.currentArm[i]]].getb()
            zT = np.transpose(z)

            self.A0 += np.dot(np.transpose(B),
                              np.dot(linalg.inv(A), B))
            self.b0 += np.dot(np.transpose(B),
                              np.dot(linalg.inv(A), b))

            # Update the arm-specific matrices: lines 19-21
            self.arms[self.resSet[self.currentArm[i]]].update(reward[i])

            # Update the general matrices again: lines 22-23

            self.A0 += np.dot(z, zT)
            self.A0 -= np.dot(np.transpose(B),
                              np.dot(linalg.inv(A), B))

            self.b0 += reward[i] * z
            self.b0 -= np.dot(np.transpose(B),
                              np.dot(linalg.inv(A), b))

    # 初始化所有的arm
    def set_allArm(self, items):
        for each in items:
            self.addArm(each, items[each])
