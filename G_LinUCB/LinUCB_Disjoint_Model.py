from G_LinUCB.LinUCB_Disjoint_Arm import *
import random

# 模型
class LinUCB_disjoint:
    x = None

    # 模型初始化
    def __init__(self, alpha=2.1):
        # 初始化属性
        self.alpha = alpha      # 1 + np.sqrt(np.log(2/delta)/2)
        self.d = 9

        # 记录所有的膀臂
        self.arms = []

        # 初始化记忆体
        self.currentArm = None
        self.total = 0
        self.IDRecorder = {}
        self.resSet = None

    # 增膀臂
    def addArm(self, id, feature):
        self.arms.append(Arm_disjoint(id, self.d, self.alpha, feature))
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
    def recommend(self, user_features, K, trueSet=None):
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

            for i in range(30):
                r = random.randint(0, self.total-1)
                while r in trueSet:
                    r = random.randint(0, self.total - 1)
                resSet.append(r)

        for each in resSet:
            bestP.append(self.arms[each].getP())

        bestP = [each[0][0] for each in bestP]
        self.currentArm = np.argpartition(bestP, -K)[-K:]
        res = [self.arms[resSet[each]].getID() for each in self.currentArm]
        self.resSet = resSet
        return res

    # 更新结果
    def update(self, reward, K):
        for i in range(0, K):
            # print('there is :', self.arms[self.resSet[self.currentArm[i]]].getID())
            # print('reward is :', reward)
            # if self.resSet[self.currentArm[i]] in self.
            self.arms[self.resSet[self.currentArm[i]]].update(reward[i])

    # 初始化所有的arm
    def set_allArm(self, items):
        for each in items:
            self.addArm(each, items[each])


