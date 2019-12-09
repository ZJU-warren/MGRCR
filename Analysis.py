import DataLinkSet as DLSet


# 读取所有数据
def GetData(dataLink):
    with open(dataLink) as f:
        data = f.readlines()
    return data


# 处理一行数据
def ExcSupervised(line):
    total = line.split(",")
    # 得到用户
    user = 0    # int(total[0])

    # 获取真实结果
    y = [int(total[1])]
    flagT = int(total[2])
    flagP = int(total[3])

    return user, y, flagT, flagP


# 评价类
class Performance:
    TP = 0          # true positive
    FP = 0          # false positive
    FN = 0          # false negative
    TN = 0          # true negative

    novSum_d = [0] * 10      # nov和
    history = {}    # 历史
    N = 0           # 推荐轮数

    # 更新
    def Update(self, u, y, flagT, flagP):
        self.N += 1
        if flagT == 1:
            self.TP += flagP        # True Positive  : 被判定为正样本，事实上也是正样本
            self.FN += 1 - flagP    # False Negative : 被判定为负样本，但事实上是正样本
        else:
            self.FP += flagP        # False Positive : 被判定为正样本，事实上是负样本
            self.TN += 1 - flagP    # True Negative  : 被判定为负样本，事实上也是负样本

        if u not in self.history:
            self.history[u] = [-1]

        if flagT == 1 and flagP == 1:
            stale = 0
            fresh = 0
            for each in y:
                if each in self.history[u]:
                    stale += 1
                else:
                    fresh += 0.3
            for i in range(10):
                self.novSum_d[i] += (fresh - (i / 10) * stale)

            for each in y:
                if each not in self.history[u]:
                    self.history[u].append(each)

    def getPrecision(self):
        return self.TP / (self.TP + self.FP)

    def getRecall(self):
        return self.TP / (self.TP + self.FN)

    def getF1(self):
        p = self.getPrecision()
        r = self.getRecall()
        return 2 * p * r / (p + r)

    def getNov(self, id):
        return self.novSum_d[id-1] / self.N


def Main(dataLink):
    data = GetData(dataLink)
    obj = Performance()
    for each in data:
        user, y, flagT, flagP = ExcSupervised(each[:-1])
        obj.Update(user, y, flagT, flagP)

    print('Precision =', obj.getPrecision())
    print('Recall =', obj.getRecall())
    print('F1 =', obj.getF1())
    for x in range(1, 8):
        print('Novelty@0.%d =' % x, obj.getNov(x))


def Run():
    print('--------------------------------')
    Main(DLSet.resLR_link[1:])
    print('--------------------------------')
    Main(DLSet.resGBDT_link[1:])
    print('--------------------------------')
    Main(DLSet.resLRGBDT_link[1:])
    print('--------------------------------')
    Main((DLSet.resLinUCB_link[1:] + '_d') % 3)


if __name__ == '__main__':
    Run()