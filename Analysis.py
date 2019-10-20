import DataLinkSet as DLSet


def GetData():
    f = open(DLSet.resLRGBDT_link)
    data = f.readlines()
    f.close()
    return data


"""
def ExChange(line):
    total = line.split("][")

    # 得到用户
    user = int(total[0][1:])

    # 获取真实结果
    y = total[1].split(',')
    y = [int(each) for each in y]

    # 获取推荐结果
    pred = total[2][:-1].split(',')
    pred = [int(each) for each in pred]

    return user, y, pred
"""



def ExChange(line):
    total = line.split(",")
    # 得到用户
    user = int(total[0])

    # 获取真实结果
    y = [int(total[1])]

    flagT = total[2] == 'True'
    flagP = total[3] == 'True'

    flag = flagT | flagP
    pred = y if flagP is True else [-1]

    return flag, user, y, pred



class Performance():
    TP = 0          # true positive
    FP = 0          # false positive
    FN = 0          # false negative
    TN = 0          # true negative

    novSum_d = [0, 0, 0]      # nov和
    history = {}    # 历史
    N = 0           # 推荐轮数

    def Update(self, u, y, pred):
        self.N += 1
        for each in pred:
            if each in y:
                self.TP += 1
            else:
                self.FP += 1

        for each in y:
            if each not in pred:
                self.FN += 1

        if u not in self.history:
            self.history[u] = [-1]

        # print(self.TP, self.FP, self.FN)
        stale = 0
        fresh = 0
        for each in pred:
            if each not in y and each in self.history[u]:
                stale += 1
                # if each not in y and each in self.history[u]:
            elif each in y and each not in self.history[u]:
                fresh += 0.3
                # print('[---]')
        # print('f, s', fresh, stale)
        self.novSum_d[0] += (fresh - 0.1 * stale)   # / len(pred)
        self.novSum_d[1] += (fresh - 0.2 * stale)   # / len(pred)
        self.novSum_d[2] += (fresh - 0.3 * stale)   # / len(pred)
        # print(self.novSum_d[0])
        for each in y:
            if each not in self.history[u]:
                self.history[u].append(each)
                # print(self.history[u])


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


def Main():
    data = GetData()
    obj = Performance()
    for each in data:
        flag, user, y, pred = ExChange(each[:-1])
        if flag is True:
            # print(flag, user, y, pred)
            obj.Update(user, y, pred)

    print('Precision =', obj.getPrecision())
    print('Recall =', obj.getRecall())
    print('F1 =', obj.getF1())
    print('Novelty@0.1 =', obj.getNov(1))
    print('Novelty@0.2 =', obj.getNov(2))
    print('Novelty@0.3 =', obj.getNov(3))


if __name__ == '__main__':
    Main()
    # p = 0.2700787401574803
    # r = 0.13165266106442577
    # print(2 * p * r / (p + r))
