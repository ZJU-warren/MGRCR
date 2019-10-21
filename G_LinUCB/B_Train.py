import sys ;sys.path.append('../')

# from G_LinUCB.LinUCB_Disjoint_Model import *
from G_LinUCB.LinUCB_Hybrid_Model import *
from Tools import *
import DataLinkSet as DLSet

scalerU = LoadObj(DLSet.scaler_U_link)
scalerI = LoadObj(DLSet.scaler_I_link)


# 加载商品数据
def GetFeature_I():
    df1 = LoadData(DLSet.feature_I_train_link)
    df2 = LoadData(DLSet.feature_I_judge_link)

    # 取出训练集和测试集所有相关商品的特征, 并去重
    df = pd.concat([df1, df2])
    df = df.drop_duplicates(['itemID'], keep='first').reset_index(drop=True)

    Item = df[['itemID']].copy()
    df = pd.DataFrame(scalerI.transform(df[DLSet.itemFeatureSet].values))
    df = pd.concat([Item, df], axis=1)

    # 转换为字典形式
    itemDic = df.set_index('itemID').T.to_dict('list')
    return itemDic


# 初始化模型实例
def Init():
    # 获取模型实例
    # obj = LinUCB_disjoint()
    obj = []
    scoreSet = []
    totalSet = []
    cnt = 0
    for alpha in np.arange(1.4, 2.3, 0.2):
        obj.append(LinUCB_hybrid(alpha))
        scoreSet.append(0)
        totalSet.append(0)
        # 初始化商品特征
        itemDic = GetFeature_I()
        obj[cnt].set_allArm(itemDic)
        cnt += 1

    # 返回模型实例
    return obj, scoreSet, totalSet


# 获取一批数据
def GetNextBatch(f):
    # 读取第一行
    line = f.readline().strip().split(',')
    user = line[0]
    itemSet = [[int(line[1]), int(line[2])]]
    code = False            # 文件结尾码

    # 读取该批次
    cnt = 0
    while True:
        cnt = cnt + 1
        loc = f.tell()
        line = f.readline()
        if line:
            line = line.strip().split(',')
            if line[0] == user:             # 隶属该批
                itemSet.append([int(line[1]), int(line[2])])
            else:                           # 该批结束
                f.seek(loc)
                break
        else:                               # 文件末尾
            code = True
            break

    # 整数化
    # itemSet = [int(each[1]) for each in itemSet]
    return int(user), itemSet, code


# 训练
def TrainBatch(obj, user, dataSet, dfU, N, f):
    total = 0
    score = 0

    # global scalerU
    # 用户特征
    userVec = dfU[dfU['userID'] == user].values.tolist()[0]
    userVec = [each for each in userVec]

    # 物品和数目
    itemSet = [data[0] for data in dataSet]
    numSet = [data[1] for data in dataSet]

    # 预测
    # N = 3   #  len(itemSet)
    total += N
    res = obj.recommend(userVec, N)

    f.write('['+str(user)+']'+str(itemSet)+str(res)+'\n')
    # 学习
    reward = []
    for i in range(N):
        flag = False
        for j in range(len(itemSet)):
            if res[i] == itemSet[j]:
                reward.append(numSet[j])
                score += numSet[j] * (N - i)
                flag = True
            break
        if flag is False:
            reward.append(-1)

    return total, score
    # if total % 10 == 0:
    #    print(total, score, score / total)

    # 更新
    # obj.update(reward, N)


# 训练
def Train(obj, dataLink, fULink, scoreSet, totalSet, N=3):
    fw = open(DLSet.resLinUCB_link % N, 'w')
    # 获取用户特征集并归一
    dfU = LoadData(fULink)
    User = dfU[['userID']].copy()
    dfU = pd.DataFrame(scalerU.transform(dfU[DLSet.userFeatureSet].values))
    dfU = pd.concat([User, dfU], axis=1)
    f = open(dataLink, 'r')
    code = False

    lenObj = len(obj)
    # 训练所有数据
    while code is not True:
        user, itemSet, code = GetNextBatch(f)

        for i in range(lenObj):
            total, score = TrainBatch(obj[i], user, itemSet, dfU, N, fw)
            scoreSet[i] += score
            totalSet[i] += total
            # if total % 10 == 0:
            #     print(total, score, score / total)

    for i in range(lenObj):
        print(i*0.2 + 1.4, totalSet[i], scoreSet[i], scoreSet[i] / totalSet[i])
    f.close()
    fw.close()


def Main():
    # global scoreSet, totalSet
    # 初始化模型
    obj, scoreSet, totalSet = Init()

    # 训练模型
    print('1-----------------------------------------------------------------')
    Train(obj, DLSet.sorted_saleInfo_Train_L_link, DLSet.feature_U_train_link, scoreSet, totalSet)
    StoreObj(obj, DLSet.modelObj_link)
    print('2-----------------------------------------------------------------')
    Train(obj, DLSet.sorted_saleInfo_Train_R_link, DLSet.feature_U_train_link, scoreSet, totalSet)
    StoreObj(obj, DLSet.modelObj_link)
    print('3-----------------------------------------------------------------')
    Train(obj, DLSet.sorted_saleInfo_Judge_L_link, DLSet.feature_U_judge_link, scoreSet, totalSet)
    StoreObj(obj, DLSet.modelObj_link)

    print('4-----------------------------------------------------------------')
    obj = LoadObj(DLSet.modelObj_link)

    lenObj = len(obj)
    for i in range(lenObj):
        for each in obj[i].arms:
             each.alpha = 1
        scoreSet[i] = 0
        totalSet[i] = 0

    for N in range(1, 4):
        Train(obj, DLSet.sorted_saleInfo_Judge_R_link, DLSet.feature_U_judge_link, scoreSet, totalSet, N)

    # StoreObj(obj, DLSet.modelObj_link)


def Run():
    Main()


if __name__ == '__main__':
    Run()


"""
    x = f.tell()
    res = f.readline()
    f.seek(x)
"""
