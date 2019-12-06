import sys ;sys.path.append('../')
from Tools import *
import DataLinkSet as DLSet
from G_LinUCB.DataTool import *
# from G_LinUCB.LinUCB_Disjoint_Model import *
from G_LinUCB.LinUCB_Hybrid_Model import *


# 获取模型实例
def Init():
    obj = []
    scoreSet = []
    totalSet = []
    objCnt = 0

    for alpha in np.arange(1.4, 2.3, 0.4):
        obj.append(LinUCB_hybrid(alpha))
        scoreSet.append(0)
        totalSet.append(0)

        # 初始化商品特征
        itemDic = GetFeature_I()
        obj[objCnt].set_allArm(itemDic)
        objCnt += 1


    # 返回模型实例
    return obj, objCnt, scoreSet, totalSet


def UpdateASample(obj, user, itemList, dfU, S):
    if S == -1:
        N = len(itemList)
    else:
        N = S
    # print(user, itemList)
    itemSet = [each[0] for each in itemList]
    numSet = [each[1] for each in itemList]

    # print(user, itemSet, numSet)

    user_features = dfU[dfU['userID'] == user].values
    if S == -1:
        recommend_list = obj.recommend(user_features, N, itemSet)
    else:
        recommend_list = obj.recommend(user_features, N)
    reward = []
    for each in recommend_list:
        if each in itemSet:
            reward.append(1)
        else:
            reward.append(0)
    obj.update(reward, N)
    return N, sum(reward)


# 给定数据集训练obj
def TrainBatch(obj, dataLink, fULink, scalerU, N=-1):
    scoreSet = [0] * len(obj)
    totalSet = [0] * len(obj)
    # 获取用户特征集并归一
    dfU = LoadData(fULink)
    User = dfU[['userID']].copy()
    dfU = pd.DataFrame(scalerU.transform(dfU[DLSet.userFeatureSet].values))
    dfU = pd.concat([User, dfU], axis=1)
    lenObj = len(obj)

    for user, itemList in GetNextBatch(dataLink):
        for i in range(lenObj):
            total, score = UpdateASample(obj[i], user, itemList, dfU, N)
            # print(score, total)
            scoreSet[i] += score
            totalSet[i] += total
            # print('alpha = %f: hitRatio =%f' % (1.4 + 0.4 * i, scoreSet[i] / totalSet[i]))

    for i in range(lenObj):
        print('alpha = %f: hitRatio =%f' % (1.4 + 0.4 * i, scoreSet[i] / totalSet[i]))


def Main():
    # 读取标准化scaler
    scalerU = LoadObj(DLSet.scaler_U_link)

    obj, objCnt, scoreSet, totalSet = Init()

    TrainBatch(obj, DLSet.sorted_saleInfo_Train_L_link, DLSet.feature_U_train_link, scalerU)
    StoreObj(obj, DLSet.modelObj_link % 'hybrid')
    TrainBatch(obj, DLSet.sorted_saleInfo_Train_R_link, DLSet.feature_U_train_link, scalerU)
    StoreObj(obj, DLSet.modelObj_link % 'hybrid')
    TrainBatch(obj, DLSet.sorted_saleInfo_Judge_L_link, DLSet.feature_U_judge_link, scalerU)
    StoreObj(obj, DLSet.modelObj_link % 'hybrid')

    print('---------------------Predict--------------------')
    obj = LoadObj(DLSet.modelObj_link % 'hybrid')
    print('---------------------Predict dont change--------------------')
    TrainBatch(obj, DLSet.sorted_saleInfo_Judge_R_link, DLSet.feature_U_judge_link, scalerU, 1)
    TrainBatch(obj, DLSet.sorted_saleInfo_Judge_R_link, DLSet.feature_U_judge_link, scalerU, 2)
    TrainBatch(obj, DLSet.sorted_saleInfo_Judge_R_link, DLSet.feature_U_judge_link, scalerU, 3)

    print('---------------------Predict set 1--------------------')
    for each in obj:
        each.alpha = 1
        for _ in each.arms:
            _.alpha = 1
    TrainBatch(obj, DLSet.sorted_saleInfo_Judge_R_link, DLSet.feature_U_judge_link, scalerU, 1)
    TrainBatch(obj, DLSet.sorted_saleInfo_Judge_R_link, DLSet.feature_U_judge_link, scalerU, 2)
    TrainBatch(obj, DLSet.sorted_saleInfo_Judge_R_link, DLSet.feature_U_judge_link, scalerU, 3)


if __name__ == '__main__':
    Main()
