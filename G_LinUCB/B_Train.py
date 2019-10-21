import sys ;sys.path.append('../')
from Tools import *
import DataLinkSet as DLSet
from G_LinUCB.DataTool import *
from G_LinUCB.LinUCB_Disjoint_Model import *
from G_LinUCB.LinUCB_Hybrid_Model import *


# 获取模型实例
def Init():
    obj = []
    scoreSet = []
    totalSet = []
    objCnt = 0

    for alpha in np.arange(1.4, 2.3, 0.4):
        objCnt += 1
        obj.append(LinUCB_hybrid(alpha))
        scoreSet.append(0)
        totalSet.append(0)

        # 初始化商品特征
        itemDic = GetFeature_I()
        obj[objCnt].set_allArm(itemDic)

    # 返回模型实例
    return obj, objCnt, scoreSet, totalSet


# 给定数据集训练obj
def TrainBatch(obj, objCnt, dataLink, fULink, scoreSet, totalSet, scalerU, N=-1):
    # 获取用户特征集并归一
    dfU = LoadData(fULink)
    User = dfU[['userID']].copy()
    dfU = pd.DataFrame(scalerU.transform(dfU[DLSet.userFeatureSet].values))
    dfU = pd.concat([User, dfU], axis=1)

    for user, itemSet in GetNextBatch(dataLink):
        print(user, itemSet)

        for i in range(lenObj):
            total, score = UpdateASample(obj[i], user, itemSet, dfU, N)
            scoreSet[i] += score
            totalSet[i] += total


def Main():
    # 读取标准化scaler
    scalerU = LoadObj(DLSet.scaler_U_link)

    obj, objCnt, scoreSet, totalSet = Init()
    TrainBatch(obj, DLSet.sorted_saleInfo_Train_L_link, DLSet.feature_U_train_link, scoreSet, totalSet, scalerU)
    TrainBatch(obj, DLSet.sorted_saleInfo_Train_R_link, DLSet.feature_U_train_link, scoreSet, totalSet, scalerU)
    TrainBatch(obj, DLSet.sorted_saleInfo_Judge_L_link, DLSet.feature_U_judge_link, scoreSet, totalSet, scalerU)

    PredictBatch(obj, DLSet.sorted_saleInfo_Judge_R_link, DLSet.feature_U_judge_link, scoreSet, totalSet, scalerU)

if __name__ == '__main__':
    Main()