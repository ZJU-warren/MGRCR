import sys ;sys.path.append('../')
from Tools import *
import DataLinkSet as DLSet


# 获取一批数据
def GetNextBatch(dataLink):
    with open(dataLink, 'r') as f:
        line = f.readline()
        while line:
            elemSet = line.strip().split(',')               # 划分行元素
            user = elemSet[0]                               # 获取用户
            itemSet = [[int(elemSet[1]), int(elemSet[2])]]  # 获取物品和物品数目

            # 若数据存在
            while True:
                line = f.readline()
                if line:                                # 若非文末
                    elemSet = line.strip().split(',')   # 划分行元素
                    if elemSet[0] == user:              # 隶属该批
                        itemSet.append([int(elemSet[1]), int(elemSet[2])])
                    else:                               # 该批结束
                        break
                else:                                   # 全文结束
                    break
            yield int(user), itemSet


# 加载商品数据
def GetFeature_I():
    scalerI = LoadObj(DLSet.scaler_I_link)
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


if __name__ == '__main__':
    for user, itemSet in GetNextBatch(DLSet.sorted_saleInfo_Judge_R_link):
        print(user, itemSet)
