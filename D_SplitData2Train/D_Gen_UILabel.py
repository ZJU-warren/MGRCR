"""
    生成UI, UILabel用于评估
"""
from Tools import *
import DataLinkSet as DLSet


# 生成所需要的U-I预测集
def Gen_UI(dfSale, dfItem):
    # 获得需预测的用户集
    dfS = dfSale[['userID']].copy()
    # 去掉重复的 UserID
    dfS = dfS.drop_duplicates(['userID'], keep='last')
    dfS['tempMark'] = 1
    print('user: ', dfS.shape[0])
    # 获得需预测的商品集
    dfI = dfItem[['itemID']].copy()
    dfI['tempMark'] = 1

    # 得到笛卡尔集
    df = pd.merge(dfS, dfI, on=['tempMark'])
    return df[['userID', 'itemID']]


# 生成UILabel
def Gen_UILabel(dfSale, dfUI):
    dfS = dfSale[['userID', 'itemID']].copy()
    # 去重
    dfS = dfS.drop_duplicates(['userID', 'itemID'], keep='last')

    # 实际发生的购买，标签为1
    dfS['label'] = 1

    # 拼接并填充空值
    dfUILabel = pd.merge(dfUI, dfS, on=['userID', 'itemID'], how='left').fillna(0).astype(int)

    return dfUILabel


# 主函数
def Main(saleLink, storeLink_UI, storeLink_UILabel):
    dfSale = LoadData(saleLink)
    dfItem = dfSale.drop_duplicates(['itemID'], keep='last')[['itemID']]
    print('item: ', dfItem.shape[0])
    # 生成UI
    dfUI = Gen_UI(dfSale, dfItem)
    dfUI.to_csv(storeLink_UI, index=False)

    # 生成UILabel
    dfUILabel = Gen_UILabel(dfSale, dfUI)
    dfUILabel.to_csv(storeLink_UILabel, index=False)

    print(dfUILabel.shape[0])
    print(dfUILabel[dfUILabel.label == 1].shape[0])


def Run():
    # 训练集
    Main(DLSet.saleInfo_Train_R_link, DLSet.saleInfo_Train_R_UI_link, DLSet.saleInfo_Train_R_UILabel_link)

    # 测试集
    Main(DLSet.saleInfo_Judge_R_link, DLSet.saleInfo_Judge_R_UI_link, DLSet.saleInfo_Judge_R_UILabel_link)


if __name__ == '__main__':
    Run()