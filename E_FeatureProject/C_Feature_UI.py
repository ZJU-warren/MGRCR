"""
    构建UI的交互特征, 并生成UI&U, UI&I特征
"""
import sys ;sys.path.append('../')
from Tools import *
import DataLinkSet as DLSet


# 统计用户购买某物的次数, 购买量, 购买金额, 促销次数以及均值
def Gen_ActionCounts(dfOrg, limitDay):
    # 保留相关天数
    df = dfOrg[dfOrg['dayID'] > limitDay]

    # 统计关键字
    timCountStr = 'UI_timCount_After_%d' % limitDay  # 用户-商品购买次数
    numCountStr = 'UI_numCount_After_%d' % limitDay  # 用户-商品购买量
    payCountStr = 'UI_payCount_After_%d' % limitDay  # 用户-商品购买额
    pmtCountStr = 'UI_pmtCount_After_%d' % limitDay  # 用户-商品促销次数

    avg_numCountStr = 'UI_avg_numCount_After_%d' % limitDay  # 平均购买商品数量
    avg_payCountStr = 'UI_avg_payCount_After_%d' % limitDay  # 平均商品消费金额
    avg_pmtRatioStr = 'UI_avg_pmtRatio_After_%d' % limitDay  # 促销次数占比

    # 统计每个用户-商品购物次数
    # 去除日多次购买记录
    dfTimCount = df[['userID', 'itemID']].copy()
    dfTimCount[timCountStr] = dfTimCount.groupby(['userID', 'itemID']).cumcount() + 1
    dfTimCount = dfTimCount.drop_duplicates(['userID', 'itemID'], keep='last')[['userID', 'itemID', timCountStr]]

    # 统计每个用户-商品购物量
    dfNumCount = df[['userID', 'itemID', '#item']].groupby(['userID', 'itemID'])['#item'].sum().reset_index()
    dfNumCount.columns = ['userID', 'itemID', numCountStr]

    # 统计每个用户-商品购买额
    dfPayCount = df[['userID', 'itemID', '#cost']].groupby(['userID', 'itemID'])['#cost'].sum().reset_index()
    dfPayCount.columns = ['userID', 'itemID', payCountStr]

    # 统计每个用户-商品促销次数
    dfPmtCount = df[['userID', 'itemID', 'isPromote']].groupby(['userID', 'itemID'])['isPromote'].sum().reset_index()
    dfPmtCount.columns = ['userID', 'itemID', pmtCountStr]

    # 合并统计特征
    dfA = pd.merge(dfTimCount, dfNumCount, on=['userID', 'itemID'])
    dfA = pd.merge(dfA, dfPmtCount, on=['userID', 'itemID'])
    dfA = pd.merge(dfA, dfPayCount, on=['userID', 'itemID'])

    # 求平均特征
    dfA[avg_numCountStr] = dfA[numCountStr] / dfA[timCountStr]  # 平均购买商品数量
    dfA[avg_payCountStr] = dfA[payCountStr] / dfA[timCountStr]  # 平均商品消费金额
    dfA[avg_pmtRatioStr] = dfA[pmtCountStr] / dfA[timCountStr]  # 促销次数占比

    if dfTimCount.shape[0] != dfA.shape[0]:
        print('****************************************')

    return dfA


# 生成用户-商品统计特征
def GenMergeUI(dfSale):
    # 统计用户购买某物的次数, 购买量, 购买金额, 促销次数以及均值
    df = dfSale.drop_duplicates(['userID', 'itemID'], keep='last')[['userID', 'itemID']]
    for each in DLSet.TimeGapSet:
        dfTemp = Gen_ActionCounts(dfSale, DLSet.SPLIT_DAY_NUM - each)
        df = pd.merge(df, dfTemp, on=['userID', 'itemID'], how='left')
    return df


# 生成二次特征
def Gen_UI_X(dfUI, dfX, keyWord, letter):
    df = pd.merge(dfUI, dfX, on=[keyWord])

    # 统计关键字
    timRatio_UI_U_Str = 'UI_%s_timRatio' % letter
    pmtRatio_UI_U_Str = 'UI_%s_pmtRatio' % letter

    # 购买占比
    df[timRatio_UI_U_Str] = df['UI_timCount_After_0'] / df['%s_timCount_After_0' % letter]
    # 促销占比
    df[pmtRatio_UI_U_Str] = df['UI_pmtCount_After_0'] / df['%s_pmtCount_After_0' % letter]

    return df.drop(columns=['%s_timCount_After_0' % letter, '%s_pmtCount_After_0' % letter])


# 主函数
def Main(dataLink, storeLink, featureLink_U, featureLink_I):
    dfSale = LoadData(dataLink)
    dfFU = LoadData(featureLink_U)[['userID', 'U_timCount_After_0', 'U_pmtCount_After_0']]
    dfFI = LoadData(featureLink_I)[['itemID', 'I_timCount_After_0', 'I_pmtCount_After_0']]

    # 生成UI统计特征
    dfFUI = GenMergeUI(dfSale)

    # 生成UI&U特征, UI&I特征
    dfFUI = Gen_UI_X(dfFUI, dfFU, 'userID', 'U')
    dfFUI = Gen_UI_X(dfFUI, dfFI, 'itemID', 'I')

    dfFUI.round(3).fillna(0).to_csv(storeLink, index=False)


def Run():
    Main(DLSet.saleInfo_Train_L_link, DLSet.feature_UI_train_link,
         DLSet.feature_U_train_link, DLSet.feature_I_train_link)
    Main(DLSet.saleInfo_Judge_L_link, DLSet.feature_UI_judge_link,
         DLSet.feature_U_judge_link, DLSet.feature_I_judge_link)


if __name__ == '__main__':
    Run()
    pass

"""
    # df = LoadData(DLSet.test_Link)
    # df = df.groupby(['名称'])['数量'].sum().reset_index()


    # 统计后k天占此前发生的冒险购比
"""

