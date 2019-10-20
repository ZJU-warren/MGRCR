"""
    构建I的交互特征，并与其标签特征合并
"""

from Tools import *
import DataLinkSet as DLSet


# 统计销售次数, 销售总量, 销售金额, 促销情况, 受众数, 以及均值

# 统计次数, 并求均值
def Gen_ActionCounts(dfOrg, limitDay):
    # 保留相关天数
    df = dfOrg[dfOrg['dayID'] > limitDay]

    # 统计关键字
    usrCountStr = 'I_usrCount_After_%d' % limitDay  # 商品受众数
    timCountStr = 'I_timCount_After_%d' % limitDay  # 商品销售次数
    numCountStr = 'I_numCount_After_%d' % limitDay  # 商品销售量
    payCountStr = 'I_payCount_After_%d' % limitDay  # 商品销售额
    pmtCountStr = 'I_pmtCount_After_%d' % limitDay  # 商品促销次数

    avg_numCountStr = 'I_avg_numCount_After_%d' % limitDay  # 商品平均销售量/次
    avg_payCountStr = 'I_avg_payCount_After_%d' % limitDay  # 商品平均销售额/次
    avg_pmtRatioStr = 'I_avg_pmtRatio_After_%d' % limitDay  # 促销销售占比

    # 统计每个商品受众数
    dfUsrCount = df.drop_duplicates(['userID', 'itemID'], keep='last')[['itemID']]
    dfUsrCount[usrCountStr] = dfUsrCount.groupby(['itemID']).cumcount() + 1
    dfUsrCount = dfUsrCount.drop_duplicates(['itemID'], keep='last')[['itemID', usrCountStr]]

    # 统计每个商品销售次数
    dfTimCount = df[['itemID']].copy()
    dfTimCount[timCountStr] = dfTimCount.groupby(['itemID']).cumcount() + 1
    dfTimCount = dfTimCount.drop_duplicates(['itemID'], keep='last')[['itemID', timCountStr]]

    # 统计每个商品销售量
    dfNumCount = df[['itemID', '#item']].groupby(['itemID'])['#item'].sum().reset_index()
    dfNumCount.columns = ['itemID', numCountStr]

    # 统计每个商品销售额
    dfPayCount = df[['itemID', '#cost']].groupby(['itemID'])['#cost'].sum().reset_index()
    dfPayCount.columns = ['itemID', payCountStr]

    # 统计每个商品促销次数
    dfPmtCount = df[['itemID', 'isPromote']].groupby(['itemID'])['isPromote'].sum().reset_index()
    dfPmtCount.columns = ['itemID', pmtCountStr]

    # 合并统计特征
    dfIA = pd.merge(dfUsrCount, dfTimCount, on=['itemID'])
    dfIA = pd.merge(dfIA, dfNumCount, on=['itemID'])
    dfIA = pd.merge(dfIA, dfPayCount, on=['itemID'])
    dfIA = pd.merge(dfIA, dfPmtCount, on=['itemID'])

    # 求平均特征
    dfIA[avg_numCountStr] = dfIA[numCountStr] / dfIA[timCountStr]   # 商品平均销售量/次
    dfIA[avg_payCountStr] = dfIA[payCountStr] / dfIA[timCountStr]   # 商品平均销售额/次
    dfIA[avg_pmtRatioStr] = dfIA[pmtCountStr] / dfIA[timCountStr]   # 促销销售占比

    if dfUsrCount.shape[0] != dfIA.shape[0]:
        print('****************************************')

    return dfIA


# 生成商品特征
def GenMergeI(dfSale, dfItem):
    # 统计销售次数, 销售总量, 销售金额, 促销情况, 受众数, 以及均值
    df = dfSale.drop_duplicates(['itemID'], keep='last')[['itemID']]
    for each in DLSet.TimeGapSet:
        dfTemp = Gen_ActionCounts(dfSale, DLSet.SPLIT_DAY_NUM - each)
        df = pd.merge(df, dfTemp, on=['itemID'], how='left')

    # 拼接商品标签特征
    df = pd.merge(df, dfItem, on=['itemID'])
    return df


# 主函数
def Main(dataLink_sale, dataLink_item, storeLink):
    dfSale = LoadData(dataLink_sale)
    dfItem = LoadData(dataLink_item)[['itemID', 'lsdj']]
    dfFI = GenMergeI(dfSale, dfItem)
    dfFI.round(3).fillna(0).to_csv(storeLink, index=False)


def Run():
    Main(DLSet.saleInfo_Train_L_link, DLSet.itemInfo_link, DLSet.feature_I_train_link)
    Main(DLSet.saleInfo_Judge_L_link, DLSet.itemInfo_link, DLSet.feature_I_judge_link)


if __name__ == '__main__':
    Run()
    pass


"""
    # df = LoadData(DLSet.test_Link)
    # df = df.groupby(['名称'])['数量'].sum().reset_index()
    
    
    # 统计后k天占此前发生的冒险购比
"""

