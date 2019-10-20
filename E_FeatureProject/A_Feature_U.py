"""
    构建U的交互特征，并与其标签特征合并
"""

from Tools import *
import DataLinkSet as DLSet


# 统计次数, 并求均值
def Gen_ActionCounts(dfOrg, limitDay):
    # 保留相关天数
    df = dfOrg[dfOrg['dayID'] > limitDay]

    # 统计关键字
    dayCountStr = 'U_dayCount_After_%d' % limitDay    # 实际发生购买天数
    timCountStr = 'U_timCount_After_%d' % limitDay    # 购买商品数量
    payCountStr = 'U_payCount_After_%d' % limitDay    # 消费金额
    pmtCountStr = 'U_pmtCount_After_%d' % limitDay    # 促销品购买数量

    avg_timCountStr = 'U_avg_numCount_After_%d' % limitDay  # 平均购买商品数量
    avg_payCountStr = 'U_avg_payCount_After_%d' % limitDay  # 平均消费金额
    avg_pmtRatioStr = 'U_avg_pmtRatio_After_%d' % limitDay  # 促销商品占比

    # 统计每个用户购物次数  --> 按实际发生购物的天数算
    # 去除日多次购买记录
    dfDayCount = df.drop_duplicates(['userID', 'dayID'], keep='last')[['userID']]
    dfDayCount[dayCountStr] = dfDayCount.groupby(['userID']).cumcount() + 1
    dfDayCount = dfDayCount.drop_duplicates(['userID'], keep='last')[['userID', dayCountStr]]

    # 统计每个用户购物数量
    # 去除日多次购买记录
    dfNumCount = df[['userID']].copy()
    dfNumCount[timCountStr] = dfNumCount.groupby(['userID']).cumcount() + 1
    dfNumCount = dfNumCount.drop_duplicates(['userID'], keep='last')[['userID', timCountStr]]

    # 统计每个用户促销品购买数量
    dfPmtCount = df[['userID', 'isPromote']].groupby(['userID'])['isPromote'].sum().reset_index()
    dfPmtCount.columns = ['userID', pmtCountStr]

    # 统计每个用户购物金额
    dfPayCount = df[['userID', '#cost']].groupby(['userID'])['#cost'].sum().reset_index()
    dfPayCount.columns = ['userID', payCountStr]

    # 合并统计特征
    dfUA = pd.merge(dfDayCount, dfNumCount, on=['userID'])
    dfUA = pd.merge(dfUA, dfPmtCount, on=['userID'])
    dfUA = pd.merge(dfUA, dfPayCount, on=['userID'])

    # 求平均特征
    dfUA[avg_timCountStr] = dfUA[timCountStr] / dfUA[dayCountStr]
    dfUA[avg_payCountStr] = dfUA[payCountStr] / dfUA[dayCountStr]
    dfUA[avg_pmtRatioStr] = dfUA[pmtCountStr] / dfUA[timCountStr]

    if dfDayCount.shape[0] != dfUA.shape[0]:
        print('****************************************')

    return dfUA


# 生成用户特征
def GenMergeU(dfSale, dfUser):
    # 统计购物次数, 购物数目, 促销数目, 购物金额, 以及均值
    df = dfSale.drop_duplicates(['userID'], keep='last')[['userID']]
    for each in DLSet.TimeGapSet:
        dfTemp = Gen_ActionCounts(dfSale, DLSet.SPLIT_DAY_NUM - each)
        df = pd.merge(df, dfTemp, on=['userID'], how='left')

    # 拼接用户标签特征
    df = pd.merge(df, dfUser, on=['userID'])
    return df


# 主函数
def Main(dataLink_sale, dataLink_user, storeLink):
    dfSale = LoadData(dataLink_sale)
    dfUser = LoadData(dataLink_user)[['userID', 'hykname', 'sex', 'mdmc', 'shdm', 'hyly']]
    dfFU = GenMergeU(dfSale, dfUser)
    dfFU.round(3).fillna(0).to_csv(storeLink, index=False)


def Run():
    Main(DLSet.saleInfo_Train_L_link, DLSet.userInfo_link, DLSet.feature_U_train_link)
    Main(DLSet.saleInfo_Judge_L_link, DLSet.userInfo_link, DLSet.feature_U_judge_link)


if __name__ == '__main__':
    Run()
    pass


"""
    # df = LoadData(DLSet.test_Link)
    # df = df.groupby(['名称'])['数量'].sum().reset_index()
    
    
    # 统计后k天占此前发生的冒险购比
"""

