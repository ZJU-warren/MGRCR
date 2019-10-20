"""
    生成sub_saleInfo表
"""
import sys ;sys.path.append('../')
from Tools import *
import DataLinkSet as DLSet
import B_DataAnalysis.CalNumOfAttr as CNAttr


# 生成商品信息表，附带接触过的用户数目
def Gen_Item_UserNum(dfOrg):
    # 去除同一用户重复购买
    df = dfOrg.drop_duplicates(['userID', 'itemID'], keep='last')[['itemID']]
    df['itemCount'] = df.groupby(['itemID']).cumcount() + 1
    # 去除掉重复item
    df = df.drop_duplicates(['itemID'], keep='last')
    # 按热门度排序
    df = df.sort_values(['itemCount'], ascending=False)
    return df


# 生成用户-购物数表
def Gen_User_BuyNum(dfOrg):
    df = dfOrg[['userID']].copy()
    # 生成用户购物次数
    df['userCount'] = df.groupby(['userID']).cumcount() + 1
    # 去除掉重复user
    df = df.drop_duplicates(['userID'], keep='last')
    # 按热门度排序
    df = df.sort_values(['userCount'], ascending=False)
    return df


# 随机选出N个具有至少L个keyWord值的
def Choice_SubKeep(df, N, L, keyWord):
    # 过滤掉低小于L的
    df = df[df[keyWord] > L]
    # 防止越界
    N = N if df.shape[0] > N else df.shape[0]
    # 随机抽取 N 个商品
    return df.sample(n=N)


# 保留Org在Keep中所有的项目
def Keep_SubSale(dfOrg, dfKeep, keyWord):
    df = dfOrg[dfOrg[keyWord].isin(dfKeep[keyWord])]
    return df


# 主函数
def Main(dataLink, storeLink):
    df = LoadData(dataLink)


    # 获得商品-用户数目表
    df_I_UNum = Gen_Item_UserNum(df)
    # 随机选出部分商品
    df_Sub_I = Choice_SubKeep(df_I_UNum, DLSet.TOP_ITEM_NUM, DLSet.MIN_ITEM_SALE_LIMIT, 'itemCount')
    # 保留所选商品saleInfo
    print('+', df.shape[0])
    df = Keep_SubSale(df, df_Sub_I, 'itemID')
    print('-', df.shape[0])


    # 获得用户-购物数目表
    df_U_BNum = Gen_User_BuyNum(df)
    # 随机选出部分用户
    df_Sub_U = Choice_SubKeep(df_U_BNum, DLSet.TOP_USER_NUM, DLSet.MIN_USER_SALE_LIMIT, 'userCount')
    # 保留所选用户对应saleInfo
    print('+', df.shape[0])
    df = Keep_SubSale(df, df_Sub_U, 'userID')
    print('-', df.shape[0])

    df.to_csv(storeLink, index=False)


def Run():
    Main(DLSet.saleInfo_link, DLSet.sub_saleInfo_link)
    CNAttr.CalNumOfAttr(DLSet.sub_saleInfo_link, ['userID', 'itemID'])


if __name__ == '__main__':
    Run()
