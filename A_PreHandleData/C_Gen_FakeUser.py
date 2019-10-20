"""
    生成虚假用户
    判定条件： 一年里有15天及以上结算超过7次，判定为虚假用户
"""
from Tools import *
import DataLinkSet as DLSet


# 计算日交易单表
def Gen_DayBuyCount(dfOrg):
    # 使得单号唯一
    df = dfOrg.drop_duplicates(['hyId', 'csfkmxId'], keep='last')[['hyId', 'csfkmxId', 'dateId']]

    # 计算每个用户每天有多少个单号
    df['dayBuyCount'] = df.groupby(['hyId', 'dateId']).cumcount() + 1
    df = df.drop_duplicates(['hyId', 'dateId'], keep='last')[['hyId', 'dateId', 'dayBuyCount']]
    return df


# 求得虚假用户
def Gen_FakeUser(dfOrg):
    # 计算日交易单表
    df = Gen_DayBuyCount(dfOrg)

    # 挑出日购物超过 MAX_VAILD_DAYBUYCOUNT 次的用户
    df = df[df.dayBuyCount > DLSet.MAX_VAILD_DAYBUYCOUNT][['hyId']]
    # 计算出他们用户日数
    df['fakeDayCount'] = df.groupby(['hyId']).cumcount() + 1
    df = df.drop_duplicates(['hyId'], keep='last')
    # 挑出可疑日数超过 MIN_FAKE_DAYTOTAL 次的用户
    df = df[df.fakeDayCount >= DLSet.MIN_FAKE_DAYTOTAL][['hyId']]

    return df


def Remove_FakeUser(dfOrg, dfFake):
    dfOrg = dfOrg[~dfOrg.hyId.isin(dfFake.hyId)]
    return dfOrg


def Main(dataLink, storeLink_fakeU, storeLink_trueSale):
    # 获得销售表和去除虚假用户后的销售表
    df = LoadData(dataLink)
    dfFake = Gen_FakeUser(df)

    # 存储虚假用户
    dfFake.to_csv(storeLink_fakeU, index=False)

    # 存储剔除虚假用户后的销售表
    df = Remove_FakeUser(df, dfFake)
    df.to_csv(storeLink_trueSale, index=False)


def Run():
    # Main(DLSet.sub_new_cssales_link, DLSet.sub_fakeUser_link, DLSet.true_cssales_link)
    Main(DLSet.new_cssales_link, DLSet.fakeUser_link, DLSet.true_cssales_link)


if __name__ == '__main__':
    Run()
