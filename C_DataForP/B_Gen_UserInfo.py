"""
    生成userInfo表
"""
import sys ;sys.path.append('../')
from Tools import *
import DataLinkSet as DLSet


def Gen_userID(dfSale, dfUser):
    # 选出所有的用户ID
    dfSale = dfSale.drop_duplicates(['hyId'], keep='last')
    print('涉及的用户数目: ', dfSale.shape[0])

    df = pd.merge(dfSale, dfUser, on=['hyId'])
    print('有记录的用户数目: ', df.shape[0])

    # 选出所需要的列标
    df = df[['hyId', 'hykname', 'sex', 'csrq', 'jkrq', 'mdmc', 'shdm', 'hyly']]

    # 更改hyId列名
    df.columns = DLSet.userInfo_head
    return df


def MapStr2Int(dfOrg, str, mapLink):
    df = dfOrg.drop_duplicates([str], keep='last')[[str]]

    # 生成映射表
    df['tempMark'] = 1
    df[str + 'ID'] = df.groupby(['tempMark']).cumcount() + 1
    df[[str+'ID', str]].to_csv(mapLink, index=False)

    # 原表和映射表合并
    dfOrg = pd.merge(dfOrg, df, on=[str])
    # print(dfOrg.head(5))

    # 替换后删掉多余列
    dfOrg[str] = dfOrg[str + 'ID']

    # print(dfOrg.head(5))

    dfOrg.drop(columns=[str + 'ID'])
    # print(dfOrg.head(5))

    return dfOrg


# 属性值映射
def XStr2Int(df, mapLink):
    mapSet = ['hykname', 'sex', 'mdmc', 'shdm', 'hyly']
    for each in mapSet:
        df = MapStr2Int(df, each, mapLink % each)
    # df = XDate(df)
    # df = XDate(df)
    return df


# 主函数
def Main(dataLink_Sales, dataLink_User, storeLink, mapLink):
    dfSale = LoadData(dataLink_Sales)
    dfUser = LoadData(dataLink_User)

    # 生成用户ID
    dfNew = Gen_userID(dfSale, dfUser)
    # 非数值换为数值
    dfNew = XStr2Int(dfNew, mapLink)

    dfNew.to_csv(storeLink, index=False)


def Run():
    Main(DLSet.true_cssales_link, DLSet.new_hyDim_link, DLSet.userInfo_link, DLSet.mapStr2Int_str_link)


if __name__ == '__main__':
    Run()


