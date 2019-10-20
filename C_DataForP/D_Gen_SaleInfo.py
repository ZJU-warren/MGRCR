"""
    生成saleInfo表
"""
import sys ;sys.path.append('../')
from Tools import *
import DataLinkSet as DLSet


# 替换外键为标准ID
def XChangeID(dfSale, dfItem, dfDate):
    # 拼接后返回替换的ID列
    df = pd.merge(dfSale, dfItem, on=['spname'])
    df = pd.merge(df, dfDate, on=['dateId'])
    df = df[['hyId', 'itemID', 'dayID', 'xssl', 'lsdj', 'zkje']]
    df.columns = ['userID', 'itemID', 'dayID', 'xssl', 'lsdj', 'zkje']
    return df


# 删掉部分干扰性商品
def DelSomeItems(df, itemSet_str):
    print('org', df.shape[0])
    df = df[~df['spname'].str.contains(itemSet_str)]
    print('now', df.shape[0])
    return df


# 生成数值属性和label
def Gen_Value_Label(df):
    df['#item'] = df['xssl']
    df['#cost'] = df['xssl'] * df['lsdj'] - df['zkje']
    df['isPromote'] = df['zkje'].apply(lambda x: 1 if x > 0.0 else 0)
    return df[DLSet.saleInfo_head]


# 主函数
def Main(dataLink_Sale, dataLink_Item, dataLink_Date, storeLink):
    dfSale = LoadData(dataLink_Sale)[['hyId', 'dateId', 'spname', 'xssl', 'lsdj', 'zkje']]
    dfItem = LoadData(dataLink_Item)[['spname', 'itemID']]
    dfDate = LoadData(dataLink_Date)[['dateId', 'dayID']]

    # 删掉部分干扰性商品
    dfSale = DelSomeItems(dfSale, '购物袋')
    # 替换外键为标准ID
    df = XChangeID(dfSale, dfItem, dfDate)
    # 生成数值属性和label
    df = Gen_Value_Label(df)

    df.to_csv(storeLink, index=False)


def Run():
    Main(DLSet.true_cssales_link,
         DLSet.itemInfo_link,
         DLSet.dateInfo_link,
         DLSet.saleInfo_link)


if __name__ == '__main__':
    Run()


