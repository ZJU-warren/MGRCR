"""
    生成itemInfo表
"""
from Tools import *
import DataLinkSet as DLSet

# 添加itemID，删除不必要列
def Gen_itemID(df):
    # 选出所有的商品名
    df = df.drop_duplicates(['spname'], keep='last')
    print('涉及的商品数目: ', df.shape[0])

    # 选出所需要的列标
    df = df[['dlId', 'zlId', 'xlId', 'spId', 'csppId', 'lsdj', 'spname']]

    # 生成itemID
    df['tempMark'] = 1
    df['itemID'] = df.groupby(['tempMark']).cumcount() + 1

    return df[DLSet.itemInfo_head]


# 主函数
def Main(dataLink, storeLink):
    dfSale = LoadData(dataLink)

    # 生成新表
    dfNew = Gen_itemID(dfSale)
    dfNew.to_csv(storeLink, index=False)


def Run():
    Main(DLSet.true_cssales_link, DLSet.itemInfo_link)


if __name__ == '__main__':
    # Run()
    pass

