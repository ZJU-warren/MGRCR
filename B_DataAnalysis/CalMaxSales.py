"""
    求热门销量商品
"""
from Tools import *
import DataLinkSet as DLSet


# 求热门销量商品
def MaxSales(dataLink):
    df = LoadData(dataLink)
    df['itemSum'] = df.groupby(['spname']).cumcount() + 1
    df = df.drop_duplicates(['spname'], keep='last')[['itemSum', 'spname', 'spId']]
    df = df.sort_values(by=['itemSum'], ascending=False)
    print(df.head(30))


if __name__ == '__main__':
    MaxSales(DLSet.true_cssales_link)
    pass
