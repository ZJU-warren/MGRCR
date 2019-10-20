"""
    按用户分割出训练集和测试集
"""
import sys ;sys.path.append('../')
from Tools import *
import DataLinkSet as DLSet

#　主函数
def Gen_SubSaleInfo(saleLink, userLink, storeLink):
    dfSale = LoadData(saleLink)
    dfUser = LoadData(userLink)

    df = pd.merge(dfSale, dfUser, on=['userID'])
    print(df.shape[0], dfUser.shape[0], df.shape[0] / dfUser.shape[0])
    df.to_csv(storeLink, index=False)


def Run():
    Gen_SubSaleInfo(DLSet.sub_saleInfo_link, DLSet.userInfo_Train_link, DLSet.saleInfo_Train_link)
    Gen_SubSaleInfo(DLSet.sub_saleInfo_link, DLSet.userInfo_Judge_link, DLSet.saleInfo_Judge_link)


if __name__ == '__main__':
    Run()
