"""
    切割出学习部分和预测部分
    预设前300天学习，预测后66天情况
"""
import sys ;sys.path.append('../')
from Tools import *
import DataLinkSet as DLSet


# 根据切分点划分出左右部分
def Gen_LR_part(dataLink, storeLink_L, storeLink_R):
    df = LoadData(dataLink)

    # 划分出左右部分
    dfL = df[df.dayID <= DLSet.SPLIT_DAY_NUM]
    dfR = df[df.dayID > DLSet.SPLIT_DAY_NUM]

    # 剔除掉预测部分中此前从未学习过的用户
    dfR = dfR[dfR['userID'].isin(dfL['userID'])]

    dfL.to_csv(storeLink_L, index=False)
    dfR.to_csv(storeLink_R, index=False)


# 按时间排序
def Format(dataLink, storeLink):
    df = LoadData(dataLink)
    df['itemTim'] = df.groupby(['userID', 'dayID', 'itemID']).cumcount() + 1
    df = df.drop_duplicates(['userID', 'dayID', 'itemID'], keep='last')
    df = df.sort_values(['dayID', 'userID', 'itemID'])
    df.to_csv(storeLink, header=None, index=False, columns=['userID', 'itemID', 'itemTim'])


def SortByUser(dataLink, storeLink):
    df = pd.read_csv(dataLink, header=None, names=['userID', 'itemID', 'itemTim'])
    # df = df.drop_duplicates(['userID', 'itemID'], keep='last')
    df = df.sort_values(['userID'])
    df.to_csv(storeLink, header=None, index=False)


def Run():
    # Gen_LR_part(DLSet.saleInfo_Train_link, DLSet.saleInfo_Train_L_link, DLSet.saleInfo_Train_R_link)
    # Gen_LR_part(DLSet.saleInfo_Judge_link, DLSet.saleInfo_Judge_L_link, DLSet.saleInfo_Judge_R_link)

    # Format(DLSet.saleInfo_Train_L_link, DLSet.sorted_saleInfo_Train_L_link)
    # Format(DLSet.saleInfo_Train_R_link, DLSet.sorted_saleInfo_Train_R_link)
    # Format(DLSet.saleInfo_Judge_L_link, DLSet.sorted_saleInfo_Judge_L_link)
    # Format(DLSet.saleInfo_Judge_R_link, DLSet.sorted_saleInfo_Judge_R_link)

    SortByUser(DLSet.sorted_saleInfo_Judge_R_link, DLSet.sortedByUser_saleInfo_Judge_R_link)


if __name__ == '__main__':
    Run()
