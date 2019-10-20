"""
    将正负样本比调整为 1:NP_RATIO
"""
import sys ;sys.path.append('../')
from Tools import *
import DataLinkSet as DLSet


# 取适当样本比
def Gen_SubData(dfOrg):
    df = dfOrg[dfOrg['class'] == 0]
    # 当前正负样本比
    npOrg = (dfOrg.shape[0] - df.shape[0]) / df.shape[0]
    # 求出所需要的正负样本比
    fracRatio = DLSet.NP_RATIO / npOrg

    for i in range(1, DLSet.TOTAL_LABEL_CLASS):
        dfTemp = dfOrg[dfOrg['class'] == i].sample(frac=fracRatio)
        df = pd.concat([df, dfTemp])

    print(df[df['class'] != 0].shape[0], df[df['class'] == 0].shape[0],
          df[df['class'] != 0].shape[0]/df[df['class'] == 0].shape[0])

    return df


def Main(dataLink, storeLink):
    df = LoadData(dataLink)
    df = Gen_SubData(df)
    df.to_csv(storeLink, index=False)


if __name__ == '__main__':
    Main(DLSet.trainData_cluster_link, DLSet.new_trainData_cluster_link)
