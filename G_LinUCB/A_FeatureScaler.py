"""
    特征归一化
"""
import sys ;sys.path.append('../')
import DataLinkSet as DLSet
from sklearn import preprocessing
from Tools import *
import pickle


def TrainStandardScale(scaler, dataLink, idStr):
    # 分块读取
    batch = 0
    for df in pd.read_csv(open(dataLink, 'r'), chunksize=150000):
        try:
            # 按label不同分开
            if idStr == 'userID':
                df = df.drop(columns=[idStr])[DLSet.userFeatureSet]
            else:
                df = df.drop(columns=[idStr])[DLSet.itemFeatureSet]
            # 分部拟合
            scaler.partial_fit(df.values)

            batch += 1
            print('chunk %d done.' % batch)
        except StopIteration:
            print('finish.')
            break

    return scaler


def Main():
    scalerU = preprocessing.StandardScaler()
    TrainStandardScale(scalerU, DLSet.feature_U_train_link, 'userID')
    TrainStandardScale(scalerU, DLSet.feature_U_judge_link, 'userID')
    StoreObj(scalerU, DLSet.scaler_U_link)

    scalerI = preprocessing.StandardScaler()
    TrainStandardScale(scalerI, DLSet.feature_I_train_link, 'itemID')
    TrainStandardScale(scalerI, DLSet.feature_I_judge_link, 'itemID')
    StoreObj(scalerI, DLSet.scaler_I_link)


def Run():
    Main()


if __name__ == '__main__':
    Run()
