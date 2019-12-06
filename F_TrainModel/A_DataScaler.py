"""
    负样本下利用Scaler规范化, 并kmeans++聚类便于下采样
"""
import sys ;sys.path.append('../')
from Tools import *
import DataLinkSet as DLSet
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
import pickle


# 分别存储label = 0 和 label = 1的数据
def Split_y(dataLink, storeLink_L0, storeLink_L1):
    # 清空文件
    f = open(storeLink_L0, "w");    f.truncate()
    f = open(storeLink_L1, "w");    f.truncate()

    # 分块读取
    batch = 0
    for df in pd.read_csv(open(dataLink, 'r'), chunksize=150000):
        try:
            # 按label不同分开
            # 烦人的batch, 导致每次都重新输一次头
            if batch == 0:
                df[df['label'] == 0].to_csv(storeLink_L0, index=False, mode='a')
                df[df['label'] == 1].to_csv(storeLink_L1, index=False, mode='a')
            else:
                df[df['label'] == 0].to_csv(storeLink_L0, index=False, header=None, mode='a')
                df[df['label'] == 1].to_csv(storeLink_L1, index=False, header=None, mode='a')

            batch += 1
            print('chunk %d done.' % batch)
        except StopIteration:
            print('finish.')
            break


def StandardScale(dataLink, dfU, dfI, dfUI):
    scaler = preprocessing.StandardScaler()

    # 分块读取
    batch = 0
    for df in pd.read_csv(open(dataLink, 'r'), chunksize=150000):
        try:
            # 按label不同分开
            orgSize = df.shape[0]
            df = pd.merge(df, dfU, on=['userID'], how='left').fillna(0)
            df = pd.merge(df, dfI, on=['itemID'], how='left').fillna(0)
            # df = pd.merge(df, dfUI, on=['userID', 'itemID'], how='left').fillna(0)
            if orgSize != df.shape[0]:
                print('------ ! -------')
            df = df.drop(columns=['userID', 'itemID', 'label'])

            # 分部拟合
            scaler.partial_fit(df.values)

            batch += 1
            print('chunk %d done.' % batch)
        except StopIteration:
            print('finish.')
            break

    return scaler


def Cluster(dataLink, dfU, dfI, dfUI, scaler):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=DLSet.TOTAL_LABEL_CLASS,
                          batch_size=500, reassignment_ratio=10 ** -4)
    labelClass = []

    # 分块读取
    batch = 0
    for df in pd.read_csv(open(dataLink, 'r'), chunksize=150000):
        try:
            # 按label不同分开
            orgSize = df.shape[0]
            df = pd.merge(df, dfU, on=['userID'], how='left').fillna(0)
            df = pd.merge(df, dfI, on=['itemID'], how='left').fillna(0)
            # df = pd.merge(df, dfUI, on=['userID', 'itemID'], how='left').fillna(0)
            if orgSize != df.shape[0]:
                print('------ ! -------')
            df = df.drop(columns=['userID', 'itemID', 'label'])

            # 利用scaler转换
            standardized_train_X = scaler.transform(df.values)
            # 分部拟合
            mbk.partial_fit(standardized_train_X)
            # 添加这部分的标签结果
            labelClass = np.append(labelClass, mbk.labels_)

            batch += 1
            print('chunk %d done.' % batch)
        except StopIteration:
            print('finish.')
            break

    labelClass = [int(each) + 1 for each in labelClass]
    return labelClass


def Main(dataLink_UILabel, dataLink_FU, dataLink_FI, dataLink_FUI,
         storeLink_L0, storeLink_L1, storeLink_scaler, storeLink_cluster):
    # 加载特征
    dfU = LoadData(dataLink_FU)
    dfI = LoadData(dataLink_FI)
    dfUI = LoadData(dataLink_FUI)

    # 存储label = 0, 1的数据
    print('------------------- Split_y ---------------------')
    Split_y(dataLink_UILabel, storeLink_L0, storeLink_L1)

    # 规范化
    print('------------------- StandardScale ---------------------')
    scaler = StandardScale(storeLink_L0, dfU, dfI, dfUI)
    # 持久化scaler用于测试集
    pickle.dump(scaler, open(storeLink_scaler, 'wb'))

    # 聚类
    print('------------------- Cluster ---------------------')
    labelClass = Cluster(storeLink_L0, dfU, dfI, dfUI, scaler)

    df0 = LoadData(storeLink_L0)
    df1 = LoadData(storeLink_L1)

    # 标记所属类别
    print(df0.shape[0], len(labelClass))
    df0['class'] = labelClass
    df1['class'] = 0

    df = pd.concat([df0, df1])
    df.to_csv(storeLink_cluster, index=False)


if __name__ == '__main__':
    Main(DLSet.saleInfo_Train_R_UILabel_link,
         DLSet.feature_U_train_link,
         DLSet.feature_I_train_link,
         DLSet.feature_UI_train_link,
         DLSet.trainData_L0_link,
         DLSet.trainData_L1_link,
         DLSet.trainData_scaler_link,
         DLSet.trainData_cluster_link)
