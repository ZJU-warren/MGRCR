"""
    负样本下利用Scaler规范化, 并kmeans++聚类便于下采样
"""
import sys ;sys.path.append('../')
from Tools import *
import DataLinkSet as DLSet
import pickle
from sklearn import metrics


class GenDataTool:
    def __init__(self, Tlink, T_Ulink, T_Ilink, T_UIlink, ScalerLink,
                 Jlink, J_Ulink, J_Ilink, J_UIlink):
        # 训练集数据
        self.__T = LoadData(Tlink)
        self.__dfT_U = LoadData(T_Ulink)
        self.__dfT_I = LoadData(T_Ilink)
        self.__dfT_UI = LoadData(T_UIlink)
        self.scaler = pickle.load(open(ScalerLink, 'rb'))

        # 测试集数据
        self.__J = LoadData(Jlink)
        self.__dfJ_U = LoadData(J_Ulink)
        self.__dfJ_I = LoadData(J_Ilink)
        self.__dfJ_UI = LoadData(J_UIlink)

    # 给定所需要的正负样本比，以及占全训练集比率，生成训练/评估数据
    def Gen_TrainData(self, subRatio=0.1, npRatio=DLSet.NP_RATIO):
        # 正样本抽取
        df = self.__T[self.__T['class'] == 0].sample(frac=subRatio)

        # 负样本先下采样再抽取
        fracRatio = npRatio * subRatio / DLSet.NP_RATIO    # 抽取比率 = (目标正负样本比 / 当前正负样本比) * 抽取率
        for i in range(1, DLSet.TOTAL_LABEL_CLASS):
            dfTemp = self.__T[self.__T['class'] == i].sample(frac=fracRatio)
            df = pd.concat([df, dfTemp])

        # 构建可使用集
        orgSize = df.shape[0]
        df = pd.merge(df, self.__dfT_U, on=['userID'], how='left').fillna(0)
        df = pd.merge(df, self.__dfT_I, on=['itemID'], how='left').fillna(0)
        # df = pd.merge(df, self.__dfT_UI, on=['userID', 'itemID'], how='left').fillna(0)
        if orgSize != df.shape[0]:
            print('------ ! -------')

        # 构建(X, y)
        # 生成 y
        y = df['label'].values
        # 去掉不相关列
        df = df.drop(columns=['userID', 'itemID', 'class', 'label'])
        # 生成并规范 X
        X = df.values
        std_X = self.scaler.transform(X)

        return std_X, y

    # 给定所需要比率，生成评判数据
    def Gen_JudgeData(self, subRatio=1):
        # 样本抽取
        # df0 = self.__J[self.__J['label'] == 0].sample(frac=subRatio)
        # df1 = self.__J[self.__J['label'] == 1].sample(frac=subRatio)
        # df = pd.concat([df0, df1])

        df = self.__J

        # 构建可使用集
        orgSize = df.shape[0]
        df = pd.merge(df, self.__dfJ_U, on=['userID'], how='left').fillna(0)
        df = pd.merge(df, self.__dfJ_I, on=['itemID'], how='left').fillna(0)
        # df = pd.merge(df, self.__dfJ_UI, on=['userID', 'itemID'], how='left').fillna(0)
        if orgSize != df.shape[0]:
            print('------ * -------')

        # 构建(X, y)
        # 生成 y
        y = df['label'].values
        # 去掉不相关列
        df = df.drop(columns=['userID', 'itemID', 'label'])
        # 生成并规范 X
        X = df.values
        std_X = self.scaler.transform(X)

        return std_X, y

    # 生成结果
    def Gen_Res(self, v, storeLink):

        v = pd.DataFrame(v, columns=['pred'])
        dfRes = pd.concat([self.__J, v], axis=1)

        # dfRes['pred'] = 0 if dfRes['pred'] is 'False' else 1

        dfRes['pred'] = dfRes['pred'].astype('int')
        f1 = metrics.f1_score(dfRes['label'].values, dfRes['pred'].values)
        p = metrics.precision_score(dfRes['label'].values, dfRes['pred'].values)
        r = metrics.recall_score(dfRes['label'].values, dfRes['pred'].values)
        print('f1 = %.4f' % f1, '  Precision = %.4f' % p, '  Recall = %.4f' % r)
        dfRes.to_csv(storeLink, header=None, index=False)


if __name__ == '__main__':
    gd = GenDataTool(
        DLSet.new_trainData_cluster_link,
        DLSet.feature_U_train_link,
        DLSet.feature_I_train_link,
        DLSet.feature_UI_train_link,
        DLSet.trainData_scaler_link,

        DLSet.saleInfo_Judge_R_UILabel_link,
        DLSet.feature_U_judge_link,
        DLSet.feature_I_judge_link,
        DLSet.feature_UI_judge_link,
    )

    X, y = gd.Gen_TrainData(0.1, 5)
    print(type(X), type(y))
    print(np.shape(np.nonzero(y)))
    print(X.shape)
