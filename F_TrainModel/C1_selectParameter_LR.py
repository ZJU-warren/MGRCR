"""
    选择LR超参数
"""
from Tools import *
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import time
import F_TrainModel.GenDataTool as GDTool
import DataLinkSet as DLSet


# 正负样本比选择
def Select_NPRatio(gd):
    f1Set = []
    npSet = []

    # 生成评估集
    valid_X, valid_y = gd.Gen_TrainData(subRatio=0.2)

    # 生成训练器
    LR_clf = LogisticRegression(solver='saga', penalty='l1', n_jobs=-1)

    for npRatio in range(1, DLSet.NP_RATIO, 3):
        t1 = time.time()

        # 抽取训练集
        train_X, train_y = gd.Gen_TrainData(npRatio=npRatio, subRatio=0.4)

        # 训练
        LR_clf.fit(train_X, train_y)

        # 预测
        valid_y_pred = LR_clf.predict(valid_X)

        # 评估
        f1Set.append(metrics.f1_score(valid_y, valid_y_pred))
        npSet.append(npRatio)

        print('LR_clf [NP ratio = %d] is fitted' % npRatio)
        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    ShowPic(npSet, f1Set, "penalty='l1'", 'LR: npRatio -> f1', 'npRatio')


# 惩罚程度选择
def Select_Regularization_Strength(gd, npRatio):
    f1Set = []
    cSet = []

    # 生成评估集
    valid_X, valid_y = gd.Gen_TrainData(subRatio=0.2)
    # 抽取训练集
    train_X, train_y = gd.Gen_TrainData(npRatio=npRatio, subRatio=0.4)

    for c in range(10, 80, 10):
        t1 = time.time()

        # 训练
        LR_clf = LogisticRegression(C=c, solver='saga', penalty='l1', n_jobs=-1)
        LR_clf.fit(train_X, train_y)

        # 预测
        valid_y_pred = LR_clf.predict(valid_X)

        # 评估
        f1Set.append(metrics.f1_score(valid_y, valid_y_pred))
        cSet.append(c)

        print('LR_clf [c = %.3f] is fitted' % c)
        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    ShowPic(cSet, f1Set, "penalty='l1'", 'LR: c -> f1', 'c')


# 阈值选择
def Select_CutOff(gd, npRatio, c):
    f1Set = []
    coSet = []

    # 生成评估集
    valid_X, valid_y = gd.Gen_TrainData(subRatio=0.2)
    # 抽取训练集
    train_X, train_y = gd.Gen_TrainData(npRatio=npRatio, subRatio=0.4)

    # 训练
    LR_clf = LogisticRegression(C=c, solver='saga', penalty='l1', n_jobs=-1)
    LR_clf.fit(train_X, train_y)

    for co in np.arange(0.1, 1, 0.1):
        t1 = time.time()
        # 预测
        valid_y_pred = (LR_clf.predict_proba(valid_X)[:, 1] > co)

        # 评估
        f1Set.append(metrics.f1_score(valid_y, valid_y_pred))
        coSet.append(co)

        print('LR_clf [co = %.3f] is fitted' % co)
        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    ShowPic(coSet, f1Set, "penalty='l1'", 'LR: co -> f1', 'co')


if __name__ == '__main__':
    # 生成 gd
    gd = GDTool.GenDataTool(
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
    # Select_NPRatio(gd)
    # Select_Regularization_Strength(gd, 15)
    Select_CutOff(gd, 15, 50)
