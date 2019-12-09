"""
    选择LR超参数
"""
import sys ;sys.path.append('../')
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
        resF1 = metrics.f1_score(valid_y, valid_y_pred)
        f1Set.append(resF1)
        npSet.append(npRatio)
        print('-------------------------------------------')
        print('LR_clf [NP ratio = %d] is fitted, [f1 = %f]' % (npRatio, resF1))
        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    # ShowPic(npSet, f1Set, "penalty='l1'", 'LR: npRatio -> f1', 'npRatio')
    print('——————START——————')
    for i in range(len(f1Set)):
        print('%d\t%f' % (npSet[i], f1Set[i]))
    print('———————END———————')


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
        resF1 = metrics.f1_score(valid_y, valid_y_pred)
        f1Set.append(resF1)
        cSet.append(c)

        print('LR_clf [c = %.3f] is fitted, [f1 = % f]' % (c, resF1))
        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    # ShowPic(cSet, f1Set, "penalty='l1'", 'LR: c -> f1', 'c')
    print('——————START——————')
    for i in range(len(f1Set)):
        print('%d\t%f' % (cSet[i], f1Set[i]))
    print('———————END———————')


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
        resF1 = metrics.f1_score(valid_y, valid_y_pred)
        f1Set.append(resF1)
        coSet.append(co)

        print('LR_clf [co = %.3f] is fitted, [f1 = % f]' % (co, resF1))
        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    # ShowPic(coSet, f1Set, "penalty='l1'", 'LR: co -> f1', 'co')
    print('——————START——————')
    for i in range(len(f1Set)):
        print('%f\t%f' % (coSet[i], f1Set[i]))
    print('———————END———————')


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
    # Select_Regularization_Strength(gd, 7)
    # Select_CutOff(gd, 7, 30)

    Select_Regularization_Strength(gd, 31)
    # Select_CutOff(gd, 7, 30)