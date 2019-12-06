"""
    选择GBDT超参数
"""
import sys ;sys.path.append('../')
from Tools import *
from sklearn import metrics
import time
import F_TrainModel.GenDataTool as GDTool
import DataLinkSet as DLSet
from sklearn.ensemble import GradientBoostingClassifier


# 正负样本比选择
def Select_NPRatio(gd):
    f1Set = []
    npSet = []

    # 生成评估集
    valid_X, valid_y = gd.Gen_TrainData(subRatio=0.2)

    # 生成训练器
    GBDT_clf = GradientBoostingClassifier(learning_rate=0.05,
                                          n_estimators=200,
                                          max_depth=7,
                                          subsample=0.65,
                                          max_features="sqrt")

    for npRatio in range(1, DLSet.NP_RATIO, 3):
        t1 = time.time()

        # 抽取训练集
        train_X, train_y = gd.Gen_TrainData(npRatio=npRatio, subRatio=0.4)

        # 训练
        GBDT_clf.fit(train_X, train_y)

        # 预测
        valid_y_pred = GBDT_clf.predict(valid_X)

        # 评估
        resF1 = metrics.f1_score(valid_y, valid_y_pred)
        f1Set.append(resF1)
        npSet.append(npRatio)

        print('GBDT_clf [NP ratio = %d] is fitted, [f1 = %f]' % (npRatio, resF1))
        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    # ShowPic(npSet, f1Set, "penalty='l1'", 'GBDT: npRatio -> f1', 'npRatio')
    print('——————START——————')
    for i in range(len(f1Set)):
        print('%d\t%f' % (npSet[i], f1Set[i]))
    print('———————END———————')


# 树的数目选择
def Select_nEstimators_as_LearningRate_0d05(gd, npRatio):
    f1Set = []
    ntSet = []

    # 生成评估集
    valid_X, valid_y = gd.Gen_TrainData(subRatio=0.2)
    # 抽取训练集
    train_X, train_y = gd.Gen_TrainData(npRatio=npRatio, subRatio=0.4)

    for nt in [10, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 400]:
        t1 = time.time()

        # 训练
        # 生成训练器
        GBDT_clf = GradientBoostingClassifier(learning_rate=0.05,
                                              n_estimators=nt,
                                              max_depth=7,
                                              subsample=0.65,
                                              max_features="sqrt")
        GBDT_clf.fit(train_X, train_y)

        # 预测
        valid_y_pred = GBDT_clf.predict(valid_X)

        # 评估
        resF1 = metrics.f1_score(valid_y, valid_y_pred)
        f1Set.append(resF1)
        ntSet.append(nt)

        print('GBDT_clf [nt = %d] is fitted, [f1 = % f]' % (nt, resF1))
        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    # ShowPic(ntSet, f1Set, "penalty='l1'", 'GBDT: nt -> f1', 'nt')
    print('——————START——————')
    for i in range(len(f1Set)):
        print('%d\t%f' % (ntSet[i], f1Set[i]))
    print('———————END———————')


# 阈值选择
def Select_CutOff(gd, npRatio, nt):
    f1Set = []
    coSet = []

    # 生成评估集
    valid_X, valid_y = gd.Gen_TrainData(subRatio=0.2)
    # 抽取训练集
    train_X, train_y = gd.Gen_TrainData(npRatio=npRatio, subRatio=0.4)

    # 训练
    GBDT_clf = GradientBoostingClassifier(learning_rate=0.05,
                                          n_estimators=nt,
                                          max_depth=7,
                                          subsample=0.65,
                                          max_features="sqrt")
    GBDT_clf.fit(train_X, train_y)

    for co in np.arange(0.1, 1, 0.1):
        t1 = time.time()
        # 预测
        valid_y_pred = (GBDT_clf.predict_proba(valid_X)[:, 1] > co)

        # 评估
        resF1 = metrics.f1_score(valid_y, valid_y_pred)
        f1Set.append(resF1)
        coSet.append(co)

        print('GBDT_clf [co = %.3f] is fitted, [f1 = % f]' % (co, resF1))
        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    # ShowPic(coSet, f1Set, "penalty='l1'", 'GBDT: co -> f1', 'co')
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
    Select_nEstimators_as_LearningRate_0d05(gd, 22)
    # Select_CutOff(gd, 15, 300)
