"""
    训练LR分类器
"""
import sys ;sys.path.append('../')
from sklearn import metrics
import F_TrainModel.GenDataTool as GDTool
import DataLinkSet as DLSet
import joblib
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# 训练模型
def TrainHyBrid(gd, npRatio, nt, modelLink_gbdt, modelLink_enc, modelLink_lr):
    t1 = time.time()

    # 训练集
    train_X, train_y = gd.Gen_TrainData(npRatio=npRatio, subRatio=1)
    # 等分为两部分
    train_X_gbdt, train_X_lr, train_y_gbdt, train_y_lr = train_test_split(train_X, train_y, test_size=0.5)

    # S1. 训练GBDT分类器
    GBDT_clf = GradientBoostingClassifier(learning_rate=0.05,
                                          n_estimators=nt,
                                          max_depth=7,
                                          subsample=0.65,
                                          max_features="sqrt")
    GBDT_clf.fit(train_X_gbdt, train_y_gbdt)
    print('gbdt has trained')

    # S2. 训练编码器
    enc = OneHotEncoder(categories='auto')
    enc.fit(GBDT_clf.apply(train_X_gbdt)[:, 0:6, 0])
    # enc.fit(GBDT_clf.apply(train_X_gbdt)[:, :, 0])
    print('enc has trained')

    # S3. 训练LR分类器
    LR_clf = LogisticRegression(solver='saga', max_iter=2000, penalty='l1', n_jobs=-1)
    LR_clf.fit(enc.transform(GBDT_clf.apply(train_X_lr)[:, 0:6, 0]), train_y_lr)
    # LR_clf.fit(enc.transform(GBDT_clf.apply(train_X_lr)[:, :, 0]), train_y_lr)
    print('lr has trained')

    t2 = time.time()
    print('train time used %d s' % (t2 - t1))

    # 存储模型
    joblib.dump(GBDT_clf, modelLink_gbdt)
    joblib.dump(enc, modelLink_enc)
    joblib.dump(LR_clf, modelLink_lr)


# 进行预测
def Predict(gd, modelLink_gbdt, modelLink_enc, modelLink_lr, thre=0.5):
    # 加载模型
    GBDT_clf = joblib.load(modelLink_gbdt)
    enc = joblib.load(modelLink_enc)
    LR_clf = joblib.load(modelLink_lr)

    # 测试集
    judge_X, judge_y = gd.Gen_JudgeData(subRatio=1)
    # 预测
    temp = LR_clf.predict_proba(enc.transform(GBDT_clf.apply(judge_X)[:, 0:6, 0]))

    # 评价
    f1Set = []
    coSet = []
    for co in np.arange(0.01, 1, 0.05):
        judge_y_pred = (temp[:, 1] > co)
        f1 = metrics.f1_score(judge_y, judge_y_pred)
        p = metrics.precision_score(judge_y, judge_y_pred)
        r = metrics.recall_score(judge_y, judge_y_pred)
        print('co = %.2f' % co, '  f1 = %.4f' % f1, '  Precision = %.4f' % p, '  Recall = %.4f' % r)

        f1Set.append(f1)
        coSet.append(co)

    # ShowPic(coSet, f1Set, "penalty='l1'", 'hybrid: co -> f1', 'co')
    print('argmax co = %f' % (0.01 + 0.05 * np.argmax(f1Set)))
    return temp[:, 1] > (0.01 + 0.05 * np.argmax(f1Set))


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

    # TrainHyBrid(gd, 15, 400,
    #             DLSet.hybrid_classifier_GBDT_link,
    #             DLSet.hybrid_classifier_ENC_link,
    #             DLSet.hybrid_classifier_LR_link)

    res = Predict(gd,
            DLSet.hybrid_classifier_GBDT_link,
            DLSet.hybrid_classifier_ENC_link,
            DLSet.hybrid_classifier_LR_link)

    gd.Gen_Res(res, DLSet.resLRGBDT_link)