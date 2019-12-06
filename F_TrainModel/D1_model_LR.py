"""
    训练LR分类器
"""
import sys ;sys.path.append('../')
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import F_TrainModel.GenDataTool as GDTool
import DataLinkSet as DLSet
import joblib
import time
import numpy as np


# 训练模型
def TrainLR(gd, npRatio, c, modelLink):
    t1 = time.time()

    # 训练集
    train_X, train_y = gd.Gen_TrainData(npRatio=npRatio, subRatio=1)

    # 训练
    LR_clf = LogisticRegression(C=c, solver='saga', penalty='l1', n_jobs=-1)
    LR_clf.fit(train_X, train_y)

    t2 = time.time()
    print('train time used %d s' % (t2 - t1))

    # 存储模型
    joblib.dump(LR_clf, modelLink)


# 进行预测
def Predict(gd, modelLink, thre=0.5):
    # 加载模型
    LR_clf = joblib.load(modelLink)

    # 测试集
    judge_X, judge_y = gd.Gen_JudgeData(subRatio=1)

    # 预测
    temp = LR_clf.predict_proba(judge_X)

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
    # TrainLR(gd, 15, 50, DLSet.classifier_LR_link)
    res = Predict(gd, DLSet.classifier_LR_link, 0.6)
    gd.Gen_Res(res, DLSet.resLR_link)
