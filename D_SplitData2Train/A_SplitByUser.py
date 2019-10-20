"""
    分割用户
"""
from Tools import *
import DataLinkSet as DLSet

# 将用户表分成两部分，训练集占frac比率
def Split(df, frac):
    dfTrain = df.sample(frac=frac)
    dfJudge = pd.concat([df, dfTrain]).drop_duplicates(keep=False)
    return dfTrain, dfJudge


#　主函数
def Main(dataLink, userLink_train, userLink_judge, frac):
    df = LoadData(dataLink)[['userID']]
    df = df.drop_duplicates(['userID'])

    # 将用户表分成两部分
    dfTrain, dfJudge = Split(df, frac)
    dfTrain.to_csv(userLink_train, index=False)
    dfJudge.to_csv(userLink_judge, index=False)


def Run():
    Main(DLSet.sub_saleInfo_link, DLSet.userInfo_Train_link, DLSet.userInfo_Judge_link, DLSet.SPLIT_FRAC)


if __name__ == '__main__':
    Run()
