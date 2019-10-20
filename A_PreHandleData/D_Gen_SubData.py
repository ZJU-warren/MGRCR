"""
    生成大小为Ｎ的cssales表子集
"""
from Tools import *
import DataLinkSet as DLSet


def Gen_SubSet(N):
    df = LoadData(DLSet.true_cssales_link)
    N = N if df.shape[0] > N else df.shape[0]
    df[:N].to_csv(DLSet.sub_new_cssales_link, index=False)


def Run():
    # 生成大小为10^5的子集
    Gen_SubSet(100000)


if __name__ == '__main__':
    # Run()
    pass
