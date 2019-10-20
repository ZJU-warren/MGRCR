"""
    对原有表的sep处理，并加入表头
"""
import pandas as pd
import DataLinkSet as DLSet


# 读入并加上表头
def LoadData(dataLink, nameArray):
    df = pd.read_csv(dataLink, header=None, names=nameArray, sep='\t')
    return df


# 改变sep
def Xchange(dataLink, nameArray, storeLink):
    df = LoadData(dataLink, nameArray)
    df.to_csv(storeLink, index=False)


# 主函数
def Run():
    Xchange(DLSet.csfkmxDim_link, DLSet.csfkmxDim_head, DLSet.new_csfkmxDim_link)
    Xchange(DLSet.cssales_link, DLSet.cssales_head, DLSet.new_cssales_link)
    Xchange(DLSet.dateDim_link, DLSet.dateDim_head, DLSet.new_dateDim_link)
    Xchange(DLSet.hyDim_link, DLSet.hyDim_head, DLSet.new_hyDim_link)


if __name__ == '__main__':
    # Run()
    print('nothing happens')

