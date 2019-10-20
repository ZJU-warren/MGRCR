"""
    计算给定关键字在指定表中出现次数
"""
from Tools import *
import DataLinkSet as DLSet


# 计算给定关键字在指定表中出现次数
def CalNumOfAttr(dataLink, attrSet):
    df = LoadData(dataLink)
    print('org shape[0] =', df.shape[0])

    for attr in attrSet:
        dfTemp = df.drop_duplicates([attr], keep='last')
        print('[ ]', attr, ': shape[0] =', dfTemp.shape[0])


# 计算给定关键字集合在指定表中出现次数
def CalNumOfAttrSet(dataLink, attrSet):
    df = LoadData(dataLink)
    print('org shape[0] =', df.shape[0])

    dfTemp = df.drop_duplicates(attrSet, keep='last')
    print(attrSet, ': shape[0] =', dfTemp.shape[0])

if __name__ == '__main__':
    print('------------------csfkmxDim--------------------------')
    # CalNumOfAttr(DLSet.new_csfkmxDim_link, DLSet.csfkmxDim_head)
    print('------------------cssales--------------------------')
    # CalNumOfAttr(DLSet.new_cssales_link, DLSet.cssales_head)
    print('------------------dateDim--------------------------')
    # CalNumOfAttr(DLSet.new_dateDim_link, DLSet.dateDim_head)
    print('------------------hyDim--------------------------')
    # CalNumOfAttr(DLSet.new_hyDim_link, DLSet.hyDim_head)
    print('------------------sub_new_cssales--------------------------')
    # CalNumOfAttr(DLSet.sub_new_cssales_link, DLSet.cssales_head)
    # CalNumOfAttrSet(DLSet.true_cssales_link, ['hyId', 'csfkmxId'])
    CalNumOfAttrSet(DLSet.sub_saleInfo_link, ['userID'])
