"""
    生成实际所需要用到的时间信息表 dateData
"""
import sys ;sys.path.append('../')
from Tools import *
import DataLinkSet as DLSet

# 求出cssales实际设计的日期
def Gen_dateId_InSales():
    df = LoadData(DLSet.new_cssales_link)

    # 求出cssales实际涉及的日期并存储
    df['dateCount'] = df.groupby(['dateId']).cumcount() + 1
    df = df.drop_duplicates(['dateId'], 'last')[['dateId', 'dateCount']]

    # 倒序排列
    df = df.sort_values(by=['dateCount'], ascending=False)
    df.to_csv(DLSet.dateNum_In_cssales_link, index=False)


# 将所涉及日期与日期信息表合并
def Gen_dateInfo_InSales():
    dfDateID = LoadData(DLSet.dateNum_In_cssales_link)
    dfDateInfo = LoadData(DLSet.new_dateDim_link)

    # 与日期信息表拼接
    df = pd.merge(dfDateID, dfDateInfo, on=['dateId'])

    # 按时间正序列排放
    df = df.sort_values(by=['date'])
    df.to_csv(DLSet.dateData_link, index=False,
              columns=['dateId', 'dateCount', 'year', 'month', 'quarter', 'weekofy', 'dayofm', 'dayofw', 'type'])


def Run():
    Gen_dateId_InSales()
    Gen_dateInfo_InSales()


if __name__ == '__main__':
    # Run()
    pass
