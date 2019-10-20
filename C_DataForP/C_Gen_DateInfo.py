"""
    生成dateInfo表
"""
from Tools import *
import DataLinkSet as DLSet


# 添加dayID
def Gen_dayID(df):
    # 选出所有的用户ID
    df = df.drop_duplicates(['dateId'], keep='last')
    print('涉及的日期数目: ', df.shape[0])

    # 生成dayID
    df['tempMark'] = 1
    df['dayID'] = df.groupby(['tempMark']).cumcount() + 1

    # 更改hyId列名
    df = df[['dateId', 'dayID', 'year', 'month', 'dayofm', 'quarter', 'dayofw', 'type']]
    df.columns = ['dateId', 'dayID', 'year', 'month', 'day', 'quarter', 'dayofw', 'type']

    return df


# 生成周末和假日的01标记
def Gen_dayLabel(df):
    df['isWeekend'] = df['dayofw'].apply(lambda x: 1 if x in [6, 7] else 0)
    df['isHoliday'] = df['type'].apply(lambda x: 0 if x in ['普通日', '周末'] else 1)
    return df[DLSet.dateInfo_head]


def Main(dataLink, storeLink):
    df = LoadData(dataLink)

    # 生成dayID和特殊日期标记
    df = Gen_dayID(df)
    df = Gen_dayLabel(df)

    df.to_csv(storeLink, index=False)


def Run():
    Main(DLSet.dateData_link, DLSet.dateInfo_link)


if __name__ == '__main__':
    Run()


