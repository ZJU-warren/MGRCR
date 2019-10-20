import DataLinkSet as DLSet

import A_PreHandleData.A_FixOrgData as AA
import A_PreHandleData.B_Gen_DateData as AB
import A_PreHandleData.C_Gen_FakeUser as AC
import A_PreHandleData.D_Gen_SubData as AD

import C_DataForP.A_Gen_ItemInfo as CA
import C_DataForP.B_Gen_UserInfo as CB
import C_DataForP.C_Gen_DateInfo as CC
import C_DataForP.D_Gen_SaleInfo as CD
import C_DataForP.E_Gen_Sub_SaleInfo as CE

import D_SplitData2Train.A_SplitByUser as DA
import D_SplitData2Train.B_SplitSaleInfo as DB
import D_SplitData2Train.C_SplitSaleInfo_LR as DC
import D_SplitData2Train.D_Gen_UILabel as DD

import E_FeatureProject.A_Feature_U as EA
import E_FeatureProject.B_Feature_I as EB
import E_FeatureProject.C_Feature_UI as EC

import G_LinUCB.A_FeatureScaler as GA
import G_LinUCB.B_Train as GB


def APart():
    # 对原有表的sep处理，并加入表头
    AA.Run()
    # 生成实际所需要用到的时间信息表 dateData
    AB.Run()
    # 生成虚假用户
    AC.Run()
    # 生成大小为Ｎ的cssales表子集
    AD.Run()


def CPart():
    # 生成itemInfo表
    # CA.Run()
    # 生成userInfo表
    # CB.Run()
    # 生成dateInfo表
    # CC.Run()
    # 生成saleInfo表
    # CD.Run()
    print('生成sub_saleInfo表')
    CE.Run()


def DPart():
    print('分割用户')
    DA.Run()
    print('按用户分割出训练集和测试集')
    DB.Run()
    print('切割出学习部分和预测部分')
    DC.Run()
    print('生成UI, UILabel用于评估')
    DD.Run()


def EPart():
    print('构建U的交互特征，并与其标签特征合并')
    EA.Run()
    print('构建I的交互特征，并与其标签特征合并')
    EB.Run()
    print('构建UI的交互特征, 并生成UI&U, UI&I特征')
    EC.Run()


def GPart():
    print('特征归一化')
    # GA.Run()
    print('训练模型')
    GB.Run()


def Main():
    # APart()
    # CPart()
    # DPart()
    # EPart()
    GPart()


if __name__ == '__main__':
    Main()
