""" 文件夹地址 """
DataSetLink = '../../DataSet'               # 数据仓库总地址
# DataSetLink = '../DataSet'               # 数据仓库总地址
OrgSetLink = DataSetLink + '/OrgSet'        # 原始数据仓库地址
CleanSetLink = DataSetLink + '/CleanSet'    # 清洗数据仓库地址
ActualUsedSetLink = DataSetLink + '/ActualUsedSet'    # 程序主数据仓库地址
MapSetLink = DataSetLink + '/MapSet'        # 映射数据仓库地址
FeatureSetLink = DataSetLink + '/FeatureSet'    # 特征数据仓库地址
TrainModelSetLink = DataSetLink + '/TrainModel'    # 模型训练数据仓库地址
LinUCBSetLink = DataSetLink + '/LinUCB'     # LinUCB仓库地址

# """ -------------------------------------------- 原始数据地址 -------------------------------------------- """
# 原始数据
csfkmxDim_link = OrgSetLink + '/csfkmxDim'
cssales_link = OrgSetLink + '/cssales'
dateDim_link = OrgSetLink + '/dateDim'
hyDim_link = OrgSetLink + '/hyDim'

# 添加head并改变分割符号
new_csfkmxDim_link = OrgSetLink + '/new_csfkmxDim'
new_cssales_link = OrgSetLink + '/new_cssales'
new_dateDim_link = OrgSetLink + '/new_dateDim'
new_hyDim_link = OrgSetLink + '/new_hyDim'

#　原始数据表头
csfkmxDim_head = ['csfkmxId', 'mdcd', 'mdmc', 'jysj', 'sktno', 'jlbh',
                  'person_code', 'skfs', 'skname', 'skje']
cssales_head = ['hyId', 'bbgmdId', 'dateId', 'csfkmxId', 'dlId', 'zlId', 'xlId', 'spId', 'csppId', 'spname',
                'lsdj', 'xssl', 'zkje']
dateDim_head = ['dateId', 'date', 'year', 'month', 'quarter', 'weekofy', 'dayofm', 'dayofw',
                'hourofd', 'minuteofh', 'type']
hyDim_head = ['hyId', 'kid', 'hyk_no', 'hykname', 'sex', 'csrq',
              'jkrq', 'kstatus', 'hystatus', 'mdmc', 'shdm', 'hyly']


# """ -------------------------------------------- 清洗数据地址 -------------------------------------------- """
dateNum_In_cssales_link = CleanSetLink + '/dateNum_In_cssales'
dateData_link = CleanSetLink + '/dateData'
fakeUser_link = CleanSetLink + '/fakeUser'
true_cssales_link = CleanSetLink + '/true_cssales'


# """ -------------------------------------------- 程序主数据数据地址 -------------------------------------------- """
# 主数据
itemInfo_link = ActualUsedSetLink + '/itemInfo'
userInfo_link = ActualUsedSetLink + '/userInfo'
saleInfo_link = ActualUsedSetLink + '/saleInfo'
dateInfo_link = ActualUsedSetLink + '/dateInfo'

# 抽取出最适宜的商品子集对应的销售记录
sub_saleInfo_link = ActualUsedSetLink + '/sub_saleInfo'

# 主数据表头
# noticed! 以spname为关键字建立itemID
itemInfo_head = ['itemID', 'dlId', 'zlId', 'xlId', 'spId', 'csppId', 'spname', 'lsdj']
# noticed! hyId改为关键字userID
userInfo_head = ['userID', 'hykname', 'sex', 'csrq', 'jkrq', 'mdmc', 'shdm', 'hyly']
# noticed! dayID按时间序dateID映射
dateInfo_head = ['dayID', 'dateId', 'year', 'month', 'day', 'quarter', 'isWeekend', 'isHoliday']
# noticed! saleInfo based on former
saleInfo_head = ['userID', 'itemID', '#item', '#cost', 'isPromote', 'dayID']

# 训练集和测试集合
userInfo_Train_link = ActualUsedSetLink + '/userInfo_Train'
userInfo_Judge_link = ActualUsedSetLink + '/userInfo_Judge'
saleInfo_Train_link = ActualUsedSetLink + '/saleInfo_Train'
saleInfo_Judge_link = ActualUsedSetLink + '/saleInfo_Judge'

# 划分学习和预测部分
saleInfo_Train_L_link = ActualUsedSetLink + '/saleInfo_L_Train'
saleInfo_Train_R_link = ActualUsedSetLink + '/saleInfo_R_Train'
saleInfo_Judge_L_link = ActualUsedSetLink + '/saleInfo_L_Judge'
saleInfo_Judge_R_link = ActualUsedSetLink + '/saleInfo_R_Judge'

sorted_saleInfo_Train_L_link = ActualUsedSetLink + '/sorted_saleInfo_L_Train'
sorted_saleInfo_Train_R_link = ActualUsedSetLink + '/sorted_saleInfo_R_Train'
sorted_saleInfo_Judge_L_link = ActualUsedSetLink + '/sorted_saleInfo_L_Judge'
sorted_saleInfo_Judge_R_link = ActualUsedSetLink + '/sorted_saleInfo_R_Judge'
sortedByUser_saleInfo_Judge_R_link = ActualUsedSetLink + '/sortedByUser_saleInfo_R_Judge'

saleInfo_Train_R_UI_link = ActualUsedSetLink + '/saleInfo_R_UI_Train'
saleInfo_Train_R_UILabel_link = ActualUsedSetLink + '/saleInfo_R_UILabel_Train'
saleInfo_Judge_R_UI_link = ActualUsedSetLink + '/saleInfo_R_UI_Judge'
saleInfo_Judge_R_UILabel_link = ActualUsedSetLink + '/saleInfo_R_UILabel_Judge'

# """ -------------------------------------------- 映射数据地址 -------------------------------------------- """
mapStr2Int_str_link = MapSetLink + '/mapInfo_%s'

# """ -------------------------------------------- 特征数据地址 -------------------------------------------- """
# FeatureSetLink
feature_U_train_link = FeatureSetLink + '/feature_U_train'
feature_U_judge_link = FeatureSetLink + '/feature_U_judge'
feature_I_train_link = FeatureSetLink + '/feature_I_train'
feature_I_judge_link = FeatureSetLink + '/feature_I_judge'
feature_UI_train_link = FeatureSetLink + '/feature_UI_train'
feature_UI_judge_link = FeatureSetLink + '/feature_UI_judge'
test_Link = FeatureSetLink + '/test'


# """ -------------------------------------------- 模型训练数据数据地址 -------------------------------------------- """
trainData_L0_link = TrainModelSetLink + '/trainData_L0'
trainData_L1_link = TrainModelSetLink + '/trainData_L1'

trainData_scaler_link = TrainModelSetLink + '/trainData_scaler'     # 规范化器
trainData_cluster_link = TrainModelSetLink + '/trainData_cluster'   # 多类别化 saleInfo_Train_R_UILabel
new_trainData_cluster_link = TrainModelSetLink + '/new_trainData_cluster'   # 调整正负样本比

classifier_LR_link = TrainModelSetLink + '/classifier_LR'           # LR分类器
classifier_GBDT_link = TrainModelSetLink + '/classifier_GBDT'       # GBDT分类器

hybrid_classifier_LR_link = TrainModelSetLink + '/hybrid_classifier_LR'       # hybrid LR分类器
hybrid_classifier_ENC_link = TrainModelSetLink + '/hybrid_classifier_ENC'     # hybrid 编码器
hybrid_classifier_GBDT_link = TrainModelSetLink + '/hybrid_classifier_GBDT'   # hybrid GBDT分类器

# """ -------------------------------------------- 子集合数据 -------------------------------------------- """
SubSetLink = DataSetLink + '/SubSet'
sub_new_cssales_link = SubSetLink + '/sub_new_cssales'
sub_fakeUser_link = SubSetLink + '/sub_fakeUser'
sub_true_cssales_link = SubSetLink + '/sub_true_cssales'

resLinUCB_link = SubSetLink + '/resLinUCB_%d'
resLR_link = SubSetLink + '/resLR_%d'
resGBDT_link = SubSetLink + '/resGBDT_%d'
resLRGBDT_link = SubSetLink + '/resLRGBDT_%d'


# """ -------------------------------------------- LinUCB数据 -------------------------------------------- """
modelObj_link = LinUCBSetLink + '/modelObj'
scaler_U_link = LinUCBSetLink + '/scaler_U'
scaler_I_link = LinUCBSetLink + '/scaler_I'
userFeatureSet = ['hykname', 'sex', 'mdmc', 'shdm', 'hyly', 'U_dayCount_After_0',
                  'U_avg_numCount_After_0', 'U_avg_payCount_After_0', 'U_avg_pmtRatio_After_0']
itemFeatureSet = ['lsdj', 'I_numCount_After_0', 'I_payCount_After_0',
                  'I_usrCount_After_0', 'I_timCount_After_0', 'I_pmtCount_After_0',
                  'I_avg_numCount_After_0', 'I_avg_payCount_After_0', 'I_avg_pmtRatio_After_0']

# """ -------------------------------------------- 预设常数 -------------------------------------------- """
MAX_VAILD_DAYBUYCOUNT = 7   # 每日虚假购物极限
MIN_FAKE_DAYTOTAL = 15      # 年虚假日数极限
SPLIT_FRAC = 0.9        # 训练集用户占比
SPLIT_DAY_NUM = 300     # 学习天数的数值

# TOP_ITEM_NUM = 10000        # Top --> |sale| = 42809254, |user| = 352668
# TOP_ITEM_NUM = 5000         # Top --> |sale| = 37961344, |user| = 351256

# 抽取数据子集
TOP_ITEM_NUM = 10500        # 选取item数
MIN_ITEM_SALE_LIMIT = 40

TOP_USER_NUM = 30000        # 选取user数
MIN_USER_SALE_LIMIT = 30

TOTAL_LABEL_CLASS = 1000    # 1000个聚类点

NP_RATIO = 35

# 统计时间分割
TimeGapSet = [300, 240, 180, 120, 90, 60, 45, 30, 20, 10, 5, 3]

if __name__ == '__main__':
    f = open(csfkmxDim_link[1:], 'r')
    print(f.readline())
    f.close()


