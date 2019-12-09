from Tools import *
import DataLinkSet as DLSet
import joblib

class FTool:
    def __init__(self):
        dfU1 = LoadData(DLSet.feature_U_train_link)
        dfU2 = LoadData(DLSet.feature_U_judge_link)
        dfI1 = LoadData(DLSet.feature_I_train_link)
        dfI2 = LoadData(DLSet.feature_I_judge_link)
        dfU = pd.concat([dfU1, dfU2])
        self.dfU = dfU.drop_duplicates(['userID'], keep='first').reset_index(drop=True)
        dfI = pd.concat([dfI1, dfI2])
        self.dfI = dfI.drop_duplicates(['itemID'], keep='first').reset_index(drop=True)

        self.GBDT_clf = joblib.load(DLSet.hybrid_classifier_GBDT_link)
        self.enc = joblib.load(DLSet.hybrid_classifier_ENC_link)

    def get(self, userID, itemID):
        dfU = self.dfU[self.dfU['userID'] == userID]
        dfI = self.dfI[self.dfI['itemID'] == itemID]
        vu = dfU.values[0][1:]
        vi = dfI.values[0][1:]
        v = np.append(vu, vi)
        res = self.enc.transform(self.GBDT_clf.apply(v.reshape(1, -1))[:, :, 0]).A.reshape(1, -1)
        return res


if __name__ == '__main__':
    fTool = FTool()
    fTool.get(19837, 204)
    # print()