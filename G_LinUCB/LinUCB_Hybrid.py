import numpy as np


class HybridLinUCB:
    def __init__(self):
        # 超参数
        self.alpha = 2.1    # 1 + np.sqrt(np.log(2/delta)/2)
        self.r1 = 40
        self.r0 = -5

        # 常量设定
        self.d = 6                          # dimension of user features = d
        self.k = self.d * self.d            # dimension of common features = k

        # 各个膀臂信息
        self.article_features = {}

        self.A0 = np.identity(self.k)       # A0 : matrix to compute hybrid part, k*k
        self.b0 = np.zeros((self.k, 1))     # b0 : vector to compute hybrid part, k

        self.Aa = {}                        # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.Ba = {}                        # Ba : collection of matrix to compute hybrid part, d*k
        self.ba = {}                        # ba : collection of vectors to compute disjoint part, d*1

        # 便于计算
        self.A0I = np.identity(self.k)
        self.AaI = {}
        self.BaT = {}
        self.AaIba = {}
        self.AaIBa = {}
        self.A0IBaTAaI = {}
        self.theta = {}
        self.beta = np.zeros((self.k, 1))

        # 记忆体
        self.index_all = {}
        self.index_all_inv = {}
        self.a_max = None
        self.z = None
        self.zT = None
        self.xaT = None
        self.xa = None

        self.total = 0                      # 总arm数

    # 初始化所有的arm
    def set_articles(self, articles):
        art_len = len(articles)

        # 初始化
        self.article_features = np.zeros((art_len, 1, self.d))
        self.Aa = np.zeros((art_len, self.d, self.d))
        self.AaI = np.zeros((art_len, self.d, self.d))
        self.Ba = np.zeros((art_len, self.d, self.k))
        self.BaT = np.zeros((art_len, self.k, self.d))
        self.ba = np.zeros((art_len, self.d, 1))
        self.AaIba = np.zeros((art_len, self.d, 1))
        self.AaIBa = np.zeros((art_len, self.d, self.k))
        self.A0IBaTAaI = np.zeros((art_len, self.k, self.d))
        self.theta = np.zeros((art_len, self.d, 1))

        # 依次赋值
        i = 0
        for key in articles:
            self.index_all[key] = i
            self.index_all_inv[i] = key
            self.article_features[i] = articles[key][:self.d]
            self.Aa[i] = np.identity(self.d)
            self.AaI[i] = np.identity(self.d)
            self.Ba[i] = np.zeros((self.d, self.k))
            self.BaT[i] = np.zeros((self.k, self.d))
            self.ba[i] = np.zeros((self.d, 1))
            self.AaIba[i] = np.zeros((self.d, 1))
            self.AaIBa[i] = np.zeros((self.d, self.k))
            self.A0IBaTAaI[i] = np.zeros((self.k, self.d))
            self.theta[i] = np.zeros((self.d, 1))
            i += 1

        self.total = i
        print('total arms =', self.total)

    # 更新奖励
    def update(self, rewardSet, K):
        for i in range(K):
            if rewardSet[i] > 0:
                r = self.r1 * rewardSet[i]
            else:
                r = self.r0

            self.A0 += self.BaT[self.a_max[i]].dot(self.AaIBa[self.a_max[i]])
            self.b0 += self.BaT[self.a_max[i]].dot(self.AaIba[self.a_max[i]])
            self.Aa[self.a_max[i]] += np.dot(self.xa, self.xaT)
            self.AaI[self.a_max[i]] = np.linalg.inv(self.Aa[self.a_max[i]])
            self.Ba[self.a_max[i]] += np.dot(self.xa, self.zT[i])
            self.BaT[self.a_max[i]] = np.transpose(self.Ba[self.a_max[i]])
            self.ba[self.a_max[i]] += r * self.xa
            self.AaIba[self.a_max[i]] = np.dot(self.AaI[self.a_max[i]], self.ba[self.a_max[i]])
            self.AaIBa[self.a_max[i]] = np.dot(self.AaI[self.a_max[i]], self.Ba[self.a_max[i]])

            self.A0 += np.dot(self.z[i], self.zT[i]) - np.dot(self.BaT[self.a_max[i]], self.AaIBa[self.a_max[i]])
            self.b0 += r * self.z[i] - np.dot(self.BaT[self.a_max[i]], self.AaIba[self.a_max[i]])
            self.A0I = np.linalg.inv(self.A0)
            self.A0IBaTAaI[self.a_max[i]] = self.A0I.dot(self.BaT[self.a_max[i]]).dot(self.AaI[self.a_max[i]])

            self.beta = np.dot(self.A0I, self.b0)
            self.theta = self.AaIba - np.dot(self.AaIBa, self.beta)


    # 进行推荐
    def recommend(self, user_features, K):
        self.xa = np.array(user_features[:self.d]).reshape((self.d, 1))             # (6,1)
        self.xaT = np.transpose(self.xa)                                            # (1,6)

        index = [i for i in range(self.total)]
        article_features_tmp = self.article_features[index]

        # za : feature of current user/article combination, k*1
        za = np.outer(article_features_tmp.reshape(-1), self.xa).reshape((self.total, self.k, 1))    # (20,36,1)
        zaT = np.transpose(za, (0, 2, 1))                               # (20,1,36)
        A0Iza = np.matmul(self.A0I, za)                                 # (20,36,1)
        A0IBaTAaIxa = np.matmul(self.A0IBaTAaI[index], self.xa)         # (20,36,1)
        AaIxa = self.AaI[index].dot(self.xa)                            # (20,6,1)
        AaIBaA0IBaTAaIxa = np.matmul(self.AaIBa[index], A0IBaTAaIxa)    # (20,6,1)

        s = np.matmul(zaT, A0Iza - 2*A0IBaTAaIxa) + np.matmul(self.xaT, AaIxa + AaIBaA0IBaTAaIxa)   # (20,1,1)
        p = zaT.dot(self.beta) + np.matmul(self.xaT, self.theta[index]) + self.alpha*np.sqrt(s)     # (20,1,1)

        # 初始化记忆体
        self.a_max = []
        self.z = []
        self.zT = []

        # 初始化答案
        res = []

        # 选出最大K个
        max_tmp = np.argmax(p)

        temp = p.tolist()
        temp = [each[0][0] for each in temp]
        max_indexSet = np.argpartition(temp, -K)[-K:]
        # if max_tmp not in max_indexSet:
        #    print(max_tmp, max_indexSet)
        print(max_indexSet)
        # 存储记忆
        for max_index in max_indexSet:
            self.z.append(za[max_index])
            self.zT.append(zaT[max_index])
            self.a_max.append(max_index)
            res.append(self.index_all_inv[max_index])
        return res
