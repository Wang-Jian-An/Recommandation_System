import numpy as np
import pandas as pd
from tqdm.contrib import itertools

from scipy import stats, spatial

class Item_based_CF():
    def __init__(self, traindata, user_item_matrix_data):
        self.traindata = traindata
        self.user_item_matrix_data = user_item_matrix_data
        return

    def compute_correlation(self, corr_methods):
        # Impute zero to NaN
        impute_zero_user_item_matrix_data = self.user_item_matrix_data.fillna(0)

        # Compute the correlation between two user
        one_user_list = list()
        two_user_list = list()
        user_user_correlation = list()

        if corr_methods == "pearson":
            for one_item, two_item in itertools.product(impute_zero_user_item_matrix_data.columns, impute_zero_user_item_matrix_data.columns):
                one_user_list.append(one_item)
                two_user_list.append(two_item)
                user_user_correlation.append(stats.pearsonr(impute_zero_user_item_matrix_data.loc[:, one_item], impute_zero_user_item_matrix_data.loc[:, two_item])[0])
        elif corr_methods == "cosine":
            for one_item, two_item in itertools.product(impute_zero_user_item_matrix_data.columns, impute_zero_user_item_matrix_data.columns):
                one_user_list.append(one_item)
                two_user_list.append(two_item)
                user_user_correlation.append(-1 * (1-spatial.distance.cosine(impute_zero_user_item_matrix_data.loc[:, one_item], impute_zero_user_item_matrix_data.loc[:, two_item])))

        self.item_item_correlation_data = pd.crosstab(index=np.array(one_user_list), \
                                                columns=np.array(two_user_list),\
                                                values=user_user_correlation,
                                                aggfunc=np.mean)
        return self.item_item_correlation_data

    def predict_without_time(self, item_id, user_id, num_item):
        # 1. 先找到相似的物品
        similar_item_and_correlation = self.item_item_correlation_data[item_id].sort_values()[-num_item:]
        similar_item = dict()
        for one_index in list(similar_item_and_correlation.index):
            similar_item[one_index] = similar_item_and_correlation[one_index]

        # 2. 找到某個相似的物品中針對某個item的rating與時間
        predict_user = self.traindata[list(map(lambda x: True if x in list(similar_item.keys()) else False, self.traindata["Item_id"]))][self.traindata["User_id"] == user_id]

        # 3. 計算分母與分子
        rating_similar = list(map(lambda x: predict_user.iloc[x, 2] * similar_item[predict_user.iloc[x, 1]], [i for i in range(predict_user.shape[0])]))
        similar = list(map(lambda x: similar_item[predict_user.iloc[x, 1]], [i for i in range(predict_user.shape[0])]))

        # 4. 把兩者相除
        pred_rating = np.sum(rating_similar) / np.sum(similar)
        return pred_rating