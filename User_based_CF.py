import numpy as np
import pandas as pd
from tqdm.contrib import itertools

from scipy import stats, spatial

class User_based_CF():
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
            for one_user, two_user in itertools.product(impute_zero_user_item_matrix_data.index, impute_zero_user_item_matrix_data.index):
                one_user_list.append(one_user)
                two_user_list.append(two_user)
                user_user_correlation.append(stats.pearsonr(impute_zero_user_item_matrix_data.loc[one_user, :], impute_zero_user_item_matrix_data.loc[two_user, :])[0])
        elif corr_methods == "cosine":
            for one_user, two_user in itertools.product(impute_zero_user_item_matrix_data.index, impute_zero_user_item_matrix_data.index):
                one_user_list.append(one_user)
                two_user_list.append(two_user)
                user_user_correlation.append(-1 * (1-spatial.distance.cosine(impute_zero_user_item_matrix_data.loc[one_user, :], impute_zero_user_item_matrix_data.loc[two_user, :])))

        self.user_user_correlation_data = pd.crosstab(index=np.array(one_user_list), \
                                                columns=np.array(two_user_list),\
                                                values=user_user_correlation,
                                                aggfunc=np.mean)
        return self.user_user_correlation_data

    def predict_without_time(self, user_id, item_id, user_column_name, item_column_name, num_user):
        # 1. 先找到相似的人
        similar_user_and_correlation = self.user_user_correlation_data[user_id].sort_values()[-num_user-1:-1]
        similar_user = dict()
        for one_index in list(similar_user_and_correlation.index):
            similar_user[one_index] = similar_user_and_correlation[one_index]

        # 2. 找到某個相似的人中針對某個item的rating與時間
        predict_user = self.traindata[list(map(lambda x: True if x in list(similar_user.keys()) else False, self.traindata[user_column_name]))][self.traindata[item_column_name] == item_id]

        # 3. 計算分母與分子
        rating_similar = list(map(lambda x: predict_user.iloc[x, 2] * similar_user[predict_user.iloc[x, 0]], [i for i in range(predict_user.shape[0])]))
        similar = list(map(lambda x: similar_user[predict_user.iloc[x, 0]], [i for i in range(predict_user.shape[0])]))

        # 4. 把兩者相除
        pred_rating = np.sum(rating_similar) / np.sum(similar)
        return pred_rating