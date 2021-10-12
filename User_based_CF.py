import numpy as np
import pandas as pd
from tqdm.contrib import itertools

from scipy import stats

def compute_correlation(user_item_matrix_data, corr_methods):
    # Impute zero to NaN
    impute_zero_user_item_matrix_data = user_item_matrix_data.fillna(0)

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
            user_user_correlation.append(stats.pearsonr(impute_zero_user_item_matrix_data.loc[one_user, :], impute_zero_user_item_matrix_data.loc[two_user, :])[0])

    user_user_correlation_data = pd.crosstab(index=np.array(one_user_list), \
                                            columns=np.array(two_user_list),\
                                            values=user_user_correlation)
    return user_user_correlation_data

def predict():
    return