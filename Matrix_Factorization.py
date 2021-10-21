import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import itertools

from User_based_CF import *
from Item_based_CF import *

from sklearn.metrics import mean_squared_error
import math

# Matrix Factorization Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        return

    def forward(self):
        return

class Loss_function(nn.Module):
    def __init__(self):
        super(Loss_function, self).__init__()
        return

    def forward(self):
        return

class Training():
    def __init__(self):
        return

    def fit():
        return
    
    def predict():
        return

    def evaluate():
        return





class Matrix_Factorization_with_bias():
    def __init__(self, num_user_id, num_item_id, num_features, all_true_user_item_matrix):
        self.p_matrix = np.random.random(size=(num_user_id, num_features)).astype("float")
        self.q_matrix = np.random.random(size=(num_features, num_item_id)).astype("float")

        # 計算global mean
        self.global_mean = all_true_user_item_matrix.values.reshape(-1, 1)[np.where(all_true_user_item_matrix.values.reshape(-1, 1) > 0)[0]].mean()

        # 計算bias of user and bias of item
        self.bu = list(map(lambda x: np.mean( all_true_user_item_matrix.loc[x, :].values[np.where(all_true_user_item_matrix.loc[x, :].values > 0)[0]] - self.global_mean), list(all_true_user_item_matrix.index)))
        self.bi = list(map(lambda x: np.mean( all_true_user_item_matrix.loc[:, x].values[np.where(all_true_user_item_matrix.loc[:, x].values > 0)[0]] - self.global_mean), list(all_true_user_item_matrix.columns)))

        self.bu, self.bi = np.array([i if i > 0 else 0 for i in self.bu]).reshape(-1, 1), np.array([i if i > 0 else 0 for i in self.bi]).reshape(1, -1)
        return

    # 預測Rating
    def MF_predict_with_bias(self):
        return np.dot(self.p_matrix, self.q_matrix) + self.bu + self.bi + self.global_mean

    # 2. 建立Loss function→不使用pytorch class
    def loss_func_with_bias(self, true_user_item_matrix, pred_user_item_matrix, _lambda_=1e-3):
        """
        true_user_item_matrix: 有遺失值的data.frame
        pred_user_item_matrix: 由模型計算出來的ndarray
        """
        # 2.1 先辨識出該值是否為有值
        identify_true_value_matrix = (true_user_item_matrix > 0).astype("int")

        # 2.4 把真實值遺失值的部分補零，並且把它轉成ndarray
        true_user_item_matrix = true_user_item_matrix.fillna(0).values  

        # 2.5 計算y_true-y_pred
        self.true_minus_pred = true_user_item_matrix-(pred_user_item_matrix*identify_true_value_matrix.values)

        # 2.4 計算loss
        loss = np.power(self.true_minus_pred, 2).sum()/2 + \
            _lambda_ * (np.power(self.p_matrix, 2).sum() + np.power(self.q_matrix, 2).sum() + np.power(self.bu, 2).sum() + np.power(self.bi, 2).sum() + np.power(self.global_mean, 2).sum())/2
        return loss

    # 3. 計算gradients
    def compute_gradients_with_bias(self, _lambda_=1e-3):
        self.gradient_p_matrix = np.dot(self.true_minus_pred, -1 * self.q_matrix.T).sum() + _lambda_ * math.sqrt(np.power(self.p_matrix, 2).sum())
        self.gradient_q_matrix = np.dot(self.true_minus_pred.T, -1 * self.p_matrix).sum() + _lambda_ * math.sqrt(np.power(self.q_matrix, 2).sum())
        self.gradient_bu_matrix = (self.true_minus_pred * -1).sum() + _lambda_ * math.sqrt( np.power(self.bu, 2).sum() )
        self.gradient_bi_matrix = (self.true_minus_pred * -1).sum() + _lambda_ * math.sqrt( np.power(self.bi, 2).sum() )
        self.gradient_global_mean_matrix = self.true_minus_pred * -1 + _lambda_ * math.sqrt( np.power(self.global_mean, 2).sum() )
        return 

    # 4. Weight Updated
    def weight_updated_with_bias(self, learning_rate):
        self.p_matrix -= learning_rate * self.gradient_p_matrix
        self.q_matrix -= learning_rate * self.gradient_q_matrix
        self.bu -= learning_rate * self.gradient_bu_matrix
        self.bi -= learning_rate * self.gradient_bi_matrix
        self.global_mean -= learning_rate * self.gradient_global_mean_matrix
        return 

class Matrix_Factorization_with_nobias():
    def __init__(self, num_user_id, num_item_id, num_features):
        self.p_matrix = np.random.random(size=(num_user_id, num_features)).astype("float")
        self.q_matrix = np.random.random(size=(num_features, num_item_id)).astype("float")
        return

    # 預測Rating
    def MF_predict_with_nobias(self):
        return np.dot(self.p_matrix, self.q_matrix)

    # 2. 建立Loss function→不使用pytorch class
    def loss_func_with_nobias(self, true_user_item_matrix, pred_user_item_matrix, _lambda_=1e-3):
        """
        true_user_item_matrix: 有遺失值的data.frame
        pred_user_item_matrix: 由模型計算出來的ndarray
        """
        # 2.1 先辨識出該值是否為有值
        identify_true_value_matrix = (true_user_item_matrix > 0).astype("int")

        # 2.4 把真實值遺失值的部分補零，並且把它轉成ndarray
        true_user_item_matrix = true_user_item_matrix.fillna(0).values  

        # 2.5 計算y_true-y_pred
        self.true_minus_pred = true_user_item_matrix-(pred_user_item_matrix*identify_true_value_matrix.values)

        # 2.4 計算loss
        loss = np.power(self.true_minus_pred, 2).sum()/2 + \
            _lambda_ * (np.power(self.p_matrix, 2).sum() + np.power(self.q_matrix, 2).sum() )/2
        return loss

    # 3. 計算gradients
    def compute_gradients_with_nobias(self, _lambda_=1e-3):
        self.gradient_p_matrix = np.dot(self.true_minus_pred, -1 * self.q_matrix.T).sum() + _lambda_ * math.sqrt(np.power(self.p_matrix, 2).sum())
        self.gradient_q_matrix = np.dot(self.true_minus_pred.T, -1 * self.p_matrix).sum() + _lambda_ * math.sqrt(np.power(self.q_matrix, 2).sum())
        return 

    # 4. Weight Updated
    def weight_updated_with_nobias(self, learning_rate):
        self.p_matrix -= learning_rate * self.gradient_p_matrix
        self.q_matrix -= learning_rate * self.gradient_q_matrix
        return 