import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from tqdm import tqdm
import itertools

import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error
import math

class matrix_factorization():
    def __init__(self, true_user_item_matrix, num_features):
        """
        true_user_item_matrix：user與item的matrix，沒有補過值
        """
        self.user_id_list = list(true_user_item_matrix.index)
        self.item_id_list = list(true_user_item_matrix.columns)

        # 辨識該值是否真的有值
        self.identify_value_exist = torch.from_numpy( true_user_item_matrix.isna().astype("float").values )
        self.true_user_item_matrix = self.preprocessing_user_item_matrix(true_user_item_matrix)

        self.p_matrix = torch.randn(size=(len(self.user_id_list), num_features), requires_grad=True)
        self.q_matrix = torch.randn(size=(num_features, len(self.item_id_list)), requires_grad=True)

        # 計算global mean
        self.global_mean = torch.mean( self.true_user_item_matrix.flatten()[self.true_user_item_matrix.flatten().nonzero()] )

        # 計算bias of user and bias of item
        self.bu = torch.Tensor(list(map(lambda x: torch.mean(x[x.nonzero()]), self.true_user_item_matrix ))).reshape(shape=(-1, 1)) - self.global_mean
        self.bi = torch.Tensor(list(map(lambda x: torch.mean(x[x.nonzero()]), torch.transpose(self.true_user_item_matrix, 0, 1) ))).reshape(shape=(1, -1)) - self.global_mean
        return

    def preprocessing_user_item_matrix(self, true_user_item_matrix):
        # 把NaN全部補零
        fill_user_item_matrix_data = true_user_item_matrix.fillna(0)
        return torch.from_numpy(fill_user_item_matrix_data.values)
    
    def fit(self, epochs, learning_rate, regularization_rate, bias_or_not):
        # 建立空的儲存以存取Loss
        self.train_loss = list()

        # 定義loss function
        loss_func = nn.MSELoss()

        # 定義optimizer
        optimizer = torch.optim.SGD([self.p_matrix, self.q_matrix], lr=learning_rate, weight_decay=regularization_rate)

        for epoch in range(epochs):
            if bias_or_not == False:
                yhat = torch.tensordot(self.p_matrix, self.q_matrix, dims=([1], [0]))
            else:
                yhat = torch.tensordot(self.p_matrix, self.q_matrix, dims=([1], [0]))+self.bu+self.bi+self.global_mean
                
            yhat = yhat * self.identify_value_exist

            loss = loss_func(yhat, self.true_user_item_matrix)
            self.train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"=== Epoch: {epoch} Train Loss: {loss.item()}")
        return

    def predict(self, user_id, item_id):
        yhat = torch.tensordot(self.p_matrix, self.q_matrix, dims=([1], [0]))
        yhat_dataframe = pd.DataFrame(yhat.detach().numpy(), index=self.user_id_list, columns=self.item_id_list)
        return yhat_dataframe.loc[user_id, item_id]

    def evaluate(self, testdata):
        """
        testdata：data.frame，<user_id, item_id, rating, (timestamp)>
        """
        testdata["yhat"] = list(map(lambda user, item: self.predict(user, item), tqdm(testdata.iloc[:, 0]), testdata.iloc[:, 1]))
        print(f"MSE: {mean_squared_error(y_true=testdata.iloc[:, 2], y_pred=testdata['yhat'])}\nr2_score: {r2_score(y_true=testdata.iloc[:, 2], y_pred=testdata['yhat'])}")
        return

    def save_model(self):
        return