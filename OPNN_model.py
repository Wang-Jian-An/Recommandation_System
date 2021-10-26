import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import itertools

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 建構模型→Movielens
class opnn_model(nn.Module):
    def __init__(self, num_user_age, num_user_occupation, num_movie_genre, num_features, methods):
        super(opnn_model, self).__init__()
        num_decoder = 3*(num_features**2)+3
        self.methods = methods
        self.user_age = nn.Linear(num_user_age, num_features)
        self.user_occupation = nn.Linear(num_user_occupation, num_features)
        self.movie_genre = nn.Linear(num_movie_genre, num_features)
        self.user_age_weight_linear = nn.Linear(num_user_age, 1)
        self.user_occupation_weight_linear = nn.Linear(num_user_occupation, 1)
        self.movie_genre_weight_linear = nn.Linear(num_movie_genre, 1)
        self.decoder = nn.Sequential(
            nn.Linear(num_decoder, int(round(num_decoder/2, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_decoder/2, 0)), int(round(num_decoder/4, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_decoder/4, 0)), 1)
        )
        return

    def forward(self, user_age_feature, user_occupation_feature, movie_genre_feature):
        model_batch_size = user_age_feature.size()[0]

        # Embedding Learning
        self.user_age_embedding = self.user_age(user_age_feature) # shape=(batch_size, num_features)
        self.user_occupation_embedding = self.user_occupation(user_occupation_feature) # shape=(batch_size, num_features)
        self.movie_genre_embedding = self.movie_genre(movie_genre_feature) # shape=(batch_size, num_features)
        self.user_age_weight = self.user_age_weight_linear(user_age_feature) # shape = (batch_size, 1)
        self.user_occupation_weight = self.user_occupation_weight_linear(user_occupation_feature) # shape = (batch_size, 1)
        self.movie_genre_weight = self.movie_genre_weight_linear(movie_genre_feature) # shape = (batch_size, 1)

        # Outer product
        # 1. 增加某一維度 (batch_size, num_features, 1) * (batch_size, 1, num_features)
        # 2. 做Inner product (batch_size, num_features, num_features)
        # 3. Flatten
        self.user_age_user_occupation =\
            torch.tensordot(torch.unsqueeze(self.user_age_embedding, 1), self.user_occupation_embedding, dims=([1], [0])) # shape=(batch_size, num_features)
        self.user_age_user_occupation = torch.reshape(self.user_age_user_occupation, shape=(model_batch_size, -1))
       
        self.user_age_movie_genre =\
            torch.tensordot(torch.unsqueeze(self.user_age_embedding, 1), self.movie_genre_embedding, dims=([1], [0])) # shape=(batch_size, num_features)
        self.user_age_movie_genre = torch.reshape(self.user_age_movie_genre, shape=(model_batch_size, -1))
        
        self.user_occupation_movie_genre =\
            torch.tensordot(torch.unsqueeze(self.user_occupation_embedding, 1), self.movie_genre_embedding, dims=([1], [0])) # shape=(batch_size, num_features)
        self.user_occupation_movie_genre = torch.reshape(self.user_occupation_movie_genre, shape=(model_batch_size, -1))

        # Concatenate
        self.all = torch.cat((self.user_age_user_occupation, self.user_age_movie_genre, self.user_occupation_movie_genre,\
                              self.user_age_weight, self.user_occupation_weight, self.movie_genre_weight), dim=-1) 

        # Decoder
        X = self.decoder(self.all)

        # identify regression or classification task
        if self.methods == "regression":
            return X
        else:
            return nn.Sigmoid()(X)