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
class wide_deep_model(nn.Module):
    def __init__(self, num_user_age, num_user_occupation, num_movie_genre, num_features, methods):
        super(wide_deep_model, self).__init__()
        num_decoder = num_user_age + num_user_occupation + num_movie_genre + int(round(num_deep_decoder/4, 0))
        num_deep_decoder = 3 * num_features
        self.methods = methods
        self.user_age = nn.Linear(num_user_age, num_features)
        self.user_occupation = nn.Linear(num_user_occupation, num_features)
        self.movie_genre = nn.Linear(num_movie_genre, num_features)
        self.user_age_weight_linear = nn.Linear(num_user_age, 1)
        self.user_occupation_weight_linear = nn.Linear(num_user_occupation, 1)
        self.movie_genre_weight_linear = nn.Linear(num_movie_genre, 1)
        self.deep_decoder = nn.Sequential(
            nn.Linear(num_deep_decoder, int(round(num_deep_decoder/2, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_deep_decoder/2, 0)), int(round(num_deep_decoder/4, 0))),
            nn.Tanh(),
        )
        self.decoder = nn.Linear(num_decoder, 1)
        return

    def forward(self, user_age_feature, user_occupation_feature, movie_genre_feature):
        # Embedding Learning
        self.user_age_embedding = self.user_age(user_age_feature) # shape=(batch_size, num_features)
        self.user_occupation_embedding = self.user_occupation(user_occupation_feature) # shape=(batch_size, num_features)
        self.movie_genre_embedding = self.movie_genre(movie_genre_feature) # shape=(batch_size, num_features)

        ## Deep part
        # Inner product
        self.user_age_user_occupation = self.user_age_embedding * self.user_occupation_embedding # shape=(batch_size, num_features)
        self.user_age_movie_genre = self.user_age_embedding * self.movie_genre_embedding # shape=(batch_size, num_features)
        self.user_occupation_movie_genre = self.user_occupation_embedding * self.movie_genre_embedding # shape=(batch_size, num_features)

        # Concatenate
        self.deep_part_all = torch.cat((self.user_age_user_occupation, self.user_age_movie_genre, self.user_occupation_movie_genre), dim=-1) 

        # Deep_Decoder
        self.deep_part_all = self.deep_decoder(self.deep_part_all)

        ## Wide part
        # Concatenate
        self.wide_part_all = torch.cat([user_age_feature, user_occupation_feature, movie_genre_feature], dim=1)

        ## All
        # Concatenate
        self.all = torch.cat([self.wide_part_all, self.deep_part_all], dim=1)

        # Decoder
        X = self.decoder(self.all)

        if self.methods == "regression":
            return X
        else:
            return nn.Sigmoid()(X)

