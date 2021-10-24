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
class fm_model(nn.Module):
    def __init__(self, num_user_age, num_user_occupation, num_movie_genre, num_features):
        super(fm_model, self).__init__()
        self.user_age = nn.Linear(num_user_age, num_features)
        self.user_occupation = nn.Linear(num_user_occupation, num_features)
        self.movie_genre = nn.Linear(num_movie_genre, num_features)
        self.user_age_weight_linear = nn.Linear(num_user_age, 1)
        self.user_occupation_weight_linear = nn.Linear(num_user_occupation, 1)
        self.movie_genre_weight_linear = nn.Linear(num_movie_genre, 1)
        self.decoder = nn.Linear(33, 1)
        return

    def forward(self, user_age_feature, user_occupation_feature, movie_genre_feature):
        # Embedding Learning
        self.user_age_embedding = self.user_age(user_age_feature) # shape=(batch_size, num_features)
        self.user_occupation_embedding = self.user_occupation(user_occupation_feature) # shape=(batch_size, num_features)
        self.movie_genre_embedding = self.movie_genre(movie_genre_feature) # shape=(batch_size, num_features)
        self.user_age_weight = self.user_age_weight_linear(user_age_feature) # shape = (batch_size, 1)
        self.user_occupation_weight = self.user_occupation_weight_linear(user_occupation_feature) # shape = (batch_size, 1)
        self.movie_genre_weight = self.movie_genre_weight_linear(movie_genre_feature) # shape = (batch_size, 1)

        # Inner product
        self.user_age_user_occupation = self.user_age_embedding * self.user_occupation_embedding # shape=(batch_size, num_features)
        self.user_age_movie_genre = self.user_age_embedding * self.movie_genre_embedding # shape=(batch_size, num_features)
        self.user_occupation_movie_genre = self.user_occupation_embedding * self.movie_genre_embedding # shape=(batch_size, num_features)

        # Concatenate
        self.all = torch.cat((self.user_age_user_occupation, self.user_age_movie_genre, self.user_occupation_movie_genre,\
                              self.user_age_weight, self.user_occupation_weight, self.movie_genre_weight), dim=-1) 

        # Decoder
        X = self.decoder(self.all)
        return X