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
class ipnn_model(nn.Module):
    def __init__(self, num_user_age, num_user_occupation, num_movie_genre, num_decoder, num_features):
        super(ipnn_model, self).__init__()
        self.user_age = nn.Linear(num_user_age, num_features)
        self.user_occupation = nn.Linear(num_user_occupation, num_features)
        self.movie_genre = nn.Linear(num_movie_genre, num_features)
        self.conv_seq = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=num_decoder, kernel_size=3),
            nn.MaxPool1d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_decoder, int(round(num_decoder/2, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_decoder/2, 0)), int(round(num_decoder/4, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_decoder/4, 0)), 1)
        )
        return

    def forward(self, user_age_feature, user_occupation_feature, movie_genre_feature):
        # Embedding Learning
        self.user_age_embedding = self.user_age(user_age_feature) # shape=(batch_size, embedding)
        self.user_occupation_embedding = self.user_occupation(user_occupation_feature) # shape=(batch_size, embedding)
        self.movie_genre_embedding = self.movie_genre(movie_genre_feature) # shape=(batch_size, embedding)

        # Concatenate
        self.all = torch.cat((torch.unsqueeze(self.user_age_user_occupation, 1), 
                              torch.unsqueeze(self.user_age_movie_genre, 1), 
                              torch.unsqueeze(self.user_occupation_movie_genre, 1)), dim=1) # shape = (batch_size, features, embedding)

        # Convulution
        self.all = self.conv_seq(self.all)[:, :, 0]

        # Decoder
        X = self.decoder(self.all)
        return X