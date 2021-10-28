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

# CCPM
class ccpm_model(nn.Module):
    def __init__(self, num_user_age, num_user_occupation, num_movie_genre, num_features, methods):
        super(ccpm_model, self).__init__()
        num_decoder = 3 * num_features
        self.methods = methods
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
        self.all = torch.cat((torch.unsqueeze(self.user_age_embedding, 1), 
                              torch.unsqueeze(self.user_occupation_embedding, 1), 
                              torch.unsqueeze(self.movie_genre_embedding, 1)), dim=1) # shape = (batch_size, features, embedding)

        # Convulution
        self.all = self.conv_seq(self.all)[:, :, 0]

        # Decoder
        X = self.decoder(self.all)
        
        # identify regression or classification task
        if self.methods == "regression":
            return X
        else:
            return nn.Sigmoid()(X)

# FM model
class fm_model(nn.Module):
    def __init__(self, num_user_age, num_user_occupation, num_movie_genre, num_features, methods):
        super(fm_model, self).__init__()
        self.user_age = nn.Linear(num_user_age, num_features)
        self.user_occupation = nn.Linear(num_user_occupation, num_features)
        self.movie_genre = nn.Linear(num_movie_genre, num_features)
        self.user_age_weight_linear = nn.Linear(num_user_age, 1)
        self.user_occupation_weight_linear = nn.Linear(num_user_occupation, 1)
        self.movie_genre_weight_linear = nn.Linear(num_movie_genre, 1)
        self.decoder = nn.Linear(num_features*3+3, 1)
        self.methods = methods
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

        # identify regression or classification task
        if self.methods == "regression":
            return X
        else:
            return nn.Sigmoid()(X)

# FNN model
class fnn_model(nn.Module):
    def __init__(self, num_user_age, num_user_occupation, num_movie_genre, num_features, methods):
        super(fnn_model, self).__init__()
        num_decoder = num_features*3+3
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
        # Embedding Learning
        self.user_age_embedding = self.user_age(user_age_feature) # shape=(batch_size, num_features)
        self.user_occupation_embedding = self.user_occupation(user_occupation_feature) # shape=(batch_size, num_features)
        self.movie_genre_embedding = self.movie_genre(movie_genre_feature) # shape=(batch_size, num_features)
        self.user_age_weight = self.user_age_weight_linear(user_age_feature) # shape = (batch_size, 1)
        self.user_occupation_weight = self.user_occupation_weight_linear(user_occupation_feature) # shape = (batch_size, 1)
        self.movie_genre_weight = self.movie_genre_weight_linear(movie_genre_feature) # shape = (batch_size, 1)

        # Concatenate
        self.all = torch.cat((self.user_age_embedding, self.user_occupation_embedding, self.movie_genre_embedding,\
                              self.user_age_weight, self.user_occupation_weight, self.movie_genre_weight), dim=-1) 

        # Decoder
        X = self.decoder(self.all)

        if self.methods == "regression":
            return X
        else:
            return nn.Sigmoid()(X)

# Generalize Matrix Factorization in NeuCF
class gmf_neucf_model(nn.Module):
    def __init__(self, num_user, num_item, num_features):
        super(gmf_neucf_model, self).__init__()
        self.user_embedding_learning = nn.Linear(num_user, num_features)
        self.item_embedding_learning = nn.Linear(num_item, num_features)
        self.decoder = nn.Linear(num_features, 1)
        return

    def forward(self, user, item):
        # Embedding Learning
        self.user_embedding = self.user_embedding_learning(user) # shape = (batch_size, num_features)
        self.item_embedding = self.item_embedding_learning(item) # shape = (batch_size, num_features)
        
        # Inner product
        self.user_item_inner_product = self.user_embedding * self.item_embedding
        
        # print(self.user_item_inner_product)

        # Decoder
        yhat = self.decoder(self.user_item_inner_product)
        return yhat


# IPNN model
class ipnn_model(nn.Module):
    def __init__(self, num_user_age, num_user_occupation, num_movie_genre, num_features, methods):
        super(ipnn_model, self).__init__()
        num_decoder = num_features*3+3
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

        if self.methods == "regression":
            return X
        else:
            return nn.Sigmoid()(X)

# MLP in NeuCF
class mlp_neucf_model(nn.Module):
    def __init__(self, num_user, num_item, num_features):
        super(mlp_neucf_model, self).__init__()
        self.user_embedding_learning = nn.Linear(num_user, num_features)
        self.item_embedding_learning = nn.Linear(num_item, num_features)
        self.decoder = nn.Sequential(
            nn.Linear(2 * num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, int(round(num_features/2, 0))),
            nn.ReLU(),
            nn.Linear(int(round(num_features/2, 0)), 1)
        )
        return

    def forward(self, user, item):
        # Embedding Learning
        self.user_embedding = self.user_embedding_learning(user) # shape = (batch_size, num_features)
        self.item_embedding = self.item_embedding_learning(item) # shape = (batch_size, num_features)
        
        # Concatenate
        self.user_item = torch.cat([self.user_embedding, self.item_embedding], dim=1)

        # Decoder
        yhat = self.decoder(self.user_item)
        return yhat

# NeuMF
class neumf(nn.Module):
    def __init__(self, num_user, num_item, num_features, methods):
        super(neumf, self).__init__()
        self.methods = methods
        self.user_embedding_learning = nn.Linear(num_user, num_features)
        self.item_embedding_learning = nn.Linear(num_item, num_features)
        self.decoder = nn.Sequential(
            nn.Linear(3 * num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, int(round(num_features/2, 0))),
            nn.ReLU(),
            nn.Linear(int(round(num_features/2, 0)), 1)
        )
        return

    def forward(self, user, item):
        # Embedding Learning
        self.user_embedding = self.user_embedding_learning(user) # shape = (batch_size, num_features)
        self.item_embedding = self.item_embedding_learning(item) # shape = (batch_size, num_features)
        
        # Inner product
        self.user_item_inner_product = self.user_embedding * self.item_embedding

        # Concatenate
        self.user_item = torch.cat([self.user_embedding, self.item_embedding], dim=1)

        # Inner product and Concatenate
        self.all = torch.cat([self.user_item_inner_product, self.user_item], dim=1)

        # Decoder
        X = self.decoder(self.all)

        # identify regression or classification task
        if self.methods == "regression":
            return X
        else:
            return nn.Sigmoid()(X)

# OPNN model
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

# PIN model
class pin_model(nn.Module):
    def __init__(self, num_user_age, num_user_occupation, num_movie_genre, num_features, methods):
        super(pin_model, self).__init__()
        num_decoder = 3 * num_features
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
        self.all = F.tanh(torch.cat((self.user_age_user_occupation, self.user_age_movie_genre, self.user_occupation_movie_genre), dim=-1) )

        # Decoder
        X = self.decoder(self.all)

        # identify regression or classification task
        if self.methods == "regression":
            return X
        else:
            return nn.Sigmoid()(X)
