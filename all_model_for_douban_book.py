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
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# CCPM
class ccpm_model(nn.Module):
    def __init__(self, num_user_group, num_book_author, num_book_publisher, num_book_year, num_features, methods):
        super(ccpm_model, self).__init__()
        num_decoder = 4 * num_features
        self.methods = methods
        self.user_group = nn.Linear(num_user_group, num_features)
        self.book_author = nn.Linear(num_book_author, num_features)
        self.book_publisher = nn.Linear(num_book_publisher, num_features)
        self.book_year = nn.Linear(num_book_year, num_features)
        self.conv_seq = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3),
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

    def forward(self, user_group_feature, book_author_feature, book_publisher_feature, book_year_feature):
        # Embedding Learning
        self.user_group_embedding = self.user_group(user_group_feature) # shape=(batch_size, embedding)
        self.book_author_embedding = self.book_author(book_author_feature) # shape=(batch_size, embedding)
        self.book_publisher_embedding = self.book_publisher(book_publisher_feature) # shape=(batch_size, embedding)
        self.book_year_embedding = self.book_year(book_year_feature) # shape=(batch_size, embedding)

        # Concatenate
        self.all = torch.cat((torch.unsqueeze(self.user_group_embedding, 1), 
                              torch.unsqueeze(self.book_author_embedding, 1), 
                              torch.unsqueeze(self.book_publisher_embedding, 1),
                              torch.unsqueeze(self.book_year_embedding, 1)), dim=1) # shape = (batch_size, features, embedding)

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
    def __init__(self, num_user_group, num_book_author, num_book_publisher, num_book_year, num_features, methods):
        super(fm_model, self).__init__()
        self.user_group = nn.Linear(num_user_group, num_features)
        self.book_author = nn.Linear(num_book_author, num_features)
        self.book_publisher = nn.Linear(num_book_publisher, num_features)
        self.book_year = nn.Linear(num_book_year, num_features)
        self.user_group_weight_linear = nn.Linear(num_user_group, 1)
        self.book_author_weight_linear = nn.Linear(num_book_author, 1)
        self.book_publisher_weight_linear = nn.Linear(num_book_publisher, 1)
        self.book_year_weight_linear = nn.Linear(num_book_year, 1)
        self.decoder = nn.Linear(num_features*3+3, 1)
        self.methods = methods
        return

    def forward(self, user_group_feature, book_author_feature, book_publisher_feature, book_year_feature):
        # Embedding Learning
        self.user_group_embedding = self.user_group(user_group_feature) # shape=(batch_size, embedding)
        self.book_author_embedding = self.book_author(book_author_feature) # shape=(batch_size, embedding)
        self.book_publisher_embedding = self.book_publisher(book_publisher_feature) # shape=(batch_size, embedding)
        self.book_year_embedding = self.book_year(book_year_feature) # shape=(batch_size, embedding)
        self.user_group_weight = self.user_group_weight_linear(user_group_feature) # shape = (batch_size, 1)
        self.book_author_weight = self.book_author_weight_linear(book_author_feature) # shape = (batch_size, 1)
        self.book_publisher_weight = self.book_publisher_weight_linear(book_publisher_feature) # shape = (batch_size, 1)
        self.book_year_weight = self.book_year_weight_linear(book_year_feature) # shape = (batch_size, 1)

        # Inner product
        self.user_group_book_author = self.user_group * self.book_author
        self.user_group_book_publisher = self.user_group * self.book_publisher
        self.user_group_book_year = self.user_group * self.book_year
        self.book_author_book_publisher = self.book_author * self.book_publisher
        self.book_author_book_year = self.book_author * self.book_year
        self.book_publisher_book_year = self.book_publisher * self.book_year

        # Concatenate
        self.all = torch.cat((self.user_group_book_author, self.user_group_book_publisher, self.user_group_book_year,\
                              self.book_author_book_publisher, self.book_author_book_year, self.book_publisher_book_year), dim=-1) 

        # Decoder
        X = self.decoder(self.all)

        # identify regression or classification task
        if self.methods == "regression":
            return X
        else:
            return nn.Sigmoid()(X)

# FNN model
class fnn_model(nn.Module):
    def __init__(self, num_user_group, num_book_author, num_book_publisher, num_book_year, num_features, methods):
        super(fnn_model, self).__init__()
        num_decoder = num_features*3+3
        self.methods = methods
        self.user_group = nn.Linear(num_user_group, num_features)
        self.book_author = nn.Linear(num_book_author, num_features)
        self.book_publisher = nn.Linear(num_book_publisher, num_features)
        self.book_year = nn.Linear(num_book_year, num_features)
        self.user_group_weight_linear = nn.Linear(num_user_group, 1)
        self.book_author_weight_linear = nn.Linear(num_book_author, 1)
        self.book_publisher_weight_linear = nn.Linear(num_book_publisher, 1)
        self.book_year_weight_linear = nn.Linear(num_book_year, 1)
        self.decoder = nn.Sequential(
            nn.Linear(num_decoder, int(round(num_decoder/2, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_decoder/2, 0)), int(round(num_decoder/4, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_decoder/4, 0)), 1)
        )
        return

    def forward(self, user_group_feature, book_author_feature, book_publisher_feature, book_year_feature):
        # Embedding Learning
        self.user_group_embedding = self.user_group(user_group_feature) # shape=(batch_size, embedding)
        self.book_author_embedding = self.book_author(book_author_feature) # shape=(batch_size, embedding)
        self.book_publisher_embedding = self.book_publisher(book_publisher_feature) # shape=(batch_size, embedding)
        self.book_year_embedding = self.book_year(book_year_feature) # shape=(batch_size, embedding)
        self.user_group_weight = self.user_group_weight_linear(user_group_feature) # shape = (batch_size, 1)
        self.book_author_weight = self.book_author_weight_linear(book_author_feature) # shape = (batch_size, 1)
        self.book_publisher_weight = self.book_publisher_weight_linear(book_publisher_feature) # shape = (batch_size, 1)
        self.book_year_weight = self.book_year_weight_linear(book_year_feature) # shape = (batch_size, 1)

        # Concatenate
        self.all = torch.cat((self.user_group_book_author, self.user_group_book_publisher, self.user_group_book_year,\
                              self.book_author_book_publisher, self.book_author_book_year, self.book_publisher_book_year), dim=-1) 

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
    def __init__(self, num_user_group, num_book_author, num_book_publisher, num_book_year, num_features, methods):
        super(ipnn_model, self).__init__()
        num_decoder = num_features*3+3
        self.methods = methods
        self.user_group = nn.Linear(num_user_group, num_features)
        self.book_author = nn.Linear(num_book_author, num_features)
        self.book_publisher = nn.Linear(num_book_publisher, num_features)
        self.book_year = nn.Linear(num_book_year, num_features)
        self.user_group_weight_linear = nn.Linear(num_user_group, 1)
        self.book_author_weight_linear = nn.Linear(num_book_author, 1)
        self.book_publisher_weight_linear = nn.Linear(num_book_publisher, 1)
        self.book_year_weight_linear = nn.Linear(num_book_year, 1)
        self.decoder = nn.Sequential(
            nn.Linear(num_decoder, int(round(num_decoder/2, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_decoder/2, 0)), int(round(num_decoder/4, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_decoder/4, 0)), 1)
        )
        return

    def forward(self, user_group_feature, book_author_feature, book_publisher_feature, book_year_feature):
        # Embedding Learning
        self.user_group_embedding = self.user_group(user_group_feature) # shape=(batch_size, embedding)
        self.book_author_embedding = self.book_author(book_author_feature) # shape=(batch_size, embedding)
        self.book_publisher_embedding = self.book_publisher(book_publisher_feature) # shape=(batch_size, embedding)
        self.book_year_embedding = self.book_year(book_year_feature) # shape=(batch_size, embedding)
        self.user_group_weight = self.user_group_weight_linear(user_group_feature) # shape = (batch_size, 1)
        self.book_author_weight = self.book_author_weight_linear(book_author_feature) # shape = (batch_size, 1)
        self.book_publisher_weight = self.book_publisher_weight_linear(book_publisher_feature) # shape = (batch_size, 1)
        self.book_year_weight = self.book_year_weight_linear(book_year_feature) # shape = (batch_size, 1)

        # Inner product
        self.user_group_book_author = self.user_group * self.book_author
        self.user_group_book_publisher = self.user_group * self.book_publisher
        self.user_group_book_year = self.user_group * self.book_year
        self.book_author_book_publisher = self.book_author * self.book_publisher
        self.book_author_book_year = self.book_author * self.book_year
        self.book_publisher_book_year = self.book_publisher * self.book_year

        # Concatenate
        self.all = torch.cat((self.user_group_book_author, self.user_group_book_publisher, self.user_group_book_year,\
                              self.book_author_book_publisher, self.book_author_book_year, self.book_publisher_book_year), dim=-1) 

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
    def __init__(self, num_user_group, num_book_author, num_book_publisher, num_book_year, num_features, methods):
        super(opnn_model, self).__init__()
        num_decoder = 6*(num_features**2)+4
        self.methods = methods
        self.user_group = nn.Linear(num_user_group, num_features)
        self.book_author = nn.Linear(num_book_author, num_features)
        self.book_publisher = nn.Linear(num_book_publisher, num_features)
        self.book_year = nn.Linear(num_book_year, num_features)
        self.user_group_weight_linear = nn.Linear(num_user_group, 1)
        self.book_author_weight_linear = nn.Linear(num_book_author, 1)
        self.book_publisher_weight_linear = nn.Linear(num_book_publisher, 1)
        self.book_year_weight_linear = nn.Linear(num_book_year, 1)
        self.decoder = nn.Sequential(
            nn.Linear(num_decoder, int(round(num_decoder/2, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_decoder/2, 0)), int(round(num_decoder/4, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_decoder/4, 0)), 1)
        )
        return

    def forward(self, user_group_feature, book_author_feature, book_publisher_feature, book_year_feature):
        model_batch_size = user_group_feature.size()[0]

        # Embedding Learning
        self.user_group_embedding = self.user_group(user_group_feature) # shape=(batch_size, embedding)
        self.book_author_embedding = self.book_author(book_author_feature) # shape=(batch_size, embedding)
        self.book_publisher_embedding = self.book_publisher(book_publisher_feature) # shape=(batch_size, embedding)
        self.book_year_embedding = self.book_year(book_year_feature) # shape=(batch_size, embedding)
        self.user_group_weight = self.user_group_weight_linear(user_group_feature) # shape = (batch_size, 1)
        self.book_author_weight = self.book_author_weight_linear(book_author_feature) # shape = (batch_size, 1)
        self.book_publisher_weight = self.book_publisher_weight_linear(book_publisher_feature) # shape = (batch_size, 1)
        self.book_year_weight = self.book_year_weight_linear(book_year_feature) # shape = (batch_size, 1)

        # Outer product
        # 1. 增加某一維度 (batch_size, num_features, 1) * (batch_size, 1, num_features)
        # 2. 做Inner product (batch_size, num_features, num_features)
        # 3. Flatten
        self.user_group_book_author =\
            torch.tensordot(torch.unsqueeze(self.user_group_embedding, 1), self.book_author_embedding, dims=([1], [0])) # shape=(batch_size, num_features)
        self.user_group_book_author = torch.reshape(self.user_group_book_author, shape=(model_batch_size, -1))
       
        self.user_group_book_publisher =\
            torch.tensordot(torch.unsqueeze(self.user_group_embedding, 1), self.book_publisher_embedding, dims=([1], [0])) # shape=(batch_size, num_features)
        self.user_group_book_publisher = torch.reshape(self.user_group_book_publisher, shape=(model_batch_size, -1))
        
        self.user_group_book_year =\
            torch.tensordot(torch.unsqueeze(self.user_group_embedding, 1), self.book_year_embedding, dims=([1], [0])) # shape=(batch_size, num_features)
        self.user_group_book_year = torch.reshape(self.user_group_book_year, shape=(model_batch_size, -1))

        self.book_author_book_publisher =\
            torch.tensordot(torch.unsqueeze(self.book_author_embedding, 1), self.book_book_publisher, dims=([1], [0]))
        self.book_author_book_publisher = torch.reshape(self.book_author_book_publisher, shape=(model_batch_size, -1))

        self.book_author_book_year =\
            torch.tensordot(torch.unsqueeze(self.book_author_embedding, 1), self.book_year_embedding, dims=([1], [0]))
        self.book_author_book_year = torch.reshape(self.book_author_book_year, shape=(model_batch_size, -1))

        self.book_publisher_book_year =\
            torch.tensordot(torch.unsqueeze(self.book_publisher_embedding, 1), self.book_year_embedding, dims=([1], [0]))
        self.book_publisher_book_year = torch.reshape(self.book_publisher_book_year, shape=(model_batch_size, -1))

        # Concatenate
        self.all = torch.cat((self.user_group_book_author, self.user_group_book_publisher, self.user_group_book_year,\
                              self.book_author_book_publisher, self.book_author_book_year, self.book_publisher_book_year,\
                              self.user_group_weight, self.book_author_weight, self.book_publisher_weight, self.book_year_weight), dim=-1) 

        # Decoder
        X = self.decoder(self.all)

        # identify regression or classification task
        if self.methods == "regression":
            return X
        else:
            return nn.Sigmoid()(X)

# PIN model
class pin_model(nn.Module):
    def __init__(self, num_user_group, num_book_author, num_book_publisher, num_book_year, num_features, methods):
        super(pin_model, self).__init__()
        num_decoder = 3 * num_features
        self.methods = methods
        self.user_group = nn.Linear(num_user_group, num_features)
        self.book_author = nn.Linear(num_book_author, num_features)
        self.book_publisher = nn.Linear(num_book_publisher, num_features)
        self.book_year = nn.Linear(num_book_year, num_features)
        self.user_group_weight_linear = nn.Linear(num_user_group, 1)
        self.book_author_weight_linear = nn.Linear(num_book_author, 1)
        self.book_publisher_weight_linear = nn.Linear(num_book_publisher, 1)
        self.book_year_weight_linear = nn.Linear(num_book_year, 1)
        self.decoder = nn.Sequential(
            nn.Linear(num_decoder, int(round(num_decoder/2, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_decoder/2, 0)), int(round(num_decoder/4, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_decoder/4, 0)), 1)
        )
        return

    def forward(self, user_group_feature, book_author_feature, book_publisher_feature, book_year_feature):
        # Embedding Learning
        self.user_group_embedding = self.user_group(user_group_feature) # shape=(batch_size, embedding)
        self.book_author_embedding = self.book_author(book_author_feature) # shape=(batch_size, embedding)
        self.book_publisher_embedding = self.book_publisher(book_publisher_feature) # shape=(batch_size, embedding)
        self.book_year_embedding = self.book_year(book_year_feature) # shape=(batch_size, embedding)
        self.user_group_weight = self.user_group_weight_linear(user_group_feature) # shape = (batch_size, 1)
        self.book_author_weight = self.book_author_weight_linear(book_author_feature) # shape = (batch_size, 1)
        self.book_publisher_weight = self.book_publisher_weight_linear(book_publisher_feature) # shape = (batch_size, 1)
        self.book_year_weight = self.book_year_weight_linear(book_year_feature) # shape = (batch_size, 1)

        # Inner product
        self.user_group_book_author = self.user_group * self.book_author
        self.user_group_book_publisher = self.user_group * self.book_publisher
        self.user_group_book_year = self.user_group * self.book_year
        self.book_author_book_publisher = self.book_author * self.book_publisher
        self.book_author_book_year = self.book_author * self.book_year
        self.book_publisher_book_year = self.book_publisher * self.book_year

        # Concatenate   
        self.all = F.tanh(torch.cat((self.user_group_book_author,
                                     self.user_group_book_publisher,
                                     self.user_group_book_year,
                                     self.book_author_book_publisher,
                                     self.book_author_book_year,
                                     self.book_publisher_book_year), dim=-1) )

        # Decoder
        X = self.decoder(self.all)

        # identify regression or classification task
        if self.methods == "regression":
            return X
        else:
            return nn.Sigmoid()(X)


# 建構模型→Movielens
class wide_deep_model(nn.Module):
    def __init__(self, num_user_group, num_book_author, num_book_publisher, num_book_year, num_features, methods):
        super(wide_deep_model, self).__init__()
        num_deep_decoder = 6 * num_features
        num_decoder = num_user_group + num_book_author + num_book_publisher + num_book_year + int(round(num_deep_decoder/4, 0))
        self.methods = methods
        self.user_group = nn.Linear(num_user_group, num_features)
        self.book_author = nn.Linear(num_book_author, num_features)
        self.book_publisher = nn.Linear(num_book_publisher, num_features)
        self.book_year = nn.Linear(num_book_year, num_features)
        self.user_group_weight_linear = nn.Linear(num_user_group, 1)
        self.book_author_weight_linear = nn.Linear(num_book_author, 1)
        self.book_publisher_weight_linear = nn.Linear(num_book_publisher, 1)
        self.book_year_weight_linear = nn.Linear(num_book_year, 1)        
        self.deep_decoder = nn.Sequential(
            nn.Linear(num_deep_decoder, int(round(num_deep_decoder/2, 0))),
            nn.Tanh(),
            nn.Linear(int(round(num_deep_decoder/2, 0)), int(round(num_deep_decoder/4, 0))),
            nn.Tanh(),
        )
        self.decoder = nn.Linear(num_decoder, 1)
        return

    def forward(self, user_group_feature, book_author_feature, book_publisher_feature, book_year_feature):
        # Embedding Learning
        self.user_group_embedding = self.user_group(user_group_feature) # shape=(batch_size, embedding)
        self.book_author_embedding = self.book_author(book_author_feature) # shape=(batch_size, embedding)
        self.book_publisher_embedding = self.book_publisher(book_publisher_feature) # shape=(batch_size, embedding)
        self.book_year_embedding = self.book_year(book_year_feature) # shape=(batch_size, embedding)
        self.user_group_weight = self.user_group_weight_linear(user_group_feature) # shape = (batch_size, 1)
        self.book_author_weight = self.book_author_weight_linear(book_author_feature) # shape = (batch_size, 1)
        self.book_publisher_weight = self.book_publisher_weight_linear(book_publisher_feature) # shape = (batch_size, 1)
        self.book_year_weight = self.book_year_weight_linear(book_year_feature) # shape = (batch_size, 1)

        ## Deep part
        # Inner product
        self.user_group_book_author = self.user_group * self.book_author
        self.user_group_book_publisher = self.user_group * self.book_publisher
        self.user_group_book_year = self.user_group * self.book_year
        self.book_author_book_publisher = self.book_author * self.book_publisher
        self.book_author_book_year = self.book_author * self.book_year
        self.book_publisher_book_year = self.book_publisher * self.book_year

        # Concatenate
        self.deep_part_all = torch.cat((self.user_group_book_author, self.user_group_book_publisher,
                                        self.user_group_book_year, self.book_author_book_publisher,
                                        self.book_author_book_year, self.book_publisher_book_year), dim=-1) 

        # Deep_Decoder
        self.deep_part_all = self.deep_decoder(self.deep_part_all)

        ## Wide part
        # Concatenate
        self.wide_part_all = torch.cat([self.user_group_embedding, self.book_author_embedding, self.book_publisher_embedding, self.book_year_embedding], dim=1)

        ## All
        # Concatenate
        self.all = torch.cat([self.wide_part_all, self.deep_part_all], dim=1)

        # Decoder
        X = self.decoder(self.all)

        if self.methods == "regression":
            return X
        else:
            return nn.Sigmoid()(X)
