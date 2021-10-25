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

class GMF_neucf_model(nn.Module):
    def __init__(self, num_user, num_item, num_features):
        super(GMF_neucf_model, self).__init__()
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