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

class neumf(nn.Module):
    def __init__(self, num_user, num_item, num_features):
        super(neumf, self).__init__()
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
        yhat = self.decoder(self.all)
        return yhat