import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class BiasDetectionPreprocessor:
    def __init__(self, sensitive_attributes):
        self.sensitive_attributes = sensitive_attributes
    
    def preprocess(self, dataset):
        sensitive_features = dataset[self.sensitive_attributes].values
        return dataset.drop(columns=self.sensitive_attributes), torch.tensor(sensitive_features)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim if _ == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class AdversarialDebiasingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_sensitive_features):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim if _ == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_dim, num_sensitive_features))
        self.debiasing_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.debiasing_net(x)
