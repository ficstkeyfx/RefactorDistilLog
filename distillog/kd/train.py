import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Parameter
from torch.nn.modules.module import Module
from tqdm import tqdm
import math
import csv
from time import time 
from torchinfo import summary
from utils import save_model, train
from data_utils import read_data, load_data
from utils import DistilLog

# Đọc config
with open('config.json', 'r') as f:
    config = json.load(f)

train_config = config["train"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = train_config["num_classes"]
batch_size = train_config["batch_size"]
learning_rate = train_config["learning_rate"]
hidden_size = train_config["hidden_size"]
input_size = train_config["input_size"]
sequence_length = train_config["sequence_length"]
num_layers = train_config["num_layers"]

train_path = train_config["train_path"]
save_teacher_path = train_config["save_teacher_path"]
save_noKD_path = train_config["save_noKD_path"]

Teacher = DistilLog(input_size, hidden_size, num_layers, num_classes, is_bidirectional=False).to(device)
noKD = DistilLog(input_size = input_size, hidden_size = 4, num_layers = 1, num_classes = num_classes, is_bidirectional=False).to(device)
#summary(Teacher, input_size=(50, 50, 30))

train_x, train_y = read_data(train_path, input_size, sequence_length)
train_loader = load_data(train_x, train_y, batch_size)

Teacher = train(Teacher, train_loader, learning_rate, num_epochs = 300)
noKD = train(noKD, train_loader, learning_rate, num_epochs = 300)
save_model(Teacher, save_teacher_path)
save_model(noKD, save_noKD_path)