import sys

sys.path.append("../")

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
from distillog.kd.models.utils import save_model, train
from distillog.kd.data.data_utils import read_data, load_data
from distillog.kd.models.utils import DistilLog
from distillog.kd.logging.clogging import setup_logger
from distillog.kd.arguments.arguments import get_train_args
train_logger = setup_logger("train.log")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == "__main__":
    train_args = get_train_args()

    num_classes = train_args.num_classes
    batch_size = train_args.batch_size
    learning_rate = train_args.learning_rate
    hidden_size = train_args.hidden_size
    input_size = train_args.input_size
    sequence_length = train_args.sequence_length
    num_layers = train_args.num_layers
    train_path = train_args.train_path
    save_teacher_path = train_args.save_teacher_path
    save_noKD_path = train_args.save_noKD_path
    num_epochs = train_args.num_epochs

    Teacher = DistilLog(input_size, hidden_size, num_layers, num_classes, is_bidirectional=False).to(device)
    noKD = DistilLog(input_size = input_size, hidden_size = 4, num_layers = 1, num_classes = num_classes, is_bidirectional=False).to(device)
    #summary(Teacher, input_size=(50, 50, 30))

    train_x, train_y = read_data(train_path, input_size, sequence_length, "../distillog/datasets/HDFS/pca_vector.csv")
    train_loader = load_data(train_x, train_y, batch_size)

    Teacher = train(Teacher, train_loader, learning_rate, num_epochs = num_epochs)
    noKD = train(noKD, train_loader, learning_rate, num_epochs = num_epochs)
    save_model(Teacher, save_teacher_path)
    save_model(noKD, save_noKD_path)