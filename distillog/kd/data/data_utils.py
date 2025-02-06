import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def mod(l, n):
    """ Truncate or pad a list """
    r = l[-1 * n:]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r

def read_data(path, input_size, sequence_length, pca_vector="../datasets/HDFS/pca_vector.csv"):
    fi = pd.read_csv(pca_vector, header = None)
    vec = []
    vec = fi
    vec = np.array(vec)

    logs_series = pd.read_csv(path)
    logs_series = logs_series.values
    label = logs_series[:, 1]
    logs_data = logs_series[:, 0]
    logs = []
    for i in range(0, len(logs_data)):
        ori_seq = [
            int(eventid) for eventid in logs_data[i].split()]
        seq_pattern = mod(ori_seq, sequence_length)
        vec_pattern = []

        for event in seq_pattern:
            if event == 0:
                vec_pattern.append([-1] * input_size)
            else:
                vec_pattern.append(vec[event - 1])
        logs.append(vec_pattern)
    logs = np.array(logs)
    train_x = logs
    train_y = np.array(label)
    train_x = np.reshape(train_x, (train_x.shape[0], -1, input_size))
    train_y = train_y.astype(int)

    return train_x, train_y


def load_data(train_x, train_y, batch_size):
    tensor_x = torch.Tensor(train_x)
    tensor_y = torch.from_numpy(train_y)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader