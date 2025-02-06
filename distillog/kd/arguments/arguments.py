from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class TeachArguments:
    num_classes: int = field(default=2, metadata={"help": "Number of output classes"})
    num_epochs: int = field(default=100, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=50, metadata={"help": "Batch size for training"})
    input_size: int = field(default=30, metadata={"help": "Input feature size"})
    sequence_length: int = field(default=50, metadata={"help": "Length of input sequences"})
    num_layers: int = field(default=1, metadata={"help": "Number of layers in the model"})
    hidden_size: int = field(default=4, metadata={"help": "Size of hidden layers"})
    train_path: str = field(default='../distillog/datasets/HDFS/train.csv', metadata={"help": "Path to training data"})
    save_teacher_path: str = field(default='../distillog/datasets/HDFS/model/teacher.pth', metadata={"help": "Path to save the teacher model"})
    save_student_path: str = field(default='../distillog/datasets/HDFS/model/student.pth', metadata={"help": "Path to save the student model"})

def get_teach_args():
    parser = HfArgumentParser(TeachArguments)
    args = parser.parse_args_into_dataclasses()
    return args[0]

@dataclass
class TrainArguments:
    num_classes: int = field(default=2, metadata={"help": "Number of output classes"})
    batch_size: int = field(default=50, metadata={"help": "Batch size for training"})
    learning_rate: float = field(default=0.0003, metadata={"help": "Learning rate for optimizer"})
    hidden_size: int = field(default=128, metadata={"help": "Size of hidden layers"})
    input_size: int = field(default=30, metadata={"help": "Input feature size"})
    sequence_length: int = field(default=50, metadata={"help": "Length of input sequences"})
    num_layers: int = field(default=2, metadata={"help": "Number of layers in the model"})
    train_path: str = field(default='../distillog/datasets/HDFS/train.csv', metadata={"help": "Path to training data"})
    save_teacher_path: str = field(default='../distillog/datasets/HDFS/model/teacher.pth', metadata={"help": "Path to save the teacher model"})
    save_noKD_path: str = field(default='../distillog/datasets/HDFS/model/noKD.pth', metadata={"help": "Path to save the no KD model"})
    num_epochs: int = field(default=300, metadata={"help": "Number of training epochs"})

def get_train_args():
    parser = HfArgumentParser(TrainArguments)
    args = parser.parse_args_into_dataclasses()
    return args[0]

@dataclass
class TestArguments:
    batch_size: int = field(default=50, metadata={"help": "Batch size for testing"})
    input_size: int = field(default=30, metadata={"help": "Input feature size"})
    sequence_length: int = field(default=50, metadata={"help": "Length of input sequences"})
    hidden_size: int = field(default=128, metadata={"help": "Size of hidden layers"})
    num_layers: int = field(default=2, metadata={"help": "Number of layers in the model"})
    num_classes: int = field(default=2, metadata={"help": "Number of output classes"})
    split: int = field(default=50, metadata={"help": "Training/testing split ratio"})
    save_teacher_path: str = field(default='../distillog/datasets/HDFS/model/teacher.pth', metadata={"help": "Path to saved teacher model"})
    save_student_path: str = field(default='../distillog/datasets/HDFS/model/student.pth', metadata={"help": "Path to saved student model"})
    save_noKD_path: str = field(default='../distillog/datasets/HDFS/model/noKD.pth', metadata={"help": "Path to saved noKD model"})
    test_path: str = field(default='../distillog/datasets/HDFS/test.csv', metadata={"help": "Path to test data"})
    save_quantized_path: str = field(default='../distillog/datasets/HDFS/model/quantized_model.pth', metadata={"help": "Path to save quantized model"})
    pca_vector: str = field(default='../distillog/datasets/HDFS/pca_vector.csv', metadata={"help": "Path to PCA vector data"})

def get_test_args():
    parser = HfArgumentParser(TestArguments)
    args = parser.parse_args_into_dataclasses()
    return args[0]