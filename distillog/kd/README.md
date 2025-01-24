## Usage

- Train Teacher and noKD student model: ```python train.py```
- KD Student model learn from Teacher: ```python teach.py```
- Test Teacher, KD Student, noKD Student: ```python test.py```


## File

#### 1. ```config.json```

This file contains configuration settings for the training, testing, and knowledge distillation (KD) processes. It includes parameters:

- Number of classes, batch size, learning rate, input size, and sequence length.
- File paths for datasets and saved models.

#### 2. ```utils.py```

Contains helper functions and model definitions, including:

- ```DistilLog```: The main neural network architecture using GRU layers and attention mechanisms.

- ```load_model``` and ```save_model```: Functions to load and save model weights.

#### 3. ```data_utils.py```

This file provides utilities for processing input data, including:

- ```read_data```: Reads and preprocesses input data into vectors using PCA.

- ```load_data```: Converts processed data into DataLoader for training and testing.

#### 4. ```attention_layers.py```

This file implements custom attention mechanisms used in the DistilLog model. Includes:

- Linear attention layers.

#### 5. ```train.py```

Training of teacher and no-KD (non-distilled) student models. Including:

- Loads data and model configurations.

- Trains models.

- Saves trained models.

#### 6. ```teach.py```

Knowledge distillation (KD) training:

- Loading configurations.

- Using pre-trained teacher model to train smaller student model.

- Saving trained student model.

#### 7. ```test.py```

Evaluate trained models on a test dataset.

- Loads configurations.

- Evaluates student models on performance metrics such as precision, recall, F1-score, and accuracy.

- Outputs detailed testing results for analysis.

#### 8. ```bgl_test.py```

This script tests models on BGL logs. It is similar to ```test.py``` but for the BGL dataset, with custom configurations:

- Paths for BGL datasets and model checkpoints.

- Batch size and sequence length adjustments.
