import torch
import scipy.io

USE_GPU = torch.cuda.is_available()

GPUS = [0]

DATASET_PATH = "./dataset"

CLASSES = [95, 119, 198, 238, 339, 401, 450, 873, 875, 939]

CLASSES_INDEX = {
    95: 0,
    119: 1,
    198: 2,
    238: 3,
    339: 4,
    401: 5,
    450: 6,
    873: 7,
    875: 8,
    939: 9
}

TRAIN_SIZE = 256

BATCH_SIZE = 2

EPOCH = 100


# 256 0.05
LR = 0.05




# USE_GPU = False
