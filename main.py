import math
import time

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms

from models.ResNet import attention_resnet
from models.ViT import ViT
from utils.constants import CLASSES
from utils.preprocess import AISDataset, AISTransform
from utils.train import train
from utils.utils import draw_confusion, draw_losses

TRAIN_SIZE = 256

BATCH_SIZE = 16

EPOCH = 100

LR = 0.05

NUM_SAMPLE = 200

TRAIN_P = 0.5


class ToDataset(Dataset):
    def __init__(self, data: list):
        super(ToDataset, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        return sample

    def __len__(self):
        return len(self.data)


def split_dataset(dataset: AISDataset, class_num: int, rate: float) -> (Dataset, Dataset):
    train_dataset = []
    validate_dataset = []
    all_samples = [[] for _ in range(class_num)]

    for data in dataset:
        all_samples[data[1]].append(data)

    for samples in all_samples:
        train_size = math.floor(len(samples) * rate)
        test_size = len(samples) - train_size
        samples = ToDataset(samples)
        train_samples, test_samples = random_split(samples, [train_size, test_size])

        train_dataset += train_samples
        validate_dataset += test_samples

    return ToDataset(train_dataset), ToDataset(validate_dataset)


def loader():  #
    transform = transforms.Compose([AISTransform(train_size=256), ])

    dataset = AISDataset("./dataset", transform=transform, num_sample=NUM_SAMPLE)

    train_dataset, test_dataset = split_dataset(dataset, 10, TRAIN_P)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    validate_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    return train_loader, validate_loader


if __name__ == "__main__":
    train_loader, validate_loader = loader()

    model = attention_resnet(num_classes=len(CLASSES))
    # model = ViT(
    #     signal_size=256,
    #     patch_size=4,
    #     num_classes=len(CLASSES),
    #     dim=16,
    #     depth=2,
    #     heads=4,
    #     mpl_dim=64,
    #     dropout=0.9,
    #     channels=1
    # )
    t = time.time()

    losses, accuracy, confusion = train(model, train_loader, validate_loader, epoch=EPOCH, learning_rate=LR)

    draw_losses(losses)

    draw_confusion(confusion)

    print("accuracy = ", accuracy)
    print("time = ", time.time() - t)

    # torch.save(model, "model_saved/cnn/model_vit_256_std_1.pt")
