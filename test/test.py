import torch
import math
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms
from utils.preprocess import AISDataset, AISTransform
from utils.constants import TRAIN_SIZE, DATASET_PATH, BATCH_SIZE
from utils.train import test
from utils.utils import draw_confusion, draw_features
from models.ResNet import ResNet


def loader():
    transform = transforms.Compose([AISTransform(train_size=256), ])

    dataset = AISDataset("../dataset", num_sample=80, transform=transform, train=False)

    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    return test_loader


if __name__ == "__main__":
    test_loader = loader()
    model: ResNet = torch.load("../model_saved/cnn/model_cbam_layer5_256_80_sample_1.pt")
    # for i, (x, y) in enumerate(validate_loader):
    #     f = model.feature_extract(x)
    #     draw_features(f, y)
    #     break
    accuracy, confusion = test(model, test_loader)
    print("accuracy = ", accuracy)

    draw_confusion(confusion)
