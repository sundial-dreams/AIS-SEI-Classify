import matplotlib.pyplot as plt
from matplotlib import ticker
import torch
import types
from typing import List
import numpy as np

def draw_losses(loss):
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(loss)
    plt.show()


def draw_confusion(confusion: torch.Tensor):
    fig, ax = plt.subplots()
    im = ax.matshow(confusion.numpy())
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xlabel="True",
        ylabel="Predict"
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


def draw_features(features: torch.Tensor, labels: List[int]):
    fs = features.detach().numpy()
    
    for i in range(len(fs)):
        plt.plot(fs[i], label=labels[i])
    plt.legend()
    plt.show()
