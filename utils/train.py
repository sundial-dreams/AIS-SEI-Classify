import torch
from torch import nn, optim
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader
from utils.constants import EPOCH, LR, USE_GPU, GPUS, CLASSES


def train(model: nn.Module, train_loader: DataLoader, validate_loader: DataLoader, class_num=len(CLASSES),
          epoch=100, learning_rate=0.05) -> (list, float):
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    losses = []

    if USE_GPU:
        model = nn.DataParallel(model, device_ids=GPUS).cuda()
    for e in range(epoch):
        loss = None
        for i, (x, y) in enumerate(train_loader):

            if USE_GPU:
                x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        print("epoch = {}, loss = {}".format(e, loss.item()))
        losses.append(loss)
    accuracy, confusion = test(model, validate_loader, class_num)
    return losses, accuracy, confusion


def test(model: nn.Module, test_loader: DataLoader, class_num=len(CLASSES)) -> tuple:
    correct, count = 0, 0
    confusion = torch.zeros(class_num, class_num)

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            out = model(x)
            y_guess = torch.argmax(out, dim=1)
            correct += (y_guess == y).sum().item()
            count += len(x)

            for j in range(y_guess.size()[0]):
                confusion[y[j]][y_guess[j]] += 1

    for i in range(class_num):
        confusion[i] = confusion[i] / confusion[i].sum()

    return correct / count, confusion
