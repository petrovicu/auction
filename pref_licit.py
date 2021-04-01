import os

import click
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class AuctionNet(torch.nn.Module):
    def __init__(self, num_feature=10, num_class=1):
        super(AuctionNet, self).__init__()
        self.fc1 = torch.nn.Linear(num_feature, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, num_class)

        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(64)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)

        return x


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)

    return acc


@click.command(help="Preferans auction estimation")
@click.option('--data_path', default='E:/datasets/pref', help='Dataset root directory')
@click.option('--lr', default=0.001, help='Learning rate')
@click.option('--num_epochs', default=100, help='Number of epochs')
@click.option('--batch_size', default=16, help='Batch size')
def train(**options):
    df = pd.read_csv(os.path.join(options['data_path'], 'pref_auction.csv'))
    # sns.countplot(x='call', data=df)

    # define input and output data
    X = df.iloc[:, 0:-1]
    Y = df.iloc[:, -1]

    # preprocess our outputs
    class2idx = {
        1: 0,
        2: 1,
        8: 2,
        13: 3,
        14: 4
    }
    idx2class = {v: k for k, v in class2idx.items()}
    df['call'].replace(class2idx, inplace=True)

    # Split into train+val and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Split train into train-val
    # X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1)

    # To scale our values, we’ll use the MinMaxScaler() from Sklearn.
    # It transforms features by scaling it to a given range which is (0,1) in our case.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    # X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Define datasets
    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    # val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    train_loader = DataLoader(dataset=train_dataset, num_workers=12, batch_size=options['batch_size'], shuffle=True)
    # val_loader = DataLoader(dataset=val_dataset, num_workers=12, batch_size=options['batch_size'])
    test_loader = DataLoader(dataset=test_dataset, num_workers=12, batch_size=options['batch_size'])

    # Run on CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AuctionNet()
    model.to(device)

    # class_weights = 1. / torch.tensor(5, dtype=torch.float)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'])

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    print("Begin training.")
    for e in tqdm(range(1, options['num_epochs'] + 1)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.eval()
            for X_val_batch, y_val_batch in test_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(test_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(test_loader))
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | '
              f'Val Loss: {val_epoch_loss/len(test_loader):.5f} | '
              f'Train Acc: {train_epoch_acc/len(train_loader):.3f}| '
              f'Val Acc: {val_epoch_acc/len(test_loader):.3f}')
        torch.save(model.state_dict(), 'E:/models/pref/auction' + str(e) + ".pt")


if __name__ == '__main__':
    train()
