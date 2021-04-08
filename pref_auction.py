import os
from datetime import datetime

import click
import h5py
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

class2idx = {
    1: 0,
    2: 1,
    8: 2,
    13: 3,
    14: 4
}

idx2class = {
    0: 1,
    1: 2,
    2: 8,
    3: 13,
    4: 14
}


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class AuctionNet(torch.nn.Module):
    def __init__(self, num_feature=10, num_class=5):
        super(AuctionNet, self).__init__()
        self.fc1 = torch.nn.Linear(num_feature, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, num_class)

        self.bn1 = torch.nn.LayerNorm(512)
        self.bn2 = torch.nn.LayerNorm(256)
        self.bn3 = torch.nn.LayerNorm(128)
        self.bn4 = torch.nn.LayerNorm(64)

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
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)

        return x


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)

    return acc


def get_class_distribution(obj):
    count_dict = {
        "call_0": 0,
        "call_1": 0,
        "call_2": 0,
        "call_3": 0,
        "call_4": 0
    }

    for i in obj:
        if i == 0:
            count_dict['call_0'] += 1
        elif i == 1:
            count_dict['call_1'] += 1
        elif i == 2:
            count_dict['call_2'] += 1
        elif i == 3:
            count_dict['call_3'] += 1
        elif i == 4:
            count_dict['call_4'] += 1
        else:
            print("Check classes.")

    return count_dict


@click.command(help="Preferans auction estimation")
@click.option('--data_path', default='/home/wingman2/datasets/pref', help='Dataset root directory')
@click.option('--lr', default=0.001, help='Learning rate')
@click.option('--num_epochs', default=500, help='Number of epochs')
@click.option('--batch_size', default=1024, help='Batch size')
@click.option('--logdir', default='/home/wingman2/pref_tb_logs', help='Path to logging directory')
@click.option('--name', default='exp-' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
@click.option('--checkpoints', required=False, help='Path to model checkpoints used as a starting point for training')
def train(**options):
    logs_dir = os.path.join(options['logdir'], options['name'])
    writer = SummaryWriter(log_dir=logs_dir)

    f_train_x = h5py.File(os.path.join(options['data_path'], 'train_x.hdf5'), 'r')
    X_train = f_train_x['train_x'][:]
    f_train_x.close()

    f_train_y = h5py.File(os.path.join(options['data_path'], 'train_y.hdf5'), 'r')
    y_train = f_train_y['train_y'][:]
    f_train_y.close()

    f_val_x = h5py.File(os.path.join(options['data_path'], 'val_x.hdf5'), 'r')
    X_val = f_val_x['val_x'][:]
    f_val_x.close()

    f_val_y = h5py.File(os.path.join(options['data_path'], 'val_y.hdf5'), 'r')
    y_val = f_val_y['val_y'][:]
    f_val_y.close()

    # Define datasets
    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]

    class_count = [i for i in get_class_distribution(y_train).values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    print(class_weights)

    class_weights_all = class_weights[target_list]

    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    # Run on CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AuctionNet()
    # Check if should load checkpoints
    if options['checkpoints']:
        model.load_state_dict(torch.load(options['checkpoints']))
    model.to(device)

    # Define data loaders
    train_loader = DataLoader(dataset=train_dataset, num_workers=12, batch_size=options['batch_size'],
                              sampler=weighted_sampler)
    val_loader = DataLoader(dataset=val_dataset, num_workers=12, batch_size=options['batch_size'])

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

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
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        val_epoch_loss /= len(val_loader)
        val_epoch_acc /= len(val_loader)
        train_epoch_loss /= len(train_loader)
        train_epoch_acc /= len(train_loader)
        scheduler.step(val_epoch_loss)
        loss_stats['train'].append(train_epoch_loss)
        loss_stats['val'].append(val_epoch_loss)
        accuracy_stats['train'].append(train_epoch_acc)
        accuracy_stats['val'].append(val_epoch_acc)
        writer.add_scalar("Test-loss-avg", val_epoch_loss, global_step=e)
        writer.add_scalar("Train-loss-avg", train_epoch_loss, global_step=e)
        writer.add_scalar("Test-accuracy", val_epoch_acc, global_step=e)
        writer.add_scalar("Train-accuracy", train_epoch_acc, global_step=e)
        print(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss:.5f} | '
              f'Val Loss: {val_epoch_loss:.5f} | '
              f'Train Acc: {train_epoch_acc:.3f}| '
              f'Val Acc: {val_epoch_acc:.3f}')
        torch.save(model.state_dict(), '/home/wingman2/models/pref/varijanta_10/auction' + str(e) + ".pt")


@click.command(help="Testing preferans auction estimation")
@click.option('--data_path', default='/home/wingman2/datasets/pref', help='Dataset root directory')
@click.option('--checkpoints', required=True, help='Path to model checkpoints')
def test(**options):
    # Load dataset from already pre-processed hdf5 files
    f_test_x = h5py.File(os.path.join(options['data_path'], 'test_x.hdf5'), 'r')
    X_test = f_test_x['test_x'][:]
    f_test_x.close()
    f_test_y = h5py.File(os.path.join(options['data_path'], 'test_y.hdf5'), 'r')
    y_test = f_test_y['test_y'][:]
    f_test_y.close()

    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    test_loader = DataLoader(dataset=test_dataset, batch_size=1024)

    model = AuctionNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(options['checkpoints']))
    model.to(device)

    y_pred_list = np.array([])

    with torch.no_grad():
        model.eval()
        for X_batch, _ in tqdm(test_loader):
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list = np.hstack((y_pred_list, y_pred_tags.cpu().numpy()))

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)
    sns.heatmap(confusion_matrix_df, annot=True)
    print(classification_report(y_test, y_pred_list))


if __name__ == '__main__':
    cli = click.Group()
    cli.add_command(train)
    cli.add_command(test)
    cli()
