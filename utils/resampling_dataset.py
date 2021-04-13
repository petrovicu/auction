import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import h5py
import numpy as np
import click


@click.command(help="Re-sampling preferans auction dataset")
@click.option('--csv_path', default='/home/wingman2/datasets/pref/pref_auction.csv', help='CSV path')
@click.option('--h5_train_x_path', default='/home/wingman2/datasets/pref/train_x.hdf5', help='File path')
@click.option('--h5_train_y_path', default='/home/wingman2/datasets/pref/train_y.hdf5', help='File path')
@click.option('--h5_val_x_path', default='/home/wingman2/datasets/pref/val_x.hdf5', help='File path')
@click.option('--h5_val_y_path', default='/home/wingman2/datasets/pref/val_y.hdf5', help='File path')
@click.option('--h5_test_x_path', default='/home/wingman2/datasets/pref/test_x.hdf5', help='File path')
@click.option('--h5_test_y_path', default='/home/wingman2/datasets/pref/test_y.hdf5', help='File path')
def re_sample_ds(**options):
    df = pd.read_csv(options['csv_path'])

    # preprocess our outputs
    class2idx = {
        1: 0,
        2: 1,
        8: 2,
        13: 3,
        14: 4
    }
    df['call'].replace(class2idx, inplace=True)

    # define input and output data
    X = df.iloc[:, 0:-1]
    Y = df.iloc[:, -1]

    print(Counter(Y))

    # define oversampling strategy
    over = RandomOverSampler(sampling_strategy={0: 37847234, 1: 22556107, 2: 5607389, 3: 5000000, 4: 5000000})
    X_over, Y_over = over.fit_resample(X, Y)
    print(Counter(Y_over))

    # then, perform under-sampling
    under = RandomUnderSampler(random_state=0)
    X_under, y_under = under.fit_resample(X_over, Y_over)

    # summarize class distribution
    print(Counter(y_under))

    # output distribution
    # sns.countplot(x='call', data=df)
    # output dist in percentages
    # ax = sns.barplot(x="call", y="call", data=df, estimator=lambda x: len(x) / len(df) * 100)
    # ax.set(ylabel="Auction call percentages")
    # plt.show()

    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(X_under, y_under, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1)

    # To scale our values, we’ll use the MinMaxScaler() from Sklearn.
    # It transforms features by scaling it to a given range which is (0,1) in our case.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    f_train_x = h5py.File(options['h5_train_x_path'], 'w')
    f_train_x.create_dataset('train_x', data=X_train)
    f_train_x.close()

    f_train_y = h5py.File(options['h5_train_y_path'], 'w')
    f_train_y.create_dataset('train_y', data=y_train)
    f_train_y.close()

    f_val_x = h5py.File(options['h5_val_x_path'], 'w')
    f_val_x.create_dataset('val_x', data=X_val)
    f_val_x.close()

    f_val_y = h5py.File(options['h5_val_y_path'], 'w')
    f_val_y.create_dataset('val_y', data=y_val)
    f_val_y.close()

    f_test_x = h5py.File(options['h5_test_x_path'], 'w')
    f_test_x.create_dataset('test_x', data=X_test)
    f_test_x.close()

    f_test_y = h5py.File(options['h5_test_y_path'], 'w')
    f_test_y.create_dataset('test_y', data=y_test)
    f_test_y.close()


if __name__ == '__main__':
    re_sample_ds()
