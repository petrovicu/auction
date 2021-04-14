import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import h5py


def add_csv_header(old_csv_path='E:/datasets/pref/auction.csv', new_csv_path='E:/datasets/pref/pref_auction.csv'):
    with open(old_csv_path) as fr, open(new_csv_path, "w", newline='') as fw:
        cr = csv.reader(fr)
        cw = csv.writer(fw)
        cw.writerow(["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "call"])
        cw.writerows(cr)


def _save_stratified_data_to_hdf5(csv_path='/home/wingman2/datasets/pref/pref_auction.csv',
                      h5_train_x_path='/home/wingman2/datasets/pref/train_x.hdf5',
                      h5_train_y_path='/home/wingman2/datasets/pref/train_y.hdf5',
                      h5_val_x_path='/home/wingman2/datasets/pref/val_x.hdf5',
                      h5_val_y_path='/home/wingman2/datasets/pref/val_y.hdf5',
                      h5_test_x_path='/home/wingman2/datasets/pref/test_x.hdf5',
                      h5_test_y_path='/home/wingman2/datasets/pref/test_y.hdf5'):
    """ This one should not be used. To generate hdf5 data use resampling_dataset.py script. """

    df = pd.read_csv(csv_path)
    # preprocess our outputs
    class2idx = {
        1: 0,
        2: 1,
        8: 2,
        13: 3,
        14: 4
    }
    df['call'].replace(class2idx, inplace=True)

    df = df.head(20000000)

    # define input and output data
    X = df.iloc[:, 0:-1]
    Y = df.iloc[:, -1]

    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, Y,
                                                              test_size=0.1, stratify=Y, random_state=69)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval,
                                                      test_size=0.1, stratify=y_trainval, random_state=21)

    # To scale our values, we’ll use the MinMaxScaler() from Sklearn.
    # It transforms features by scaling it to a given range which is (0,1) in our case.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    f_train_x = h5py.File(h5_train_x_path, 'w')
    f_train_x.create_dataset('train_x', data=X_train)
    f_train_x.close()

    f_train_y = h5py.File(h5_train_y_path, 'w')
    f_train_y.create_dataset('train_y', data=y_train)
    f_train_y.close()

    f_val_x = h5py.File(h5_val_x_path, 'w')
    f_val_x.create_dataset('val_x', data=X_val)
    f_val_x.close()

    f_val_y = h5py.File(h5_val_y_path, 'w')
    f_val_y.create_dataset('val_y', data=y_val)
    f_val_y.close()

    f_test_x = h5py.File(h5_test_x_path, 'w')
    f_test_x.create_dataset('test_x', data=X_test)
    f_test_x.close()

    f_test_y = h5py.File(h5_test_y_path, 'w')
    f_test_y.create_dataset('test_y', data=y_test)
    f_test_y.close()


if __name__ == '__main__':
    add_csv_header()
