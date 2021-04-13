# Preferans auction estimator

## Create environment
```
conda env create -f environment.yml
```

## Dataset re-sampling
Perform this as a first step prior to any training.
Since the dataset contains very imbalanced data (~ 66m entries), re-sampling strategy is used. To perform it run:
```
python resampling_dataset.py --csv_path=DATASET_CSV_PATH 
--h5_train_x_path=YOUR_PATH_SAME_FOLDER
--h5_train_y_path=YOUR_PATH_SAME_FOLDER 
--h5_val_x_path=YOUR_PATH_SAME_FOLDER 
--h5_val_y_path=YOUR_PATH_SAME_FOLDER 
--h5_test_x_path=YOUR_PATH_SAME_FOLDER
--h5_test_y_path=YOUR_PATH_SAME_FOLDER
```
IMPORTANT - Save all hdf5 files within a same folder.

## Training
Run (with default options for training):
```
python pref_auction.py train
--data_path=HDF5_FOLDER
```

## Model evaluation
```
python pref_auction.py test
--checkpoints=PATH_TO_TRAINED_MODEL
```

## Current results
                precision    recall  f1-score   support

           0       0.86      0.86      0.86    499619
           1       0.77      0.72      0.74    500729
           2       0.89      0.87      0.88    499437
           3       0.97      1.00      0.99    500304
           4       0.93      0.98      0.96    499911

    accuracy                           0.89   2500000
    macro avg       0.89      0.89     0.89   2500000
    weighted avg    0.89      0.89     0.89   2500000
