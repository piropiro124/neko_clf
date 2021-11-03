import argparse
from ast import parse

import numpy as np
from sklearn.model_selection import KFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical


def read_data(data_file):
    return np.load(data_file, allow_pickle=True)


def build_model(num_categories):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu",
              input_shape=(150, 150, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(num_categories, activation="sigmoid"))

    model.compile(loss=categorical_crossentropy,
                  optimizer=RMSprop(learning_rate=1e-4),
                  metrics=["acc"])
    return model


def init_train_records(rec):
    records = []
    epochs = list(rec.values())[0]
    for _ in epochs:
        key_records = {}
        for k in rec.keys():
            key_records[k] = []
        records.append(key_records)
    return records


def store_train_record(records, rec):
    for k in rec.keys():
        for e, v in enumerate(rec[k]):
            records[e][k].append(v)


def train(inputs, targets, num_categories, num_epochs, batch_size, num_folds=5):
    # kerasで扱えるようにcategoriesをベクトルに変換
    targets = to_categorical(targets, num_categories)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # 学習経過の記録
    train_records = None

    # K-fold Cross Validation model evaluation
    fold_no = 0
    for train, test in kfold.split(inputs, targets):
        # モデルの作成
        model = build_model(num_categories)

        # training
        record = model.fit(inputs[train], targets[train],
                           epochs=num_epochs, batch_size=batch_size,
                           validation_data=(inputs[test], targets[test]),
                           verbose=1)
        if train_records is None:
            train_records = init_train_records(record)
        store_train_record(train_records, record)

        # 最後にテストして結果を表示
        s = model.evaluate(validation_data=(inputs[test], targets[test]),
                           verbose=0)
        msg = "Fold {}: ".format(fold_no + 1)
        for idx, metrics_name in enumerate(model.metrics_names):
            msg = msg + " {} = {}".format(metrics_name, s[idx])
        print(msg)

        fold_no = fold_no + 1

    # foldごとの平均と標準偏差を計算
    train_stat_records = []
    for epoch_record in train_records:
        epoch_stats = {}
        for k in epoch_record.keys():
            epoch_stats[k] = {"ave": np.mean(epoch_record[k]),
                              "std": np.std(epoch_record[k])}
        train_stat_records.append(epoch_stats)

    return train_stat_records


def parse_args():
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("-d", "--data_file", type=str, nargs=1,
                        help="Data file.")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=20,
                        help="Batch size.")
    parser.add_argument("-f", "--num_folds", type=int, default=5,
                        help="# of folds in Cross Validation.")
    parser.add_argument("-r", "--results", type=str,
                        help="File to which results will be saved.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inputs, targets, num_categories = read_data(args.data_file[0])
    train_stat_records = train(inputs,
                               targets,
                               num_categories,
                               args.epochs,
                               args.batch_size,
                               args.num_folds)
    if args.results is not None:
        np.save(args.results, train_stat_records)
    else:
        for idx, epoch_stat in enumerate(train_stat_records):
            msg = "{}".format(idx)
            for metric in train_stat_records[0].keys():
                ave = epoch_stat[metric]["ave"]
                std = epoch_stat[metric]["std"]
                msg = msg + ", {}: {} ({})".format(metric, ave, std)
            print(msg)
