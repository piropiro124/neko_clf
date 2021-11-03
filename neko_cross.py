import glob
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import KFold


# ----------------------- データの準備 -------------------------
# 画像が保存されているルートディレクトリのパス
ROOT_DIR = "/Users/ayano/Downloads/neko_collector-main/neko_collector"

# 商品名（ルートディレクトリ以下のサブディレクトリ名と一致させる）
CATEGORIES = ["otonasi", "yancha"]

# ----------------------- データの準備 -------------------------


def list_files(root_dir, categories):
    # root_dir以下のファイルをcategoriesにあるディレクトリごとにリストアップする
    file_list = []

    for idx, cat in enumerate(categories):
        image_dir = root_dir + "/" + cat + "_150"
        files = glob.glob(image_dir + "/*")
        for f in files:
            file_list.append((idx, f))
    return file_list


def read_one_image(filename):
    # ファイルから画像を読み込む
    img = Image.open(filename)
    img = img.convert("RGB")
    img = img.resize((150, 150))
    return np.asfarray(img, dtype="float") / 255


def read_all_files(files):
    # ファイルごとに画像を読み込む
    X = []
    Y = []
    for cat, filename in files:
        img = read_one_image(filename)
        X.append(img)
        Y.append(cat)
    return np.array(X), np.array(Y)


def prepare_data():
    # データの準備
    file_list = list_files(ROOT_DIR, CATEGORIES)
    return read_all_files(file_list)


# ------------------- モデルの学習 ------------------------


def build_model():
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
    model.add(Dense(len(CATEGORIES), activation="sigmoid"))

    model.compile(loss=categorical_crossentropy,
                  optimizer=RMSprop(learning_rate=1e-4),
                  metrics=["acc"])
    return model


def init_history(history, num_epochs):
    keys = ["acc", "loss", "val_acc", "val_loss"]
    for k in keys:
        k_vals = []
        for e in range(num_epochs):
            k_vals.append([])
        history[k] = k_vals


def store_history(history, h):
    keys = ["acc", "loss", "val_acc", "val_loss"]
    for k in keys:
        all_key_hist = history[k]
        key_hist = h[k]
        for e in range(len(key_hist)):
            all_key_hist[e].append(key_hist[e])


batch_size = 50
num_epochs = 50
num_folds = 5

# データの読み込み
input, target = prepare_data()

# kerasで扱えるようにcategoriesをベクトルに変換
target = to_categorical(target, len(CATEGORIES))

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# 学習経過の記録
history = {}
init_history(history, num_epochs)

# K-fold Cross Validation model evaluation
fold_no = 0
for train, test in kfold.split(input, target):
    # モデルの作成
    model = build_model()
    h = model.fit(input[train], target[train],
                  epochs=num_epochs, batch_size=batch_size)
    store_history(history, h)

    eval = model.evaluate(validation_data=(input[test], target[test]))
    msg = "Score for fold {}:".format(fold_no + 1)
    for idx, metrics_name in enumerate(model.metrics_names):
        msg += " {} = {}".format(metrics_name, eval[idx])
    print(msg)

# 学習経過の整理


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('/Users/ayano/Downloads/neko_collector-main/neko_collector/accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(
    '/Users/ayano/Downloads/neko_collector-main/neko_collector/lost_value')


# モデルの保存

json_string = model.model.to_json()
open('/Users/ayano/Downloads/neko_collector-main/neko_collector/save/tea_predict.json',
     'w').write(json_string)

# 重みの保存

hdf5_file = "/Users/ayano/Downloads/neko_collector-main/neko_collector/save/tea_predict.hdf5"
model.model.save_weights(hdf5_file)


# モデルの精度を測る

# 評価用のデータの読み込み
eval_X = np.load(
    "/Users/ayano/Downloads/neko_collector-main/neko_collector/test/neko_data_test_X_150.npy", allow_pickle=True)
eval_Y = np.load(
    "/Users/ayano/Downloads/neko_collector-main/neko_collector/test/neko_data_test_Y_150.npy", allow_pickle=True)

# Yのデータをone-hotに変換

# y_test = np_utils.to_categorical(y_test, 4)

score = model.model.evaluate(x=X_test, y=y_test)

print('loss=', score[0])
print('accuracy=', score[1])
