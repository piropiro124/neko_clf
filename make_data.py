import argparse
import glob

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert images to train data.")
    parser.add_argument("-s", "--save_to", type=str,
                        help="Data filename.")
    parser.add_argument("DIRS", type=str, nargs="+",
                        help="Directories in which images in the same category are stored.")
    return parser.parse_args()


def list_files(dirs):
    # root_dir以下のファイルをcategoriesにあるディレクトリごとにリストアップする
    file_list = []

    for idx, image_dir in enumerate(dirs):
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
    inputs = []
    targets = []
    for cat, filename in files:
        img = read_one_image(filename)
        inputs.append(img)
        targets.append(cat)
    return np.array(inputs), np.array(targets)


def prepare_data(dirs):
    # データの準備
    file_list = list_files(dirs)
    return read_all_files(file_list)


def save_data(inputs, targets, filename):
    np.save(filename, (inputs, targets))


if __name__ == "__main__":
    args = parse_args()
    inputs, targets = prepare_data(args.DIRS)
    if args.save_to is not None:
        save_data(inputs, targets, args.save_to)
    else:
        print("# of images: {}".format(len(inputs)))
