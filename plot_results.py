import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot results.")
    parser.add_argument("-s", "--save_to", type=str,
                        help="Save figures to the file.")
    parser.add_argument("FILE", type=str,
                        help="Result file.")
    return parser.parse_args()


def plot_for_metric(results, metrics, title="Results"):
    formats = ["b", "ro"]
    fig = plt.figure()
    epochs = range(len(results))
    for idx, met in enumerate(metrics):
        ave = []
        std = []
        fmt = formats[idx % len(formats)]
        for e in epochs:
            ave.append(results[e][met]["ave"])
            std.append(results[e][met]["std"])
        plt.errorbar(epochs, ave, yerr=std, fmt=fmt, label=met, capsize=3)
    plt.title(title)
    plt.legend()
    return fig


def plot_results(results, save_to):
    plot_set = [
        [["acc", "val_acc"], "Training and validation accuracy."],
        [["loss", "val_loss"], "Training and validation loss."],
    ]

    for metrics, title in plot_set:
        fig = plot_for_metric(results, metrics, title)
        if save_to is not None:
            basename, ext = os.path.splitext(save_to)
            fig.savefig("{}_{}{}".format(basename, metrics[0], ext))
        else:
            fig.show()
            input("Press Enter to proceed.")


if __name__ == "__main__":
    args = parse_args()
    if args.FILE is not None:
        results = np.load(args.FILE, allow_pickle=True)
        plot_results(results, args.save_to)
