"""
File that contains functions to visualize the data.
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distribution_of_events(df_acc_rej: pd.DataFrame,
                                hline_y: int) -> None:
    """
    Plot distribution of events in signal regions.

    1. Get number of accepted events per signal region.
    2. Plot.

    Args:
        df_acc_rej (pd.DataFrame): Dataframe with accepted or rejected events.
        hline_y (int): Y value for horizontal line.

    Returns:
        None
    """
    # 1. Get number of accepted events per signal region.
    x = np.arange(len(df_acc_rej.columns))
    y = np.array(df_acc_rej.sum())
    # 2. Plot in histogram like plot.
    plt.bar(x=x,
            height=y,
            label="Accepted Events below Cut",
            color="blue",
            align="center",
            width=1,
            alpha=0.7,
            linewidth=1)

    plt.hlines(y=hline_y,
               xmin=x[0],
               xmax=x[-1],
               label="Cut",
               colors="black",
               linestyles="solid")
    # 3. Additional bar plot that marks the bars that are above the horizontal
    #    line.
    plt.bar(x=x[y > hline_y],
            height=y[y > hline_y],
            label="Accepted Events above Cut",
            color="green",
            align="center",
            width=1,
            alpha=0.7,
            linewidth=1)
    plt.ylabel("Accepted Events")
    plt.xlabel("Signal Regions")
    plt.legend()
    plt.show()
    # 4. Print number of signal regions above and below the horizontal line.
    print("Number of signal regions above cut: "
          f"{len(y[y > hline_y])} of {len(y)}")
    print("Number of signal regions below cut: "
          f"{len(y[y <= hline_y])} of {len(y)}")
    return


def plot_acc_loss_curve(model) -> None:
    """
    Plot the loss curve of the model.

    1. Get loss from model.
    2. Get accuracy from model.
    3. Plot the loss and accuracy.

    Args:
        model (Model): Model object.

    Returns:
        None
    """

    # 1. Get loss from model.
    train_loss = model.tl

    # 2. Get accuracy from model.
    train_acc = model.ta

    # 3. Plot the loss and accuracy.
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.plot(train_loss, label="Train Loss", color="blue")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel("Accuracy")  # we already handled the x-label with ax1
    ax2.plot(train_acc, label="Train Accuracy", color="red")
    ax2.tick_params(axis="y")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return
