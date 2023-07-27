#! /usr/bin/env python3

import os
import time

import torch
import torch.nn as nn

from utils.network_model import Net
from utils.TrainingClass import TrainingClass
from utils.utils_functions import (
    plot_results,
    get_args,
    check_args,
)


def mainTrainLoop(args):
    # Check the arguments
    check_args(args)

    # Create an instance of the network
    net = Net()

    # Define your loss function
    criterion = nn.BCELoss()

    # Define your optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", verbose=True
    )
    # Timing variables
    start_time = time.time()

    trainer = TrainingClass(
        args=args,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    training_results = trainer.train_loop()

    # Plotting the training and testing losses and accuracies
    print("Plotting...")
    plot_results(
        training_results=training_results,
        outputDir=args.outputDir,
    )

    # Total training duration
    total_duration = time.time() - start_time
    print(f"Total Training Duration: {total_duration:.2f} seconds")
    return


if __name__ == "__main__":
    mainTrainLoop(args=get_args())
    print("-------------------- Program finished --------------------")
