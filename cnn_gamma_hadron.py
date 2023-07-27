#! /usr/bin/env python3

import os
import time

import torch
import torch.nn as nn

from utils.network_model import Net
from utils.utils_functions import (
    load_data,
    plot_results,
    train,
    test,
    get_args,
    check_args,
)


def mainTrainLoop(args):
    # Check the arguments
    check_args(args)

    # Load and preprocess the training data
    (
        train_input_tensor,
        train_input_tensor2,
        train_output_tensor,
        train_weights,
    ) = load_data(args=args, is_train=True)

    # Load and preprocess the test data
    (
        test_input_tensor,
        test_input_tensor2,
        test_output_tensor,
        test_weights,
    ) = load_data(args=args, is_train=False)

    # Create an instance of the network
    net = Net()

    # Define your loss function
    criterion = nn.BCELoss()

    # Define your optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", verbose=True
    )

    # Lists to store the training and testing losses and accuracies
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Timing variables
    start_time = time.time()

    # Training loop
    for epoch in range(args.numEpochs):
        # Start time of the current epoch
        epoch_start_time = time.time()

        # Training
        train_loss, train_accuracy = train(
            args,
            net,
            train_input_tensor,
            train_input_tensor2,
            train_output_tensor,
            train_weights,
            criterion,
            optimizer,
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Testing
        test_loss, test_accuracy = test(
            net,
            test_input_tensor,
            test_input_tensor2,
            test_output_tensor,
            criterion,
            test_weights,
        )
        scheduler.step(test_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Calculate the duration of the current epoch
        epoch_duration = time.time() - epoch_start_time

        print(
            ""
            + f"Epoch: {epoch+1}/{args.numEpochs}, "
            + f"Train Loss: {train_loss:.4f}, "
            + f"Test Loss: {test_loss:.4f}, "
            + f"Train Accuracy: {train_accuracy:.4f}, "
            + f"Test Accuracy: {test_accuracy:.4f}, "
            + f"Epoch Duration: {epoch_duration:.2f} seconds"
        )
        # Save the model
        torch.save(
            net.state_dict(),
            f"{args.outputDir}/model/model_epoch{str(epoch).zfill(4)}.pth",
        )
        # Save the training and testing losses and accuracies
        torch.save(
            {
                "train_losses": train_losses,
                "test_losses": test_losses,
                "train_accuracies": train_accuracies,
                "test_accuracies": test_accuracies,
            },
            f"{args.outputDir}/model/losses_accuracies.pth",
        )

    print("Saving the model...")
    # Save the model
    torch.save(net.state_dict(), f"{args.outputDir}/model/model.pth")

    # Plotting the training and testing losses and accuracies
    print("Plotting...")
    plot_results(
        args=args,
        train_losses=train_losses,
        test_losses=test_losses,
        train_accuracies=train_accuracies,
        test_accuracies=test_accuracies,
    )

    # Total training duration
    total_duration = time.time() - start_time
    print(f"Total Training Duration: {total_duration:.2f} seconds")
    return


if __name__ == "__main__":
    mainTrainLoop(args=get_args())
    print("-------------------- Program finished --------------------")
