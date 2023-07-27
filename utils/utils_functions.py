#!/usr/bin/env python3

import os
import pathlib
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputDir", help="input folder", type=str)
    parser.add_argument("-o", "--outputDir", help="output folder", type=str)
    parser.add_argument("-y", "--year", help="year of the simulation", type=int)
    parser.add_argument("--energyStart", help="energy start", type=float)
    parser.add_argument("--energyEnd", help="energy end", type=float)
    parser.add_argument("--batchSize", help="batch size", type=int)
    parser.add_argument("--numEpochs", help="number of epochs", type=int)
    return parser.parse_args()


def check_args(args):
    if not os.path.isdir(args.inputDir):
        print(f"{args.inputDir} is not a directory")
        sys.exit(1)
    if not os.path.isdir(args.outputDir):
        print(f"{args.outputDir} is not a directory, creating it...")
        pathlib.Path(args.outputDir).mkdir(parents=True, exist_ok=True)

    # Creating the subfolders for the plots and the model
    pathlib.Path(f"{args.outputDir}/model").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{args.outputDir}/plots").mkdir(parents=True, exist_ok=True)
    return


def load_data(args, is_train=True):
    if is_train:
        print("Loading training data...")
    else:
        print("Loading testing data...")

    E_range = np.around(
        np.arange(args.energyStart, args.energyEnd, 0.1),
        decimals=1,
    )
    # Initialize an empty dataframe
    df = pd.DataFrame()
    # Load the data
    for primary in ["gamma", "proton"]:
        dataFrame = pd.DataFrame()
        print(f"Loading {primary} data...")
        for E in E_range:
            fileName = f"{args.inputDir}/{primary}/{args.year}/hdf5/{primary}_2012_E{E}_{'train' if is_train else 'test'}.hdf5"
            # Do a few checks
            if E >= 7.0:
                print(f"Skipping E = {E}")
                continue
            if not os.path.isfile(fileName):
                print(f"{os.path.basename(fileName)} not found")
                continue

            # Concatenate the dataframe with the next dataset
            dataFrame = pd.concat(
                [
                    dataFrame,
                    pd.read_hdf(
                        fileName,
                        key="data",
                    ),
                ],
                ignore_index=True,
            )

        # Assign output values based on the primary particle
        dataFrame["output"] = (
            np.ones(len(dataFrame)) if primary == "gamma" else np.zeros(len(dataFrame))
        )

        # Append the data to the main dataframe
        df = pd.concat(
            [df, dataFrame],
            ignore_index=True,
        )

    print("Total events", len(df))
    # Print how many gammas and proton events there are and the ratio
    gammaEvents = len(df[df["output"] == 1])
    protonEvents = len(df[df["output"] == 0])
    print("Gamma events", gammaEvents)
    print("Proton events", protonEvents)
    print("Ratio Gamma/Proton", gammaEvents / protonEvents)

    # Check if there are any invalid values
    if np.sum(np.logical_not(np.isfinite(df["output"].values))):
        print(df["output"].values[np.logical_not(np.isfinite(df["output"].values))])
        print(np.sum(np.logical_not(np.isfinite(df["output"].values))))
        print("Invalid output values")

    # Shuffle the dataframe (if desired)
    df = df.sample(frac=1).reset_index(drop=True)
    return make_input_tensors(df)


def make_input_tensors(df):
    MapHLCq = np.array((df["MapHLCq"].values).tolist()).astype(float)

    # Check for invalid values
    if np.sum(np.logical_not(np.isfinite(MapHLCq))):
        for i, el in enumerate(MapHLCq):
            if np.sum(np.logical_not(np.isfinite(el))):
                print(el)
                print(el[np.logical_not(np.isfinite(el))])
                print(np.sum(np.logical_not(np.isfinite(el))))
                print("Invalid MAP values")
                print("Index", i)
                print(el.shape)
                exit()
                MapHLCq[np.logical_not(np.isfinite(MapHLCq))] = 0.0

    MapHLCq_tensor = torch.from_numpy(MapHLCq).view(-1, 1, 10, 10, 2).float()

    MapSLCq = np.array((df["MapSLCq"].values).tolist()).astype(float)
    MapSLCq = np.sum(MapSLCq, axis=(1, 2, 3))
    SumMapSLCq_tensor = torch.from_numpy(MapSLCq).view(-1, 1).float()

    log10_S125_tensor = (
        torch.tensor(
            df["Laputop3s3s_Log10_S125"].values,
        )
        .view(-1, 1)
        .float()
    )
    zenith_tensor = (
        torch.tensor(
            df["Laputop3s3s_zenith"].values,
        )
        .view(-1, 1)
        .float()
    )
    beta_tensor = (
        torch.tensor(
            df["Laputop3s3s_beta"].values,
        )
        .view(-1, 1)
        .float()
    )

    fccInput_tensor = torch.cat(
        (
            log10_S125_tensor,
            zenith_tensor,
            beta_tensor,
            SumMapSLCq_tensor,
        ),
        dim=1,
    )

    if torch.sum(torch.logical_not(torch.isfinite(fccInput_tensor))):
        print(fccInput_tensor[torch.logical_not(torch.isfinite(fccInput_tensor))])
        print(torch.sum(torch.logical_not(torch.isfinite(fccInput_tensor))))
        print("Invalid input tensor2")
        sys.exit(1)

    output_tensor = torch.tensor(df["output"].values).view(-1, 1).float()
    weights = torch.tensor(df["weights"].values).view(-1, 1).float()
    return MapHLCq_tensor, fccInput_tensor, output_tensor, weights


def train(
    args,
    net,
    train_input_tensor,
    train_input_tensor2,
    train_output_tensor,
    weights,
    criterion,
    optimizer,
):
    net.train()  # Set the network in training mode
    num_samples = train_input_tensor.size(0)
    num_batches = (num_samples + args.batchSize - 1) // args.batchSize

    total_loss = 0.0
    total_accuracy = 0.0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batchSize
        end_idx = min(start_idx + args.batchSize, num_samples)

        optimizer.zero_grad()

        # Forward pass
        output = net(
            train_input_tensor[start_idx:end_idx],
            train_input_tensor2[start_idx:end_idx],
        )

        loss = criterion(output, train_output_tensor[start_idx:end_idx])

        accuracy = torch.mean(
            ((output > 0.5).float() == train_output_tensor[start_idx:end_idx]).float()
        )

        weighted_loss = torch.mean(
            loss * weights[start_idx:end_idx]  # * 1e5
        )  # TODO: Fix weights

        # Backward pass
        weighted_loss.backward()
        optimizer.step()

        # Check if the job is running on an dev_accelerated partition
        partition_name = os.environ.get("SLURM_JOB_PARTITION")
        is_dev_accelerated_partition = "dev_ccelerated" in partition_name.lower()
        if is_dev_accelerated_partition:
            # Print progress
            progress = (batch_idx + 1) / num_batches * 100
            print(
                f"Batch: [{batch_idx+1}/{num_batches}], Progress: {progress:.2f}%, Loss: {weighted_loss.item():.4f}",
                end="\r",
            )

        total_loss += weighted_loss.item()
        total_accuracy += accuracy.item()

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def test(
    net,
    test_input_tensor,
    test_input_tensor2,
    test_output_tensor,
    criterion,
    weights,
):
    with torch.no_grad():
        test_output = net(test_input_tensor, test_input_tensor2)
        loss = criterion(
            test_output[list(torch.isfinite(test_output))],
            test_output_tensor[list(torch.isfinite(test_output))],
        )

        weighted_loss = torch.mean(
            loss * weights[list(torch.isfinite(test_output))]
        ).item()

        accuracy = torch.mean(
            ((test_output > 0.5).float() == test_output_tensor).float()
        ).item()

    return weighted_loss, accuracy


def plot_results(
    args,
    train_losses,
    test_losses,
    train_accuracies,
    test_accuracies,
):
    num_epochs = len(train_losses)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Testing Losses")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracies")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{args.outputDir}/plots/training_results.png")
    plt.close()
