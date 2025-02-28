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


def format_duration(seconds):
    """Format a duration given in seconds to a string D-HH:MM:SS"""
    days = seconds // 86400
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_simulations(args, is_train=True):
    if is_train:
        print("Loading training data...")
    else:
        print("Loading testing data...")

    E_range = np.linspace(
        args.energyStart,  # energy starting point
        args.energyEnd,  # energy end point
        int((args.energyEnd - args.energyStart) / 0.1)
        + 1,  # number of energies +1 because of the linspace
    ).round(1)

    # Initialize an empty dataframe for both gammas and protons
    df = pd.DataFrame()
    # Load the data
    for primary in ["gamma", "proton"]:
        # Initialize an empty dataframe for the current primary particle
        dataFrame = pd.DataFrame()
        print(f"Loading {primary} data...")
        for E in E_range:
            fileName = f"{args.inputDir}/{primary}/{args.year}/hdf5/{primary}_2012_E{E}_{'train' if is_train else 'test'}.hdf5"
            # Do a few checks
            if E > 7.0:
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
        exit()

    # Shuffle the dataframe (if desired)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def check_inputs(df):
    # Check if there are invalid values in the dataframe
    indexToDrop = []
    for col in df.columns:
        if col in [
            "Laputop3s3s_fit_status",
            "Laputop_fit_status",
            "I3EventHeader",
        ]:
            continue
        elif col in [
            "MapSLCt",
            "MapSLCq",
            "MapHLCt",
            "MapHLCq",
            "Map_SLC_HLC_q",
            "Map_SLC_HLC_qt",
        ]:
            for i, el in enumerate(df[col]):
                # Check for invalid values
                if np.sum(np.logical_not(np.isfinite(el))):
                    # Get the index of the event with invalid values
                    indexToDrop.append(i)

        else:
            if np.sum(False * np.isfinite(df[col])):
                print(col)
                exit()
    # Drop the events with invalid values
    df = df.drop(index=indexToDrop, axis=0)
    return df


def read_df_from_file(fileName, apply_cuts=True):
    df = pd.read_hdf(fileName, key="data")
    if apply_cuts:
        radius = np.sqrt(df["Laputop3s3s_x"] ** 2 + df["Laputop3s3s_y"] ** 2)
        mask = (
            (df["Laputop3s3s_fit_status"] == "OK")
            & (radius < 500.0)
            & (df["Laputop3s3s_zenith"] < np.deg2rad(38.0))
            & (df["Laputop3s3s_beta"] < 4.5)
            & (df["Laputop3s3s_beta"] > 1.9)
        )

        df = df[mask]

        df = check_inputs(df.sample(frac=1).reset_index(drop=True))
    return df.sample(frac=1).reset_index(drop=True)


def load_data(fileName):
    print("Loading data", fileName)
    # Initialize an empty dataframe
    dataFrame = pd.DataFrame()

    df = read_df_from_file(fileName, apply_cuts=True)
    # Concatenate the dataframe with the next dataset
    dataFrame = pd.concat(
        [
            dataFrame,
            df,
        ],
        ignore_index=True,
    )

    # Data are labeled as -1 since they are unknown
    dataFrame["output"] = -1 * np.ones(len(dataFrame))
    # Weights are set to 1 since there is no need to weight the data
    dataFrame["weights"] = np.ones(len(dataFrame))
    # Shuffle the dataframe
    dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)
    return dataFrame


def make_input_tensors(df, weight_index=-3.0):
    # MapHLCq
    MapHLCq = np.array((df["MapHLCq"].values).tolist()).astype(float)
    check_if_map_is_valid(MapHLCq)
    MapHLCq_tensor = torch.from_numpy(MapHLCq).view(-1, 1, 10, 10, 2).float()

    # MapSLCq
    # MapSLCq = np.array((df["MapSLCq"].values).tolist()).astype(float)
    # check_if_map_is_valid(MapSLCq)
    # SumMapSLCq = np.sum(MapSLCq, axis=(1, 2, 3))
    # SumMapSLCq_tensor = torch.from_numpy(SumMapSLCq).view(-1, 1).float()
    SumMapSLCq_tensor = (
        torch.from_numpy(
            np.array((df["NumberOfSLCPulses"].values).tolist()).astype(int)
        )
        .view(-1, 1)
        .float()
    )

    # MapHLCt
    MapHLCt = np.array((df["MapHLCt"].values).tolist()).astype(float) * 1e3
    check_if_map_is_valid(MapHLCt)

    # MapSLCt
    # MapSLCt = np.array((df["MapSLCt"].values).tolist()).astype(float) * 1e3
    # check_if_map_is_valid(MapSLCt)

    # ArrayTime
    # ArrayTime = np.zeros((len(df), 1))

    for i in range(MapHLCt.shape[0]):
        # Set the starting time of each time map to 1
        if np.sum(MapHLCt[i] != 0.0):
            MapHLCt[i][MapHLCt[i] != 0.0] -= (
                np.amin(MapHLCt[i][MapHLCt[i] != 0.0]) + 1.0
            )
        #     # Get the minimum time and the maximum time of HLC and SLC time maps
        # mapTime = np.concatenate((MapHLCt[i], MapSLCt[i]), axis=0)
        # if np.sum(mapTime[mapTime != 0.0]):
        #     maxTime = np.amax(mapTime[mapTime != 0.0])
        #     minTime = np.amin(mapTime[mapTime != 0.0])
        #     # Set the ArrayTime to the time interval
        #     ArrayTime[i] = maxTime - minTime

    # ArrayTime_tensor = torch.from_numpy(ArrayTime).view(-1, 1).float()

    MapHLCt_tensor = torch.from_numpy(MapHLCt).view(-1, 1, 10, 10, 2).float()

    log10_S125_tensor = (
        torch.tensor(df["Laputop3s3s_Log10_S125"].values).view(-1, 1).float()
    )
    zenith_tensor = torch.tensor(df["Laputop3s3s_zenith"].values).view(-1, 1).float()
    beta_tensor = torch.tensor(df["Laputop3s3s_beta"].values).view(-1, 1).float()

    fccInput_tensor = torch.cat(
        (
            log10_S125_tensor,
            zenith_tensor,
            beta_tensor,
            SumMapSLCq_tensor,
            # ArrayTime_tensor,  # TODO comment, uncomment depending on the network
        ),
        dim=1,
    )

    check_if_tensor_is_valid(fccInput_tensor)

    output_tensor = torch.tensor(df["output"].values).view(-1, 1).float()
    # weights = torch.tensor(df["weights"].values).view(-1, 1).float()

    w = df["weights-3"].values
    updated_weights = rescale_weights(w)
    weights = torch.from_numpy(updated_weights).view(-1, 1).float()

    tensor_dict = {
        "MapHLCq": MapHLCq_tensor,
        "MapHLCt": MapHLCt_tensor,
        "fccInput": fccInput_tensor,
        "output": output_tensor,
        "weights": weights,
    }
    return tensor_dict


def check_if_map_is_valid(Map):
    if np.sum(np.logical_not(np.isfinite(Map))):
        for i, el in enumerate(Map):
            if np.sum(np.logical_not(np.isfinite(el))):
                print(el)
                print(el[np.logical_not(np.isfinite(el))])
                print(np.sum(np.logical_not(np.isfinite(el))))
                print("Invalid MAP values")
                print("Index", i)
                print(el.shape)
                exit()
                Map[np.logical_not(np.isfinite(Map))] = 0.0
    return


def check_if_tensor_is_valid(tensor):
    if torch.sum(torch.logical_not(torch.isfinite(tensor))):
        print(tensor[torch.logical_not(torch.isfinite(tensor))])
        print(torch.sum(torch.logical_not(torch.isfinite(tensor))))
        print("Invalid input tensor2")
        sys.exit(1)
        return


def plot_results(training_results, outputDir):
    train_losses = training_results["train_losses"]
    val_losses = training_results["val_losses"]
    test_losses = training_results["test_losses"]
    #
    train_accuracies = training_results["train_accuracies"]
    val_accuracies = training_results["val_accuracies"]
    test_accuracies = training_results["test_accuracies"]
    num_epochs = len(train_losses)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Testing Losses")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracies")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{outputDir}/plots/training_results.png")
    plt.close()


def rescale_weights(weights, minimum=1.0):
    print(f"Reseting smallest weight to {minimum}")
    weights_array = np.array(weights)
    weights_array /= np.amin(weights_array)
    weights_array *= minimum
    return weights_array
