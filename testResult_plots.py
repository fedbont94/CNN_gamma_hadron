#! /usr/bin/env python3

import glob
import pickle
import argparse

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data

from utils.network_model import Net
from utils.TrainTestClass import TrainTestClass
from utils.utils_functions import (
    load_data,
    make_input_tensors,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputDir", help="input folder", type=str)
    parser.add_argument("-m", "--modelDir", help="model folder", type=str)
    parser.add_argument("-o", "--outputDir", help="output folder", type=str)
    parser.add_argument("-y", "--year", help="year of the simulation", type=int)
    parser.add_argument("--energyStart", help="energy start", type=float)
    parser.add_argument("--energyEnd", help="energy end", type=float)
    return parser.parse_args()


################ Side functions #####################################
def calculate_openingAngle(data, reco):
    """
    calculates the angular resolution
    """
    trueXYZ = np.array(
        [
            np.sin(data["MCPrimary_zenith"]) * np.cos(data["MCPrimary_azimuth"]),  # x
            np.sin(data["MCPrimary_zenith"]) * np.sin(data["MCPrimary_azimuth"]),  # y
            np.cos(data["MCPrimary_zenith"]),  # z
        ]
    )
    reco_valueXYZ = np.array(
        [
            np.sin(data[f"{reco}_zenith"]) * np.cos(data[f"{reco}_azimuth"]),  # x
            np.sin(data[f"{reco}_zenith"]) * np.sin(data[f"{reco}_azimuth"]),  # y
            np.cos(data[f"{reco}_zenith"]),  # z
        ]
    )
    # Opening angle is defined via the scalar product of the true - reco vector as follows
    opening_angle = (np.arccos(np.sum(trueXYZ * reco_valueXYZ, axis=0)) * 180 / np.pi,)[
        0
    ]
    return opening_angle


def solidAngleBinning(
    theta_start=0,
    theta_end=np.deg2rad(5),
    numbOfBins=501,
    inDegrees=True,
):
    """
    Makes a binning in solid angle for the given theta range.
    It ensures that the solid angle in each bin is equal.
    """

    solid_angle = 2 * np.pi * (1 - np.cos(theta_end)) / (numbOfBins - 1)
    theta = np.zeros(numbOfBins)
    for i in range(numbOfBins):
        theta[i] = np.arccos(np.cos(theta_start) - (i * solid_angle / (2 * np.pi)))

    # Returns the theta binning in degrees if requested
    if inDegrees:
        return np.rad2deg(theta)
    else:
        return theta


################ Plotting functions #################################
def plot_trainLossAccuracy(args):
    print("Plotting loss accuracy")
    # Load the saved data
    saved_data = torch.load(f"{args.inputDir}/model/losses_accuracies.pth")

    # Retrieve the arrays from the loaded data
    train_losses = saved_data["train_losses"]
    test_losses = saved_data["test_losses"]
    val_losses = saved_data["val_losses"]
    train_accuracies = saved_data["train_accuracies"]
    test_accuracies = saved_data["test_accuracies"]
    val_accuracies = saved_data["val_accuracies"]
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
    # plt.show()
    plt.savefig(f"{args.outputDir}/plots/training_results.png")
    plt.close()


def plot_output(args, gamma, proton):
    print("Plotting output")
    plt.rcParams.update({"font.size": 16})
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({"font.size": 16})

    plt.hist(gamma["output"], bins=100, label="Gammas", color="purple", histtype="step")
    plt.hist(proton["output"], bins=100, label="Protons", color="red", histtype="step")
    plt.legend()
    plt.xlabel("Moldel Output")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.title("Model prediction for (not weighted) test data")
    plt.savefig(f"{args.outputDir}/plots/model_output.png", bbox_inches="tight")
    plt.close()
    return


def plot_overTrashold(args, gamma, proton):
    print("Plotting over trashold")
    trasholds = [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.99,
        0.995,
        0.997,
        0.998,
        0.9985,
        0.999,
    ]

    # Plot number of events vs. energy for gammas and protons
    plt.rcParams.update({"font.size": 16})
    # Create a square with multiple subplots depending on the len of trasholds
    fig, axs = plt.subplots(
        int(np.ceil(len(trasholds) / 3)),
        3,
        figsize=(30, 24),
    )
    plt.rcParams.update({"font.size": 16})

    for i, trashold in enumerate(trasholds):
        gamma_trashold = len(gamma["energy"][gamma["output"] > trashold])
        gamma_fraction = (
            len(gamma["energy"][gamma["output"] > trashold])
            / len(gamma["energy"])
            * 100
        )
        print(
            f"Total Gammas > {trashold}: {gamma_trashold}. Fraction: {gamma_fraction:.2f}%"
        )
        proton_trashold = len(proton["energy"][proton["output"] > trashold])
        proton_fraction = (
            len(proton["energy"][proton["output"] > trashold])
            / len(proton["energy"])
            * 100
        )
        print(
            f"Total Protons > {trashold}: {proton_trashold}. Fraction: {proton_fraction:.2f}%"
        )

        axs[i // 3, i % 3].hist(
            gamma["energy"][gamma["output"] > trashold],
            bins=np.arange(4.0, 7.1, 0.1),
            histtype="step",
            label=f"Gammas {gamma_trashold}, {gamma_fraction:.2f}%",
            weights=gamma["weights"][gamma["output"] > trashold],
        )
        axs[i // 3, i % 3].hist(
            proton["energy"][proton["output"] > trashold],
            bins=np.arange(4.0, 7.1, 0.1),
            histtype="step",
            label=f"Protons {proton_trashold}, {proton_fraction:.2f}%",
            weights=proton["weights"][proton["output"] > trashold],
        )
        axs[i // 3, i % 3].set_yscale("log")
        axs[i // 3, i % 3].set_xlabel("log$_{10}$(Energy / GeV)")
        axs[i // 3, i % 3].set_ylabel("Weighted events")
        axs[i // 3, i % 3].set_title(f"Model output > {trashold}")
        axs[i // 3, i % 3].legend()

    plt.tight_layout()
    plt.savefig(f"{args.outputDir}/plots/overTrashold.png", bbox_inches="tight")
    plt.close()

    # Redo the exact same plot but with s125 instead of energy
    plt.rcParams.update({"font.size": 16})
    # Create a square with multiple subplots depending on the len of trasholds
    fig, axs = plt.subplots(
        int(np.ceil(len(trasholds) / 3)),
        3,
        figsize=(30, 24),
    )
    plt.rcParams.update({"font.size": 16})

    for i, trashold in enumerate(trasholds):
        gamma_trashold = len(gamma["s125"][gamma["output"] > trashold])
        gamma_fraction = (
            len(gamma["s125"][gamma["output"] > trashold]) / len(gamma["s125"]) * 100
        )
        print(
            f"Total Gammas > {trashold}: {gamma_trashold}. Fraction: {gamma_fraction:.2f}%"
        )
        proton_trashold = len(proton["s125"][proton["output"] > trashold])
        proton_fraction = (
            len(proton["s125"][proton["output"] > trashold]) / len(proton["s125"]) * 100
        )
        print(
            f"Total Protons > {trashold}: {proton_trashold}. Fraction: {proton_fraction:.2f}%"
        )

        axs[i // 3, i % 3].hist(
            gamma["s125"][gamma["output"] > trashold],
            bins=np.arange(-2.5, 1.5, 0.1),
            histtype="step",
            label=f"Gammas {gamma_trashold}, {gamma_fraction:.2f}%",
            weights=gamma["weights"][gamma["output"] > trashold],
        )
        axs[i // 3, i % 3].hist(
            proton["s125"][proton["output"] > trashold],
            bins=np.arange(-2.5, 1.5, 0.1),
            histtype="step",
            label=f"Protons {proton_trashold}, {proton_fraction:.2f}%",
            weights=proton["weights"][proton["output"] > trashold],
        )
        axs[i // 3, i % 3].set_yscale("log")
        axs[i // 3, i % 3].set_xlabel("log$_{10}$(S$_{125}$ / VEM)")
        axs[i // 3, i % 3].set_ylabel("Weighted events")
        axs[i // 3, i % 3].set_title(f"Model output > {trashold}")
        axs[i // 3, i % 3].legend()

    plt.tight_layout()
    plt.savefig(f"{args.outputDir}/plots/overTrasholdS125.png", bbox_inches="tight")
    plt.close()


def plot_energyS125(args, gamma, trashold=0.999):
    print("Plotting energy vs. S125")
    mask = gamma["output"] > trashold
    plt.rcParams.update({"font.size": 16})
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({"font.size": 16})
    plt.hist2d(
        gamma["s125"][mask],
        gamma["energy"][mask],
        bins=[
            np.arange(-0.5, 1.2, 0.1),
            np.arange(5.5, 7.1, 0.1),
        ],
        weights=gamma["weights"][mask],
        norm=mpl.colors.LogNorm(),
    )
    plt.xlabel("log$_{10}$(S$_{125}$ / VEM)")
    plt.ylabel("log$_{10}$(Energy / GeV)")
    plt.title(f"Energy vs. S$_{{125}}$ for Gammas > {trashold}")
    plt.colorbar()
    plt.savefig(f"{args.outputDir}/plots/energyS125.png", bbox_inches="tight")
    plt.close()


def plot_angularResolution(args, gamma, trashold=0.999):
    print("Plotting angular resolution")
    mask = gamma["output"] > trashold
    plt.rcParams.update({"font.size": 16})
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({"font.size": 16})
    plt.hist2d(
        gamma["energy"][mask],
        gamma["angularResolution"][mask],
        bins=[
            np.arange(5.5, 7.1, 0.1),
            solidAngleBinning(inDegrees=True),
        ],
        norm=mpl.colors.LogNorm(),
        weights=gamma["weights"][mask],
    )
    plt.xlabel("log$_{10}$(Energy / GeV)")
    plt.ylabel("Angular Resolution / deg")
    plt.title(f"Angular Resolution for Gammas > {trashold}")
    plt.colorbar()
    plt.savefig(f"{args.outputDir}/plots/angularResolution.png", bbox_inches="tight")
    plt.close()


################ Getting model and data #############################
def load_model(args):
    print("Loading model")
    # list all the models and get the last one
    models_paths = glob.glob(f"{args.modelDir}/model/model*.pth")
    models_paths.sort()
    model_path = models_paths[-2]
    print(f"Loading model: {model_path}")
    loaded_state_dict = torch.load(model_path)
    net = pickle.load(open(f"{args.modelDir}/model/classModel.pkl", "rb"))
    net.load_state_dict(loaded_state_dict)

    # Define your loss function
    variablesModel = pickle.load(
        open(f"{args.modelDir}/model/variablesModel.pkl", "rb")
    )
    criterion = variablesModel["criterion"]
    optimizer = variablesModel["optimizer"]
    scheduler = variablesModel["scheduler"]

    trainer = TrainTestClass(
        args=args,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loadData=False,
    )
    return trainer


def get_dicts(args, trainer):
    print("Loading data")
    # Load the test dataset with both gamma and protons
    df = load_data(args, is_train=False)
    tensorDict = make_input_tensors(df)

    # Get the gamma and the protons masks
    gamma_mask = df["output"] == 1.0
    proton_mask = df["output"] == 0.0

    # Test the entire test dataset in a single pass without shuffling or batching
    test_loader = data.DataLoader(
        dataset=data.TensorDataset(
            tensorDict["MapHLCq"],
            tensorDict["MapHLCt"],
            tensorDict["fccInput"],
            tensorDict["output"],
            tensorDict["weights"],
        ),
        # Set batch size to the size of the entire test dataset
        batch_size=tensorDict["MapHLCq"].size(0),
        shuffle=False,  # No shuffling
    )

    # Make the test predictions
    print("Running test predictions")
    test_loss, test_accuracy, test_output = trainer.test(test_loader)

    # Test_output to numpy array 1d
    test_output = test_output.numpy().reshape(-1)

    gamma = {
        "output": test_output[gamma_mask],
        "energy": np.log10(df["MCPrimary_energy"][gamma_mask]),
        "weights": df["weights"].values[gamma_mask],
        "s125": df["Laputop3s3s_Log10_S125"].values[gamma_mask],
        "angularResolution": calculate_openingAngle(df, "Laputop3s3s")[gamma_mask],
    }

    proton = {
        "output": test_output[proton_mask],
        "energy": np.log10(df["MCPrimary_energy"][proton_mask]),
        "weights": df["weights"].values[proton_mask],
        "s125": df["Laputop3s3s_Log10_S125"].values[proton_mask],
        "angularResolution": calculate_openingAngle(df, "Laputop3s3s")[proton_mask],
    }

    return gamma, proton


def main(args):
    trainer = load_model(args)
    gamma, proton = get_dicts(args, trainer)
    # plot_output(args, gamma, proton)
    plot_overTrashold(args, gamma, proton)
    trashold = 0.995
    plot_energyS125(args, gamma, trashold)
    plot_angularResolution(args, gamma, trashold)


if __name__ == "__main__":
    main(args=get_args())
    print("-------------------- Program finished --------------------")
