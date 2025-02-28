#! /usr/bin/env python3

import multiprocessing
import os
import glob
import pickle
import argparse

import numpy as np
import pandas as pd
import sys
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data


from icecube import astro
from icecube import icetray, dataio, dataclasses, simclasses
from icecube.recclasses import I3LaputopParams, LaputopParameter


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
    return parser.parse_args()


def load_model(args):
    """
    Load the model, criterion, optimizer and scheduler from the last epoch
    """

    # list all the models and get the last one
    models_paths = glob.glob(f"{args.modelDir}/model/model*.pth")
    models_paths.sort()
    model_path = models_paths[-1]
    print(f"\nLoading model: {model_path}")
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


def quality_cut_ok(df, reco="Laputop3s3s"):
    x = df[f"{reco}_x"]
    y = df[f"{reco}_y"]
    radius = np.sqrt(x**2 + y**2)
    zenith = df[f"{reco}_zenith"]

    beta = df[f"{reco}_beta"]
    fit_status = df[f"{reco}_fit_status"]

    mask = ~(
        (zenith > np.deg2rad(38.0))
        & (radius > 500)
        & (beta < 1.9)
        & (beta > 4.5)
        & (fit_status != "OK")
    )
    return mask


def caluclate_rightAscension_declination(azimuth, zenith, time):
    print("Calculating right ascension and declination")

    ra, dec = astro.dir_to_equa(
        azimuth=azimuth,
        zenith=zenith,
        mjd=time,
    )

    return ra, dec


def get_dict(trainer, fileName):
    """
    Create a dictionary with the data and the predictions
    It can be used to make plots
    """

    df = load_data(fileName)
    print("Making input tensors")
    tensorDict = make_input_tensors(df, weight_index=0)

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

    print("Running test predictions")
    test_loss, test_accuracy, test_output = trainer.test(test_loader)

    # Test_output to numpy array 1d
    test_output = test_output.numpy().reshape(-1)

    # Copy the dataframe and add the predictions,
    # reurn df and after the selection,
    # add right ascension and declination
    df = df.copy()
    df["output"] = test_output

    return df


def process_file(args, file, trainer, trashold):
    """
    This function is called by the multiprocessing.Process
    It runs the predictions and saves the dataframe
    writes the stdout and stderr to separate files
    """
    # Create the output directory
    pathlib.Path(f"{args.outputDir}/logs/").mkdir(parents=True, exist_ok=True)

    # Redirect stdout and stderr to separate files
    sys.stdout = open(
        f"{args.outputDir}/logs/{os.path.basename(file).replace('.hdf5', '.out')}", "w"
    )
    sys.stderr = open(
        f"{args.outputDir}/logs/{os.path.basename(file).replace('.hdf5', '.err')}", "w"
    )

    df = get_dict(trainer, file)

    mask = df["output"] > trashold
    events = len(mask)

    df = df[mask].reset_index(drop=True)

    # If there are some events above the trashold
    if mask.sum() > 0:
        # Calculate the right ascension and declination
        ra, dec = caluclate_rightAscension_declination(
            azimuth=df["Laputop3s3s_azimuth"].values,
            zenith=df["Laputop3s3s_zenith"].values,
            time=df["time_mjd_sec"].values,
        )
        df["ra"] = ra
        df["dec"] = dec

        # Save the dataframe
        df.to_hdf(
            f"{args.outputDir}/gamma_candidates/{os.path.basename(file)}",
            key="df",
            mode="w",
        )

    print(f"Total number of events above {trashold}: {mask.sum()} of {events}")
    print(f"Total ratio of events above {trashold}: {mask.sum()/events}")

    return events, mask.sum()


def process_file_inIce(args, file, trashold):
    df = pd.read_hdf(file, key="data")

    maskQualityCuts = quality_cut_ok(df)
    maskS125 = df["Laputop3s3s_Log10_S125"] < 1.5
    maskInIceCharge = df["Laputop3s3sCleanInIcePulses"] < trashold
    maskContained = df["Laputop3s3s_inice_FractionContainment"] < 0.9
    mask = maskQualityCuts & maskS125  # & maskInIceCharge & maskContained

    events = len(mask)

    df = df[mask].reset_index(drop=True)

    # If there are some events above the trashold
    if mask.sum() > 0:
        # Calculate the right ascension and declination
        ra, dec = caluclate_rightAscension_declination(
            azimuth=df["Laputop3s3s_azimuth"].values,
            zenith=df["Laputop3s3s_zenith"].values,
            time=df["time_mjd_sec"].values,
        )
        df["ra"] = ra
        df["dec"] = dec

        # Save the dataframe
        df.to_hdf(
            f"{args.outputDir}/gamma_candidates/{os.path.basename(file)}",
            key="df",
            mode="w",
        )

    print(f"Total number of events above {trashold}: {mask.sum()} of {events}")
    print(f"Total ratio of events above {trashold}: {mask.sum()/events}")

    return events, mask.sum()


def mainNN(args):
    trainer = load_model(args)

    fileList = sorted(glob.glob(f"{args.inputDir}/*.hdf5"))
    trashold = 0.5

    processes = []

    for file in fileList:
        process = multiprocessing.Process(
            target=process_file, args=(args, file, trainer, trashold)
        )
        processes.append(process)

    for process in processes:
        process.start()
    for process in processes:
        process.join()


def mainInIce(args):
    fileList = sorted(glob.glob(f"{args.inputDir}/*.hdf5"))
    trashold = 0.1

    processes = []

    for file in fileList:
        process = multiprocessing.Process(
            target=process_file_inIce, args=(args, file, trashold)
        )
        processes.append(process)

    for process in processes:
        process.start()
    for process in processes:
        process.join()


if __name__ == "__main__":
    # mainNN(args=get_args())
    mainInIce(args=get_args())
    print("-------------------- Program finished --------------------")
