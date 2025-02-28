#! /usr/bin/env python3

import argparse

import glob
import multiprocessing as mp
import pandas as pd

import healpy as hp

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data


from icecube import astro
from icecube import icetray, dataio, dataclasses, simclasses
from icecube.recclasses import I3LaputopParams, LaputopParameter


# from utils.network_model import Net
# from utils.TrainTestClass import TrainTestClass
# from utils.utils_functions import (
#     load_data,
#     make_input_tensors,
# )


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputDir", help="input folder", type=str)
    parser.add_argument("-o", "--outputDir", help="output folder", type=str)
    parser.add_argument("-y", "--year", help="year of the simulation", type=int)

    return parser.parse_args()


def read_dataFrame(file):
    """
    Read the dataFrame from the file
    """
    df = pd.read_hdf(file, key="df")
    # mask = df["output"] > trashold
    # df = df[mask].reset_index(drop=True)
    return df


def load_dataFrame(file):
    print(f"Reading file: {file}")
    df = pd.read_hdf(file, key="df")
    if "I3EventHeader" in df.columns:
        start_time = df["I3EventHeader"][0].start_time
        end_time = df["I3EventHeader"][len(df) - 1].end_time
        total_time = (end_time - start_time) * 1e-9  # from ns to s
    else:
        total_time = 0.0

    df = quality_cut_ok(df)
    if not len(df):
        print(f"Skipping {file}")
        return pd.DataFrame()
    return df, total_time


def get_dataframe_mp(fileList):
    print("Reading files...")
    pool = mp.Pool(10)  # mp.cpu_count() // 2)
    poolMap = pool.map(load_dataFrame, fileList)
    df_list, total_time_list = zip(*poolMap)
    total_time = np.sum(total_time_list)
    days, remainder = divmod(total_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(
        f"Total time: {(int(days))} d {int(hours)} h {int(minutes)} m {int(seconds)} s"
    )

    df = pd.concat(df_list)
    df = df.reset_index(drop=True)
    df = caluclate_rightAscension_declination(df)
    return df, total_time


# def quality_cut_ok(frame, reco="Laputop3s3s"):
#     x = frame[reco].pos.x
#     y = frame[reco].pos.y
#     radius = np.sqrt(x**2 + y**2)
#     if radius > 500:
#         return False
#     zenith = frame[reco].dir.zenith
#     if zenith > np.deg2rad(38.0):
#         return False
#     beta = frame[f"{reco}Params"].value(LaputopParameter.Beta)
#     if beta < 1.9 or beta > 4.5:
#         return False
#     fit_status = frame[reco].fit_status_string
#     if fit_status != "OK":
#         return False
#     return True


def quality_cut_ok(df, reco="Laputop3s3s"):
    x = df[f"{reco}_x"]
    y = df[f"{reco}_y"]
    radius = np.sqrt(x**2 + y**2)
    maskRadius = radius < 500.0

    zenith = df[f"{reco}_zenith"]
    maskZenith = zenith < np.deg2rad(38.0)
    beta = df[f"{reco}_beta"]
    maskBeta = (beta > 1.9) & (beta < 4.5)

    fit_status = df[f"{reco}_fit_status"]
    maskFit_status = fit_status == "OK"

    df = df[maskRadius & maskZenith & maskBeta & maskFit_status]
    df = df.reset_index(drop=True)
    return df


def get_fractionInIceChargeTheta(fileList, pulseKey="Laputop3s3sCleanInIcePulses"):
    primaryDict = {
        "containmentList": [],
        "inIceChargeList": [],
        "thetaList": [],
        "log10S125List": [],
    }
    for i, fileName in enumerate(fileList, start=1):
        print(f"Processing file {i} of {len(fileList)}: {fileName}")
        for frame in dataio.I3File(fileName):
            if frame.Stop != icetray.I3Frame.Physics:
                continue
            if not quality_cut_ok(frame, reco="Laputop3s3s"):
                continue
            primaryDict["containmentList"].append(
                frame["Laputop3s3s_inice_FractionContainment"]
            )
            primaryDict["thetaList"].append(frame["Laputop3s3s"].dir.zenith)
            primaryDict["log10S125List"].append(
                frame["Laputop3s3sParams"].value(LaputopParameter.Log10_S125)
            )

            totPulse = 0.0
            pulseMap = frame[pulseKey].apply(frame)
            for omkey in pulseMap.keys():
                for pulse in pulseMap[omkey]:
                    totPulse += pulse.charge
            primaryDict["inIceChargeList"].append(totPulse)
    return primaryDict


def get_gammaDict():
    from icecube import icetray, dataio, dataclasses, simclasses
    from icecube.recclasses import I3LaputopParams, LaputopParameter

    print("Reading gamma files")

    filesGamma = glob.glob(
        "/hkfs/work/workspace/scratch/rn8463-gamma-detectorResponse/sibyll2.1/*"
    )
    gammaDict = get_fractionInIceChargeTheta(fileList=sorted(filesGamma))
    print("Finished reading gamma files")
    gammaDict = {key: np.array(value) for key, value in gammaDict.items()}
    return gammaDict


def caluclate_rightAscension_declination(df):
    print("Calculating right ascension and declination")

    azimuth = df["Laputop3s3s_azimuth"].values
    zenith = df["Laputop3s3s_zenith"].values
    time = df["time_mjd_sec"].values

    ra, dec = astro.dir_to_equa(
        azimuth=azimuth,
        zenith=zenith,
        mjd=time,
    )

    df["ra"] = ra
    df["dec"] = dec

    return df


class PolarSkyMap:
    """Class for plotting sky maps using Basemap projected from the Pole.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        matplotlib figure instance
    axes : matplotlib.axes._subplots.AxesSubplot
        matplotlib axes instance

    Attributes
    ----------
    basemap : mpl_toolkits.basemap.Basemap
        Basemap instance set to a South Pole Projection

    """

    def __init__(
        self,
    ):
        plt.rcParams.update({"font.size": 16})
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 8))
        plt.rcParams.update({"font.size": 16})
        self.fig = fig
        self.ax = ax

    def plot_grid(self):
        """Plot grid lines."""

        for dec in [-90, -80, -70, -60, -50]:
            x, y = np.deg2rad([45, dec])
            self.ax.text(x, y, "%s$^{\circ}$" % dec, fontsize=16)

    def plot_prelim(self, x=3 / 4 * np.pi, y=-0.6):
        """Draw a label for unpublished plots."""
        self.ax.text(
            x,
            y,
            "IceCube Preliminary",
            color="r",
            fontsize=14,
        )

    def plot_galactic_plane(self):
        """Plot the galactic plance region"""

        cRot = hp.Rotator(coord=["G", "C"], rot=[0, 0])
        tl = np.radians(np.arange(0, 360, 0.01))
        tb = np.radians(np.full(tl.size, 90))
        tdec, tra = cRot(tb, tl)
        x, y = (tra, np.pi / 2 - tdec)
        sc = self.ax.plot(x, y, "k--", linewidth=1, label="Galactic Plane")

        tb = np.radians(np.full(tl.size, 95))
        tdec, tra = cRot(tb, tl)
        x, y = (tra, np.pi / 2 - tdec)
        sc = self.ax.plot(x, y, "k-", linewidth=1)

        tb = np.radians(np.full(tl.size, 85))
        tdec, tra = cRot(tb, tl)
        x, y = (tra, np.pi / 2 - tdec)
        sc = self.ax.plot(x, y, "k-", linewidth=1)

    def plot_sky_map(self, df):
        """
        Plot the sky map of the gamma candidates
        """
        # Set the colormap
        # cmap = mpl.colormaps.get_cmap("viridis")
        cmap = mpl.colormaps.get_cmap("rainbow")

        # Plot the sky map

        self.ax.scatter(
            df["ra"],
            df["dec"],
            c=df["Laputop3s3s_Log10_S125"],
            # c=df["output"],
            cmap=cmap,
            s=1,
            # alpha=0.5,
        )
        self.fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=-2.5, vmax=1.5),
                # norm=mpl.colors.Normalize(vmin=0.95, vmax=1.0),
                cmap=cmap,
            ),
            ax=self.ax,
            label="log$_{10}$(S$_{125}$)",
            # label="Output",
        )
        self.ax.set_title(f"Sky map of the gamma candidates")
        self.ax.set_yticks(
            np.deg2rad(
                [
                    -90,
                    -80,
                    -70,
                    -60,
                    -50,
                ]
            )
        )
        self.ax.set_yticklabels(
            [
                "-90$^{\circ}$",
                "-80$^{\circ}$",
                "-70$^{\circ}$",
                "-60$^{\circ}$",
                "-50$^{\circ}$",
            ]
        )

        self.ax.set_ylim(np.deg2rad(-90), np.deg2rad(-50.0))

    def plot_sky_map_hist(self, df):
        """
        Plot the sky map of the gamma candidates
        """
        ra = df["ra"].values
        dec = df["dec"].values

        binsPhi = np.linspace(0, 2 * np.pi, int(3600 / 4 + 1))
        binsTheta = solidAngleBinning(
            theta_start=0.0,
            theta_end=np.deg2rad(38.0),
            numbOfBins=int(380 / 4 + 1),
            inDegrees=False,
        ) - (np.pi / 2.0)

        hist, _, _ = np.histogram2d(
            dec,
            ra,
            bins=[binsTheta, binsPhi],
        )
        # maxVal = np.max(hist)
        # whereMax = np.where(hist == maxVal)
        # azimuth = binsPhi[whereMax[1][0]]
        # zenith = binsTheta[whereMax[0][0]] + (np.pi / 2.0)
        hist = np.ma.masked_where(hist == 0, hist)

        self.ax.set_title(f"Sky map histogram of the 1% of Data")
        self.ax.set_yticks(
            np.deg2rad(
                [
                    -90,
                    -80,
                    -70,
                    -60,
                    -50,
                ]
            )
        )
        self.ax.set_yticklabels(
            [
                "-90$^{\circ}$",
                "-80$^{\circ}$",
                "-70$^{\circ}$",
                "-60$^{\circ}$",
                "-50$^{\circ}$",
            ]
        )

        self.ax.set_ylim(np.deg2rad(-90), np.deg2rad(-50.0))
        self.ax.grid(True)

        cmap = mpl.colormaps.get_cmap("rainbow")
        vmin = np.floor(hist.min())
        vmax = np.ceil(hist.max())

        sc = self.ax.pcolormesh(
            binsPhi,
            binsTheta,
            hist,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        self.fig.colorbar(
            sc,
            ax=self.ax,
            label="Counts",
        )


def read_outFile(outputDir):
    """
    Read the output file
    """
    events = 0

    for file in glob.glob(f"{outputDir}/logs/Level3*.out"):
        with open(file, "r") as f:
            for line in f:
                if "Total number of events above" in line:
                    # Get the numbers
                    numbers = line.split(":")[-1].split("of")

                    events += int(numbers[1])

    return events


def plot_sky_map(df, args):
    """
    Plot the sky map of the gamma candidates
    """
    skymap = PolarSkyMap()
    skymap.plot_galactic_plane()
    skymap.plot_prelim()
    skymap.plot_sky_map(df)

    plt.tight_layout()
    plt.savefig(f"{args.outputDir}/plots/sky_map.png")
    print(f"Saved plot: {args.outputDir}/plots/sky_map.png")
    plt.close()
    # events = read_outFile(args.outputDir)

    # gamma_candidates = len(df)
    # print(
    #     f"Total number of events above {trashold}: {gamma_candidates:,} of {events:,}"
    # )
    # print(f"Total ratio of events above {trashold}: {gamma_candidates/events:.2e}")


def plot_s125Zenith(df, args):
    """
    Plot the zenith angle of the gamma candidates
    """
    addGamma = True
    if addGamma:
        gammaDict = get_gammaDict()

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(4, 5, figsize=(20, 16))
    plt.rcParams.update({"font.size": 12})
    s125Steps = np.linspace(-2.5, 1.5, int(4.0 / 0.2) + 1)
    bins = solidAngleBinning(
        theta_start=0,
        theta_end=np.deg2rad(38.0),
        numbOfBins=21,
        inDegrees=True,
    )
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for i, (s125min, s125max) in enumerate(zip(s125Steps[:-1], s125Steps[1:])):
        maskS125 = (df["Laputop3s3s_Log10_S125"] > s125min) & (
            df["Laputop3s3s_Log10_S125"] < s125max
        )
        dfS125 = df[maskS125]

        maskInIceCharge = dfS125["Laputop3s3sCleanInIcePulses"] < 1.0
        maskContained = dfS125["Laputop3s3s_inice_FractionContainment"] < 0.9

        zenith = np.rad2deg(dfS125["Laputop3s3s_zenith"])
        zenithSelected = np.rad2deg(
            dfS125[maskInIceCharge & maskContained]["Laputop3s3s_zenith"]
        )

        histBase, _ = np.histogram(zenith, bins=bins)
        histSelected, _ = np.histogram(zenithSelected, bins=bins)

        ax[i // 5, i % 5].plot(
            bin_centers,
            histSelected / histBase,
            label="Data",
            c="k",
        )
        if addGamma:
            plot_Gamma(
                gammaDict, ax[i // 5, i % 5], s125min, s125max, bins, bin_centers
            )
        ax[i // 5, i % 5].set_xlabel("Zenith angle [deg]")
        ax[i // 5, i % 5].set_ylabel("Number of events")
        ax[i // 5, i % 5].set_title(
            f"{s125min:.1f} < log$_{{10}}$(S$_{{125}}$ / VEM) < {s125max:.1f}"
        )
        ax[i // 5, i % 5].legend()
        ax[i // 5, i % 5].set_ylim(1e-6, 1.0)
        ax[i // 5, i % 5].set_yscale("log")

    plt.tight_layout()
    plt.savefig(f"{args.outputDir}/plots/s125_zenith.png")
    print(f"Saved plot: {args.outputDir}/plots/s125_zenith.png")
    plt.close()


def plot_Gamma(gammaDict, ax, s125min, s125max, bins, bin_centers):
    maskS125 = (gammaDict["log10S125List"] > s125min) & (
        gammaDict["log10S125List"] < s125max
    )

    maskInIceCharge = gammaDict["inIceChargeList"] < 1.0
    maskContained = gammaDict["containmentList"] < 0.9

    zenith = np.rad2deg(gammaDict["thetaList"])[maskS125]
    zenithSelected = np.rad2deg(
        np.array(gammaDict["thetaList"])[maskS125 & maskInIceCharge & maskContained]
    )

    histBase, _ = np.histogram(zenith, bins=bins)
    histSelected, _ = np.histogram(zenithSelected, bins=bins)

    ax.plot(
        bin_centers,
        histSelected / histBase,
        "--",
        label="Gamma",
        c="magenta",
    )


def plot_skymapHist(df):
    skymap = PolarSkyMap()
    skymap.plot_galactic_plane()
    skymap.plot_prelim()
    skymap.plot_sky_map_hist(df)

    plt.tight_layout()
    plt.savefig(f"/home/hk-project-pevradio/rn8463/gamma_hadron/plots/skymapHist.png")
    print(
        f"Saved plot: /home/hk-project-pevradio/rn8463/gamma_hadron/plots/skymapHist.png"
    )
    plt.close()


def main(args):
    fileList = sorted(glob.glob(f"{args.inputDir}/*.hdf5"))

    df, _ = get_dataframe_mp(fileList)

    # #### Initialize an empty pandas dataframe to store the results
    #  trashold = 0.999
    # df = pd.DataFrame()

    # for i, file in enumerate(fileList, 1):
    #     print(f"{i:7d} / {len(fileList)} Reading file: {file}")
    #     # Concatenate the dataframes
    #     df = pd.concat(
    #         [df, read_dataFrame(file, trashold)],
    #         ignore_index=True,
    #     )

    # if len(df) != 0:
    #     for event in df["I3EventHeader"]:
    #         print(
    #             event.run_id,
    #             event.event_id,
    #             event.sub_event_id,
    #             event.sub_event_stream,
    #         )
    #     print(df["Laputop3s3s_Log10_S125"].values)

    # Plot the sky map
    # plot_sky_map(df, args)
    # plot_s125Zenith(df, args)

    plot_skymapHist(df)


if __name__ == "__main__":
    main(args=get_args())
    print("-------------------- Program finished --------------------")
