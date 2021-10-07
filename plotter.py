import argparse
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas
import seaborn as sns

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
# sns.set_palette("tab10")
# Plot settings
parser = argparse.ArgumentParser(description="Neural Networks data plotter")
parser.add_argument(
    "-f", "--files", nargs="*", type=str, default="./log.csv", help="exp directory"
)
parser.add_argument(
    "--inter", action="store_true", default=False, help="psnr interpolation"
)

args = parser.parse_args()

fig = plt.figure("Training")
ax1 = fig.add_subplot(141)
ax3 = fig.add_subplot(142)
ax5 = fig.add_subplot(143)
ax7 = fig.add_subplot(144)

fig2 = plt.figure("Validation")
ax2 = fig2.add_subplot(141)
ax4 = fig2.add_subplot(142)
ax6 = fig2.add_subplot(143)
ax8 = fig2.add_subplot(144)


for file in args.files:

    expid = os.path.dirname(file).split("/")[-1]

    if os.path.exists(file + "/fold0/log.csv"):

        data_f0 = pandas.read_csv(
            file + "/fold0/log.csv", names=["epoch", "T_LOSS", "V_LOSS"]
        )

        # data_f0['T_LOSS'] = 180 * data_f0['T_LOSS'] / math.pi
        # data_f0['V_LOSS'] = 180 * data_f0['V_LOSS'] / math.pi

        sns.lineplot(
            x=data_f0["epoch"],
            y=data_f0["T_LOSS"],
            ax=ax1,
            linewidth=1,
            label=expid,
            alpha=0.7,
        )
        sns.scatterplot(
            x=[data_f0["T_LOSS"].idxmin()],
            y=[data_f0["T_LOSS"].min()],
            ax=ax1,
            label=data_f0["T_LOSS"].min(),
        )

        sns.lineplot(
            x=data_f0["epoch"],
            y=data_f0["V_LOSS"],
            ax=ax2,
            linewidth=1,
            label=expid,
            alpha=0.7,
        )
        sns.scatterplot(
            x=[data_f0["V_LOSS"].idxmin()],
            y=[data_f0["V_LOSS"].min()],
            ax=ax2,
            label=data_f0["V_LOSS"].min(),
        )

    if os.path.exists(file + "/fold1/log.csv"):

        data_f1 = pandas.read_csv(
            file + "/fold1/log.csv", names=["epoch", "T_LOSS", "V_LOSS"]
        )

        # data_f1['T_LOSS'] = 180 * data_f1['T_LOSS'] / math.pi
        # data_f1['V_LOSS'] = 180 * data_f1['V_LOSS'] / math.pi

        sns.lineplot(
            x=data_f1["epoch"],
            y=data_f1["T_LOSS"],
            ax=ax3,
            linewidth=1,
            label=expid,
            alpha=0.7,
        )
        sns.scatterplot(
            x=[data_f1["T_LOSS"].idxmin()],
            y=[data_f1["T_LOSS"].min()],
            ax=ax3,
            label=data_f1["T_LOSS"].min(),
        )

        sns.lineplot(
            x=data_f1["epoch"],
            y=data_f1["V_LOSS"],
            ax=ax4,
            linewidth=1,
            label=expid,
            alpha=0.7,
        )
        sns.scatterplot(
            x=[data_f1["V_LOSS"].idxmin()],
            y=[data_f1["V_LOSS"].min()],
            ax=ax4,
            label=data_f1["V_LOSS"].min(),
        )

    if os.path.exists(file + "/fold2/log.csv"):

        data_f2 = pandas.read_csv(
            file + "/fold2/log.csv", names=["epoch", "T_LOSS", "V_LOSS"]
        )

        # data_f2['T_LOSS'] = 180 * data_f2['T_LOSS'] / math.pi
        # data_f2['V_LOSS'] = 180 * data_f2['V_LOSS'] / math.pi

        sns.lineplot(
            x=data_f2["epoch"],
            y=data_f2["T_LOSS"],
            ax=ax5,
            linewidth=1,
            label=expid,
            alpha=0.7,
        )
        sns.scatterplot(
            x=[data_f2["T_LOSS"].idxmin()],
            y=[data_f2["T_LOSS"].min()],
            ax=ax5,
            label=data_f2["T_LOSS"].min(),
        )

        sns.lineplot(
            x=data_f2["epoch"],
            y=data_f2["V_LOSS"],
            ax=ax6,
            linewidth=1,
            label=expid,
            alpha=0.7,
        )
        sns.scatterplot(
            x=[data_f2["V_LOSS"].idxmin()],
            y=[data_f2["V_LOSS"].min()],
            ax=ax6,
            label=data_f2["V_LOSS"].min(),
        )

    if (
        os.path.exists(file + "/fold0/log.csv")
        and os.path.exists(file + "/fold1/log.csv")
        and os.path.exists(file + "/fold2/log.csv")
    ):

        data_tot = data_f0.copy()
        data_tot["T_LOSS"] = (
            data_f0["T_LOSS"] + data_f1["T_LOSS"] + data_f2["T_LOSS"]
        ) / 3
        data_tot["V_LOSS"] = (
            data_f0["V_LOSS"] + data_f1["V_LOSS"] + data_f2["V_LOSS"]
        ) / 3

        sns.lineplot(
            x=data_tot["epoch"],
            y=data_tot["T_LOSS"],
            ax=ax7,
            linewidth=1,
            label=expid,
            alpha=0.7,
        )
        sns.scatterplot(
            x=[data_tot["T_LOSS"].idxmin()],
            y=[data_tot["T_LOSS"].min()],
            ax=ax7,
            label=data_tot["T_LOSS"].min(),
        )

        sns.lineplot(
            x=data_tot["epoch"],
            y=data_tot["V_LOSS"],
            ax=ax8,
            linewidth=1,
            label=expid,
            alpha=0.7,
        )
        sns.scatterplot(
            x=[data_tot["V_LOSS"].idxmin()],
            y=[data_tot["V_LOSS"].min()],
            ax=ax8,
            label=data_tot["V_LOSS"].min(),
        )

plt.show()
