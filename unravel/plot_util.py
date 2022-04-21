import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["axes.edgecolor"] = "#333F4B"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.color"] = "#333F4B"
plt.rcParams["ytick.color"] = "#333F4B"


def plot_scores(feature_names, scores, num_features=13, savefig=False):
    """
    Utility function for plotting the feature importance scores

    Args:
        feature_names (Numpy Array): Array containing feature names
        scores (Numpy Array): Array containing feature importance scores for each feature
        num_features (int, optional): Top number of features to display
        savefig (bool, optional): If True saves the figure using the default name. Defaults to False.
    """
    df = pd.DataFrame(scores, index=feature_names, columns=["Importance Score"])

    df = df.sort_values("Importance Score")[:num_features]
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hlines(
        y=df.index,
        xmin=0,
        xmax=df["Importance Score"],
        color="#FF2052",
        alpha=0.2,
        linewidth=5,
    )
    plt.plot(
        df["Importance Score"],
        df.index,
        "o",
        markersize=5,
        color="#FF2052",
        alpha=0.6,
    )
    ax.set_xlabel("Importance Score", fontsize=15, fontweight="black", color="#333F4B")
    # ax.set_ylabel(df.index)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_bounds((1, len(plot_range)))
    # ax.set_xlim(0, 25)
    # add some space between the axis and the plot
    ax.spines["left"].set_position(("outward", 8))
    ax.spines["bottom"].set_position(("outward", 5))

    # df.plot(kind="barh", legend=False)
    # plt.xlabel("Importance Score")
    # plt.ylabel("Feature Name")

    if savefig:
        plt.savefig("fig.png", dpi=72)
