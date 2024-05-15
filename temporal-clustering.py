from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMeanVariance

filename = Path("/Users/mark/Downloads/inGCS_outlabel_2019_1k.csv")
output_dir = Path("/Users/mark/Downloads/clustering_results_change/")


data = pd.read_csv(filename)

if not output_dir.exists():
    output_dir.mkdir()

# drop nan rows
data = data.dropna()

# get final column as outcome
outcome = np.array(data.values[:, -1]).astype(dtype=np.float32)

time_labels = data.columns[:-1]
time_labels = [int(x.split(" ")[0]) for x in time_labels]

# get into same format as tslearn, removing the outcome column
X_train = np.array(data.values[:, :-1]).astype(dtype=np.float32)

# uncomment to subtract the first time point from each row, to cluster based on change from baseline
# X_train = X_train - X_train[:, 0][:, None]

# for now, no scaling
X_train_scaled = X_train

sz = X_train_scaled.shape[1]

assert len(X_train_scaled) == len(outcome)
print(f"Found {len(X_train_scaled)} samples with {sz} time points each")
# Euclidean k-means
print("Euclidean k-means")
# loop over the number of clusters we want to try
for n_clusters in range(5, 16):
    # fit the k-means
    km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=True, random_state=42)

    # get predictions
    y_pred = km.fit_predict(X_train_scaled)

    plt.figure(figsize=(n_clusters * 3, 6))
    clusters_accounted = 0
    for yi in range(n_clusters):
        plt.subplot(1, n_clusters, yi + 1)
        cluster_outcomes = outcome[y_pred == yi]
        clusters_accounted += cluster_outcomes.sum()
        for idx, xx in enumerate(X_train_scaled[y_pred == yi]):
            if cluster_outcomes[idx] == 1:
                plt.plot(xx.ravel(), "r-", alpha=0.2, linewidth=2)
            else:
                plt.plot(xx.ravel(), "k-", alpha=0.2, linewidth=2)
        plt.plot(km.cluster_centers_[yi].ravel(), "b-", linewidth=4)
        plt.xlim(0, sz)
        plt.ylim(0, 15)

        # set x-axis to time_labels
        plt.xticks(range(sz), time_labels)
        plt.xlabel("Time (hours)")

        # make ticks less dense
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(4))

        plt.title(f"C{yi+1}: {int(cluster_outcomes.sum())}/{len(cluster_outcomes)}")
        if yi == 0:
            plt.ylabel("GCS")

    # # DBA-k-means
    # print("DBA k-means")
    # dba_km = TimeSeriesKMeans(n_clusters=n_clusters,
    #                           n_init=2,
    #                           metric="dtw",
    #                           verbose=True,
    #                           max_iter_barycenter=10,
    #                           random_state=42)
    # y_pred = dba_km.fit_predict(X_train_scaled)
    #
    # for yi in range(n_clusters):
    #     plt.subplot(2, n_clusters, n_clusters + yi + 1)
    #     for xx in X_train_scaled[y_pred == yi]:
    #         plt.plot(xx.ravel(), "k-", alpha=.2)
    #     plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    #     plt.xlim(0, sz)
    #     plt.ylim(0, 15)
    #     plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
    #              transform=plt.gca().transAxes)
    #     if yi == 0:
    #         plt.ylabel("DBA k-means")
    plt.savefig(output_dir / f"{n_clusters}_clusters.png")
