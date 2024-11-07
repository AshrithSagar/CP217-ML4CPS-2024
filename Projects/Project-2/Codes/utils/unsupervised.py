"""
unsupervised.py
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def kmeans_clustering(
    data: pd.DataFrame, n_clusters: int, random_state=42
) -> pd.DataFrame:
    """
    Perform K-means clustering on the data.

    Parameters:
    - data: The data to cluster.
    - n_clusters: The number of clusters to form.

    Returns:
    - A DataFrame with the cluster labels added.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(data)
    return clusters


def pca_dimensionality_reduction(
    data: pd.DataFrame, n_components: int, random_state=42
) -> pd.DataFrame:
    """
    Perform PCA dimensionality reduction on the data.

    Parameters:
    - data: The data to reduce dimensionality.
    - n_components: The number of components to keep.

    Returns:
    - A DataFrame with the reduced data.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_data = pca.fit_transform(data)
    return reduced_data


def mds_dimensionality_reduction(
    data: pd.DataFrame, n_components: int, random_state=42
) -> pd.DataFrame:
    """
    Perform MDS dimensionality reduction on the data.

    Parameters:
    - data: The data to reduce dimensionality.
    - n_components: The number of components to keep.

    Returns:
    - A DataFrame with the reduced data.
    """
    from sklearn.manifold import MDS

    mds = MDS(n_components=n_components, random_state=random_state)
    reduced_data = mds.fit_transform(data)
    return reduced_data


def tsne_dimensionality_reduction(
    data: pd.DataFrame, n_components: int, random_state=42
) -> pd.DataFrame:
    """
    Perform t-SNE dimensionality reduction on the data.

    Parameters:
    - data: The data to reduce dimensionality.
    - n_components: The number of components to keep.

    Returns:
    - A DataFrame with the reduced data.
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=n_components, random_state=random_state)
    reduced_data = tsne.fit_transform(data)
    return reduced_data


def plot_clusters_2d(data: pd.DataFrame, clusters: pd.Series, title: str = "") -> None:
    """
    Plot the clusters in 2D using the first two components of the data.

    Parameters:
    - data: The data to plot.
    - clusters: The cluster labels.
    - title: The title of the plot.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=clusters, cmap="viridis")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar()
    plt.show()


def plot_clusters_3d(data: pd.DataFrame, clusters: pd.Series, title: str) -> None:
    """
    Plot the clusters in 3D using the first three components of the data.

    Parameters:
    - data: The data to plot.
    - clusters: The cluster labels.
    - title: The title of the plot.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=clusters, cmap="viridis"
    )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.show()


def evaluate_clustering_performance(data: pd.DataFrame, clusters: pd.Series) -> float:
    """
    Evaluate the performance of the clustering using the silhouette score.

    Parameters:
    - data: The data used for clustering.
    - clusters: The cluster labels.

    Returns:
    - The silhouette score.
    """
    from sklearn.metrics import silhouette_score

    return silhouette_score(data, clusters)


def evaluate_dimensionality_reduction_performance(
    original_data: pd.DataFrame, reduced_data: pd.DataFrame
) -> float:
    """
    Evaluate the performance of dimensionality reduction using the mean squared error.

    Parameters:
    - original_data: The original data.
    - reduced_data: The reduced data.
    Returns:
    - The mean squared error.
    """
    from sklearn.metrics import mean_squared_error

    return mean_squared_error(original_data, reduced_data)
