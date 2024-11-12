import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import rich
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import DistanceMetric, pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.preprocessing import LabelEncoder, StandardScaler
from umap import UMAP


class DataProcessor:
    def __init__(self, df, seed=42, verbose=False):
        self.df_data = df
        self.seed = seed
        self.verbose = verbose
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def get_columns(self):
        columns = {
            "Values": self.df_data.columns[2:],
            "Community": self.df_data.columns[0:2],
            "Geography": self.df_data.columns[2:17],
            "Land_Use": self.df_data.columns[17:27],
            "Population_2012": self.df_data.columns[27:52],
            "Population_2007": self.df_data.columns[52:77],
            "Population_Change_2007_2012": self.df_data.columns[77:90],
            "Services": self.df_data.columns[90:114],
            "Socio_Demographic": self.df_data.columns[114:170],
            "Diversity": self.df_data.columns[170:210],
            "Hospital": self.df_data.columns[210:226],
            "Coordinates": self.df_data.columns[226:],
        }
        return columns

    def list_columns(self):
        columns = self.get_columns()
        for key in columns.keys():
            print(f"{key}")

    def get_verbose(self, verbose):
        if verbose is None:
            return self.verbose
        else:
            return verbose

    def prepare_data(self, columns, verbose=None):
        verbose = self.get_verbose(verbose)
        df_Data = self.df_data[columns]

        for column in columns:
            df_Data.loc[:, column] = pd.to_numeric(df_Data[column], errors="coerce")

        nan_cols = df_Data.columns[df_Data.isna().any()].tolist()
        if verbose:
            if nan_cols:
                print(f"Columns with NaN values: {nan_cols}")
            else:
                print("No NaN values found in the DataFrame.")

        for col in df_Data.columns:
            df_Data.loc[:, col] = self.label_encoder.fit_transform(df_Data[col])

        self.df = self.scaler.fit_transform(df_Data)

    def plot_suburbs_in_map(self):
        df = self.df_data
        center_lat = df["Latitude"].mean()
        center_lon = df["Longitude"].mean()

        fig = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            hover_name=df.iloc[:, 0],
            zoom=8,
            color=df.index,
        )
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_center={"lat": center_lat, "lon": center_lon},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )
        fig.show()


if __name__ == "__main__":

    def data_preperation_distance(df, columns, distance):
        X = df_Data.iloc[:, 1:].values
        dist_matrix = pairwise_distances(X, X, metric=distance)
        return dist_matrix, y

    dist_matrix, y = data_preperation_distance(df, Values, "euclidean")

    def data_analysis_PCA(df):
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
        X_scaled = StandardScaler().fit_transform(X)
        y_df = pd.DataFrame(y, columns=["real"])

        le = LabelEncoder()

        y_df["encoded"] = le.fit_transform(y_df["real"])
        y_array = np.array(y_df["encoded"])

        features = X_scaled.T
        cov_matrix = np.cov(features)
        values, vectors = np.linalg.eig(cov_matrix)
        projected_1 = X_scaled.dot(vectors.T[0])
        projected_2 = X_scaled.dot(vectors.T[1])
        res = pd.DataFrame(projected_1, columns=["PC1"])
        res["PC2"] = projected_2
        res["Y"] = y.flatten()

        plt.figure(figsize=(20, 10))
        sns.scatterplot(data=res, x="PC1", y=[0] * len(res), hue=res["Y"], s=200)

        # lets apply PCA with n_components =4
        pca = PCA(n_components=10)
        pca.fit(X_scaled)

        # lets visualize the explained variance ratio.
        plt.figure(figsize=(20, 10))
        percent_variance = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
        columns = [
            "PC1",
            "PC2",
            "PC3",
            "PC4",
            "PCA5",
            "PCA6",
            "PCA7",
            "PCA8",
            "PCA9",
            "PCA10",
        ]
        plt.bar(x=range(1, 11), height=percent_variance, tick_label=columns)
        plt.ylabel("Percentate of Variance Explained")
        plt.xlabel("Principal Component")
        plt.title("PCA Scree Plot")
        plt.show()

        plt.figure(figsize=(20, 10))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel("number of components")
        plt.ylabel("cumulative explained variance")

        pca = PCA(2)
        projected = pca.fit_transform(X_scaled)
        plt.figure(figsize=(20, 10))
        plt.scatter(
            projected[:, 0],
            projected[:, 1],
            c=y_array,
            edgecolor="none",
            alpha=0.5,
            cmap=plt.cm.get_cmap("Spectral", 10),
        )
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        plt.colorbar()

    def mapData_MDS(dist_matrix, y, metric, title, random_state=0):
        mds = MDS(metric=metric, dissimilarity="precomputed", random_state=random_state)
        # Get the embeddings
        pts = mds.fit_transform(dist_matrix)

        fig = plt.figure(2, (15, 8))
        ax = fig.add_subplot(1, 2, 1)
        ax = sns.scatterplot(x=pts[:, 0], y=pts[:, 1], hue=y)
        plt.title(title)
        # plt.legend(y, title="Communities", loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=9) # Adjust ncol as needed
        plt.tight_layout()

        le = LabelEncoder()
        y_encodings = le.fit_transform(y)
        y_and_encodings = list(zip(y, y_encodings))
        for i in range(len(y)):
            plt.annotate(y_encodings[i], (pts[i, 0], pts[i, 1]))

        legend = ["-".join(map(str, pair)) for pair in y_and_encodings]
        # Create a separate legend with labels
        plt.legend(
            legend,
            title="Communities",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=9,
        )  # Adjust ncol as needed

        # ax.legend(labels=y, title="Communities", loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()

        return pts, y_and_encodings

    def mapData_Manifold(X, y, title=""):
        # Convert y to a categorical type for accurate legend display
        y_categorical = pd.Categorical(y, categories=np.unique(y))

        # Plot the X, colored according to the class labels
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        # Scatter plot with class labels using Seaborn
        # custom_palette = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe'][:len(np.unique(y))]
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_categorical, ax=ax, legend="full")
        ax.set_title(f"Scatter Plot - {title}")
        ax.legend(title="Classes", bbox_to_anchor=(1, 1), loc="lower right")

        le = LabelEncoder()
        y_encodings = le.fit_transform(y)
        y_and_encodings = list(zip(y, y_encodings))
        for i in range(len(y)):
            plt.annotate(y_encodings[i], (X[i, 0], X[i, 1]))

        legend = ["-".join(map(str, pair)) for pair in y_and_encodings]
        plt.legend(
            legend,
            title="Communities",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=9,
        )  # Adjust ncol as needed

        plt.show()

        return X, y_and_encodings

    def data_analysis_MDS(df, distance):

        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
        X_scaled = StandardScaler().fit_transform(X)

        dist_metric = distance(X_scaled)
        mapData_MDS(dist_metric, y, True, "Metric MDS", random_state=0)

        dist_non_metric = euclidean_distances(X_scaled)
        mapData_MDS(dist_non_metric, y, False, "Non Metric MDS", random_state=0)

    def data_analysis_TSNE(df, perplexity=20.0, random_state=5):
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
        X_scaled = StandardScaler().fit_transform(X)

        tsne_suburb = TSNE(perplexity=perplexity, random_state=random_state)
        X_tsne_suburb = tsne_suburb.fit_transform(X)

        return mapData_Manifold(
            X_tsne_suburb, y, title=f"Suburb - t-SNE [perplexity - {perplexity}]"
        )

    def data_analysis_UMAP(df, n_neighbors=5, min_dist=0.1, random_state=5):
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
        X_scaled = StandardScaler().fit_transform(X)

        umap_suburb = UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
        )
        X_umap_suburb = umap_suburb.fit_transform(X)

        return mapData_Manifold(
            X_umap_suburb,
            y,
            title=f"Suburb [min_dist={min_dist}, n_neighbors={n_neighbors}] - UMAP",
        )

    def normalize_matrix(dist_matrix):
        min_val = np.min(dist_matrix)
        max_val = np.max(dist_matrix)
        normalized_matrix = (dist_matrix - min_val) / (max_val - min_val)
        return normalized_matrix

    # run_analysis(df, Hospital, 'sqeuclidean', 1, Hospital, 'minkowski', 0)

    def distance_metric_analysis_MDS(
        df, columns1, distance1, similar1, columns2, distance2, similar2
    ):
        df_Data1 = data_preperation(df, columns1)
        df_Data2 = data_preperation(df, columns2)
        dist_matrix_1, y = data_preperation_distance(df_Data1, columns1, distance1)
        dist_matrix_2, y = data_preperation_distance(df_Data2, columns2, distance2)

        if similar1 == 0:
            dist_matrix1 = -dist_matrix_1

        if similar2 == 0:
            dist_matrix2 = -dist_matrix_2

        dist_matrix_1_normalized = normalize_matrix(dist_matrix_1)
        dist_matrix_2_normalized = normalize_matrix(dist_matrix_2)

        dist_matrix = (dist_matrix_1 + dist_matrix_2) / 2
        dist_matrix_from_normalized = (
            dist_matrix_1_normalized + dist_matrix_2_normalized
        ) / 2

        pts_metric, y_and_encodings_metric = mapData_MDS(
            dist_matrix, y, True, "Metric MDS", random_state=0
        )
        pts_non_metric, y_and_encodings_non_metric = mapData_MDS(
            dist_matrix, y, False, "Non-Metric MDS", random_state=0
        )

        pts_metric_from_norm, y_and_encodings_metric_from_norm = mapData_MDS(
            dist_matrix_from_normalized, y, True, "Metric MDS", random_state=0
        )
        pts_non_metric_from_norm, y_and_encodings_non_metric_from_norm = mapData_MDS(
            dist_matrix_from_normalized, y, False, "Non-Metric MDS", random_state=0
        )

        return (
            pts_metric,
            pts_non_metric,
            pts_metric_from_norm,
            pts_non_metric_from_norm,
            y_and_encodings_metric,
            y_and_encodings_non_metric,
            y_and_encodings_metric_from_norm,
            y_and_encodings_non_metric_from_norm,
        )

    def distance_heatmap(x, y):
        dist_matrix = pairwise_distances(x)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            dist_matrix, annot=False, xticklabels=y, yticklabels=y, cmap="viridis"
        )
        plt.title("Pairwise Distance Heatmap")
        plt.show()

    def run_analysis(
        df, columns1, distance1, isnotsimilar1, columns2, distance2, isnotsimilar2
    ):
        pts1, pts2, pts3, pts4, y1, y2, y3, y4 = distance_metric_analysis_MDS(
            df, columns1, distance1, isnotsimilar1, columns2, distance2, isnotsimilar2
        )
        distance_heatmap(pts1, y1)
        distance_heatmap(pts2, y2)
        distance_heatmap(pts3, y3)
        distance_heatmap(pts4, y4)
        print(
            "================================\nDistance matrix without normalization and  metric mds \n\n"
        )
        run_kmeans(pts1, y1)
        print("=================================\n")

        print(
            "================================\nDistance matrix without normalization and non metric mds\n\n"
        )
        run_kmeans(pts2, y2)
        print("=================================\n")

        print(
            "================================\nDistance matrix with normalization and metric mds\n\n"
        )
        run_kmeans(pts3, y3)
        print("=================================\n")

        print(
            "================================\nDistance matrix without normalization and non metric mds\n\n"
        )
        run_kmeans(pts4, y4)
        print("=================================\n")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans

    df_clean = df.dropna(subset=["Latitude", "Longitude"])
    coordinates = df_clean[["Latitude", "Longitude"]]
    kmeans = KMeans(n_clusters=6, random_state=42)
    df_clean["Cluster"] = kmeans.fit_predict(coordinates)
    plt.figure(figsize=(10, 8))
    plt.scatter(
        df_clean["Longitude"],
        df_clean["Latitude"],
        c=df_clean["Cluster"],
        cmap="viridis",
        marker="o",
        edgecolor="k",
        s=100,
        alpha=0.6,
    )
    plt.title("Clustering of Suburbs Based on Latitude and Longitude")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label="Cluster")
    for i in range(len(df_clean)):
        plt.annotate(
            df_clean["Community Name"].iloc[i],
            (df_clean["Longitude"].iloc[i], df_clean["Latitude"].iloc[i]),
            fontsize=8,
            alpha=0.7,
        )
    plt.show()

    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    from yellowbrick.cluster import KElbowVisualizer

    def kmeans_elbow(pts, y):
        # Instantiate the clustering model and visualizer
        km = KMeans(random_state=42)
        visualizer = KElbowVisualizer(km, k=(2, 10))

        visualizer.fit(coordinates)  # Fit the data to the visualizer
        visualizer.show()

        print("Best K : ", visualizer.elbow_value_)

        kmeans = KMeans(
            n_clusters=visualizer.elbow_value_, random_state=0
        )  # You can adjust the number of clusters
        kmeans.fit(pts)
        labels = kmeans.labels_
        plt.figure(figsize=(20, 15))
        plt.scatter(pts[:, 0], pts[:, 1], c=labels, s=50, cmap="viridis")
        # print(labels)
        for i in range(len(y)):
            plt.annotate(y[i][1], (pts[i, 0], pts[i, 1]))

        # You can visualize the clusters:
        plt.title("KMeans Clustering of pts (k using elbow method")
        plt.show()

        Clusters = []
        for i in range(len(y)):
            print(f"Value: {y[i]}, Cluster: {labels[i]}")
            Clusters.append([labels[i], y[i]])

        return Clusters

    def kmeans_silhouette(pts, y):
        sil_scores = []  # List to store silhouette scores

        # Try different values of k
        for k in range(2, 11):  # Silhouette score is undefined for k=1
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(coordinates)
            sil_score = silhouette_score(coordinates, cluster_labels)
            sil_scores.append(sil_score)

        # Plot the silhouette scores
        plt.figure(figsize=(8, 6))
        plt.plot(range(2, 11), sil_scores, marker="o")
        plt.title("Silhouette Score for Optimal k")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.show()

        print("Best k : ", np.argmax(sil_scores) + 2)

        kmeans = KMeans(
            n_clusters=np.argmax(sil_scores) + 2, random_state=0
        )  # You can adjust the number of clusters
        kmeans.fit(pts)
        labels = kmeans.labels_
        plt.figure(figsize=(20, 15))
        plt.scatter(pts[:, 0], pts[:, 1], c=labels, s=50, cmap="viridis")
        # print(labels)
        for i in range(len(y)):
            plt.annotate(y[i][1], (pts[i, 0], pts[i, 1]))

        # You can visualize the clusters:
        plt.title("KMeans Clustering of pts (k using Silhouette scores")
        plt.show()

        Clusters = []
        for i in range(len(y)):
            print(f"Value: {y[i]}, Cluster: {labels[i]}")
            Clusters.append([labels[i], y[i]])

        return Clusters

    def kmeans_davies_bouldin(pts, y):

        db_scores = []  # List to store Davies-Bouldin scores

        # Try different values of k
        for k in range(2, 11):  # The Davies-Bouldin index is undefined for k=1
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(coordinates)
            db_score = davies_bouldin_score(coordinates, cluster_labels)
            db_scores.append(db_score)

        # Plot the Davies-Bouldin scores
        plt.figure(figsize=(8, 6))
        plt.plot(range(2, 11), db_scores, marker="o")
        plt.title("Davies-Bouldin Index for Optimal k")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Davies-Bouldin Index")
        plt.show()

        print("Best K : ", np.argmin(db_scores) + 2)

        kmeans = KMeans(
            n_clusters=np.argmin(db_scores) + 2, random_state=0
        )  # You can adjust the number of clusters
        kmeans.fit(pts)
        labels = kmeans.labels_
        plt.figure(figsize=(20, 15))
        plt.scatter(pts[:, 0], pts[:, 1], c=labels, s=50, cmap="viridis")
        # print(labels)
        for i in range(len(y)):
            plt.annotate(y[i][1], (pts[i, 0], pts[i, 1]))

        # You can visualize the clusters:
        plt.title("KMeans Clustering of pts (k using Davies Bouldin")
        plt.show()

        Clusters = []
        for i in range(len(y)):
            print(f"Value: {y[i]}, Cluster: {labels[i]}")
            Clusters.append([labels[i], y[i]])

        return Clusters

    def run_kmeans(pts, y):
        # Instantiate the clustering model and visualizer
        kmeans_elbow(pts, y)
        kmeans_silhouette(pts, y)
        kmeans_davies_bouldin(pts, y)

    """## Part A

    Check whether the distance measure you provide are based on similarity or dissimilarity. Provide 1 if it is dissimilarity and 0 if it is similar

    ### All features from hospital
    """

    run_analysis(
        df,
        [
            "Number of Households",
            "Average persons per household",
            "Female-headed lone parent families",
            "Male-headed lone parent families",
        ],
        "minkowski",
        0,
        ["Rural (km^2)", "Residential (km^2)", "Commercial (km^2)"],
        "sqeuclidean",
        1,
    )

    """## Part B"""

    df_B = data_preperation(df, Values)
    df_B_distance_martrix, df_B_distance_y = data_preperation_distance(
        df_B, Values, "cosine"
    )

    data_analysis_TSNE(df_B, perplexity=20.0, random_state=5)
    run_kmeans(pts, y)

    def kmeans_B(pts, y, n_clusters, random_state=0):
        kmeans = KMeans(
            n_clusters=3, random_state=0
        )  # You can adjust the number of clusters
        kmeans.fit(pts)
        labels = kmeans.labels_
        plt.figure(figsize=(20, 15))
        plt.scatter(pts[:, 0], pts[:, 1], c=labels, s=50, cmap="viridis")
        for i in range(len(y)):
            plt.annotate(y[i][1], (pts[i, 0], pts[i, 1]))

        # You can visualize the clusters:
        plt.title("KMeans Clustering of pts")
        plt.show()

        cluster_array = []
        for i in range(len(y)):
            print(y[i], labels[i])
            cluster_array.append([labels[i], y[i][0]])
        cluster_array.sort()
        return cluster_array

    X, y = data_analysis_UMAP(df_B, n_neighbors=5, min_dist=0.1, random_state=5)

    test = kmeans_B(pts, y, 3, 0)

    pts = X

    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(pts)
    labels = kmeans.labels_
    plt.figure(figsize=(20, 15))
    plt.scatter(pts[:, 0], pts[:, 1], c=labels, s=50, cmap="viridis")
    for i in range(len(y)):
        plt.annotate(y[i][1], (pts[i, 0], pts[i, 1]))
    plt.title("KMeans Clustering of pts (k using elbow method")
    plt.show()
