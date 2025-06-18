import marimo

__generated_with = "0.13.15"
app = marimo.App(
    width="medium",
    app_title="P5 Essais",
    auto_download=["ipynb", "html"],
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Elaborer un modèle de clustering""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Rappel
        type: info

    ///
    La phase d'exploration précédente m'a permis d'identifier plusieurs caractéristiques pour caractériser les clients de Olist.

    - **'customer_unique_id'**,
    - **'recency'** (nombre de jours depuis la dernière commande),
    - **'frequency'** (nombre d'achat effectuée),
    - **'monetary'** (somme total des commandes),
    - **'log_frequency'**, ➡️ A privilégier dans les modèles de ML
    - **'log_monetary'**, ➡️ A privilégier dans les modèles de ML
    - **'zip_code'**,
    - **'average_review_score'** (note moyenne de satisfaction),
    - **'nb_reviews'**

    Les données concernent la période entre 2016 et 2018.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Nature du problème

    L'objectif de cette étude est d'utiliser les caractéristiques clients disponibles pour identifier des segments de clients pertinents pour le marketing.

    Ici, on a un grand nombre de donnée et les groupes de clients ne sont pas définis.

    Ce sera au modèle de les déterminer.

    Il s'agit d'un problème de classification non supervisé.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Définitions

    **Fonction de coût/Métrique**

    Les métriques choisit ici pour définir la performance des modèles sont le coefficient de silhouette, le coefficient de calinski_harabasz et la méthode du coude.

    Le coefficient de silhouette mesure la cohérence interne des clusters en comparant:

    - la distance moyenne des points au sein de chaque cluster
    - la distance moyenne entre les clusters.

    A savoir que:

    > Un coefficient de silhouette de 1 indique des clusters parfaitement distincts, tandis qu'un coefficient de -1 signale des clusters mal définis.

    > Des valeurs proches de 0 suggèrent des clusters qui se chevauchent, et des valeurs négatives indiquent généralement que certains points ont été mal classés, car ils sont plus proches d'un autre cluster.

    > Des valeurs négatives indiquent généralement que certains points ont été mal classés, car ils sont plus proches d'un autre cluster.

    > source: [scikit-learn_ silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score)

    Le coefficient calinski_harabasz de mesure le rapport entre la dispersion entre les clusters et la dispersion au sein des clusters
    > Un coefficient élevé indique de meilleures performances de clustering, avec une séparation plus élevée entre les clusters et une variance plus faible au sein des cluster

    Également lors de l'utilisation de l'algorithme des kmean, je vais mesurer la distorsion (la méthode du coude) afin d'identifier le nombre optimal de k voisins.

    > La méthode du coude aide à trouver le nombre optimal de clusters. Pour cela, on trace une courbe montrant comment la variance des points au sein des clusters diminue avec le nombre de clusters.
    > Le "coude" de la courbe indique le point où ajouter plus de clusters ne améliore plus significativement la séparation des données.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | En résumé
        type: info

        - Pour mesurer la séparation des clusters, on utilise les coefficients de silhouettes et de calinski_harabasz.

        - calinski_harabasz est plus rapide à calculer mais moins précis que silhouette

        - Pour déterminer le nombre de clusters optimum, on utilise la méthode du coude.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Algorithmes**

    Je vais tester les algorithmes de classification non supervisés à savoir K-MEANS, AgglomerativeClustering et DBSCAN.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Importation des librairies""")
    return


@app.cell
def _():
    # Import librairies
    from pathlib import Path

    # Data viz
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    import seaborn as sns
    import sqlalchemy

    pio.renderers.default = "browser"

    sns.set_theme()

    # ML
    from scipy.cluster.hierarchy import dendrogram
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import calinski_harabasz_score, silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    # Clustering Analysis
    from yellowbrick.cluster import KElbowVisualizer, silhouette_visualizer

    return (
        AgglomerativeClustering,
        DBSCAN,
        KMeans,
        NearestNeighbors,
        StandardScaler,
        TSNE,
        calinski_harabasz_score,
        dendrogram,
        np,
        pd,
        plt,
        px,
        silhouette_score,
        silhouette_visualizer,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Préparation des données""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Chargement du dataset précédent""")
    return


@app.cell
def _(pd):
    # load previous dataset
    final_df = pd.read_csv("data/processed/finale_df.csv")
    return (final_df,)


@app.cell(hide_code=True)
def _(final_df):
    final_df.dtypes
    return


@app.cell
def _(final_df):
    final_df.isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Je regarde la distribution des notes pour savoir si j'impute les valeurs manquantes ou non.""")
    return


@app.cell
def _(final_df, plt, sns):
    sns.kdeplot(final_df["average_review_score"])
    # Calculate median and mean
    median = final_df["average_review_score"].median()
    mean = final_df["average_review_score"].mean()

    # Add dashed blue line for median
    plt.axvline(median, color="blue", linestyle="--", label="Median")

    # Add dashed red line for mean
    plt.axvline(mean, color="red", linestyle="--", label="Mean")

    # Add a legend to identify the lines
    plt.legend()

    # Show the plot
    plt.show()
    return


@app.cell(hide_code=True)
def _(final_df, mo):
    mo.md(
        rf"""
    Les valeurs manquantes représentent **{(final_df["average_review_score"].isna().sum() / final_df.shape[0]):.2%}**

    Je supprime les lignes du jeu de données pour ne pas générer de bais.
    """
    )
    return


@app.cell
def _(final_df):
    final_df_filtered = final_df.dropna()
    return (final_df_filtered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Analyse de la segmentation RFM""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Lors de la phase d'exploration, j'ai appliqué une transformé logarithmique pour transformer la distribution des caractéristiques frequency et monetary. C'est celles que je vais conserver pour mon analyse.""")
    return


@app.cell
def _(final_df_filtered):
    rfm_df = final_df_filtered[["recency", "log_frequency", "log_monetary"]]
    return (rfm_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Encodage des features et détermination du nombre de cluster optimal""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Avant d'utiliser l'algorithme de Kmeans, je centre la distribution de chaque caractéristiques avec StandardScaler de Scikit Learn.

    Puis j'utilise la méthode du coude et les scores de silhouette pour déterminer le nombre optimal de clusters dans le clustering. 

    La méthode du coude mesure l'inertie, tandis que les scores de silhouette évaluent la cohésion et la séparation des clusters.
    """
    )
    return


@app.cell
def _(KMeans, StandardScaler, calinski_harabasz_score, mo, silhouette_score):
    def kmeans_optimized(X, clusters: list = range(2, 12), silhouettes_plot=False):
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # List to store the inertia values
        inertia = []
        # List to store the silhouette scores
        silhouettes = []
        # List to store the silhouette scores
        calinski = []

        # Calculate inertia for different numbers of clusters
        for i in mo.status.progress_bar(clusters):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X_scaled)

            # Get the computed labels
            cluster_labels = kmeans.labels_

            # Calculate inertia
            inertia.append(kmeans.inertia_)

            # Calculate calinski-harabasz score
            calinski.append(calinski_harabasz_score(X, kmeans.labels_))

            if silhouettes_plot:
                # Calculate silhouette score only if there are more than 1 cluster
                # Takes time to compute so use only for small numbers of k
                silhouettes.append(silhouette_score(X_scaled, cluster_labels, n_jobs=-1))

        return inertia, calinski, silhouettes

    return (kmeans_optimized,)


@app.cell
def _(plt):
    def plot_kmeans_optimized(clusters, inertia, calinski, silhouettes=None):
        # Plot the cost curve
        # Create a figure with 1 row and 2 columns
        plt.figure(figsize=(20, 6))
        grid = plt.GridSpec(1, 3, wspace=0.5, hspace=0.3)

        # In the first subplot, plot the results of the elbow method
        plt.subplot(grid[0, 0])
        plt.plot(clusters, inertia, marker="o")
        plt.title("Elbow Method")
        plt.xlabel("Number of clusters")
        plt.ylabel("Model Cost (Inertia)")

        # In the second subplot, plot the results of the calinski method
        plt.subplot(grid[0, 1])
        plt.plot(clusters, calinski, marker="o")
        plt.xlabel("Number of clusters")
        plt.ylabel("calinski-harabasz score")
        plt.title("Calinski-Harabasz Method")

        if silhouettes is not None:
            # In the third subplot, plot the results of the silhouette method
            plt.subplot(grid[0, 2])
            plt.plot(clusters, silhouettes, marker="o")
            plt.xlabel("Number of clusters")
            plt.ylabel("Silhouette score")
            plt.title("Silhouette Method")

        plt.show()

    return (plot_kmeans_optimized,)


@app.cell
def _(kmeans_optimized, mo, plot_kmeans_optimized, rfm_df):
    _clusters = range(2, 12)
    with mo.persistent_cache(name="Kmeans_silhouettes_rfm_cache"):
        _inertia, _calinski, _silhouettes = kmeans_optimized(rfm_df, _clusters, silhouettes_plot=True)

    plot_kmeans_optimized(_clusters, _inertia, _calinski, _silhouettes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Méthode du coude (Elbow Method)**

    La méthode du coude consiste à identifier le point où la diminution de la somme des carrés intra-clusters (inertie) commence à ralentir. Ce point est souvent considéré comme le nombre optimal de clusters.

    **Analyse :**Sur le graphique de gauche, on observe une forte diminution de l'inertie jusqu'à environ 4 clusters. Après ce point, la diminution de l'inertie devient moins prononcée.

    Conclusion : Le nombre optimal de clusters semble être autour de 4.

    **Méthode de Calinski-Harabasz**

    La méthode de Calinski-Harabasz évalue la qualité des clusters en calculant le rapport de la variance inter-clusters à la variance intra-clusters. Un score plus élevé indique une meilleure qualité des clusters.

    Analyse : Sur le graphique du milieu, le score de Calinski-Harabasz est le plus élevé pour 2 clusters et diminue rapidement ensuite. Cependant, il y a un léger plateau autour de 4 clusters.

    Conclusion : Bien que le score soit le plus élevé pour 2 clusters, un nombre de clusters autour de 4 pourrait également être considéré comme optimal.

    **Méthode de la silhouette**

    La méthode de la silhouette mesure la similarité d'un point avec son propre cluster par rapport aux autres clusters. Un score de silhouette plus élevé indique une meilleure séparation des clusters.

    Analyse : Sur le graphique de droite, le score de silhouette est le plus élevé pour 2 clusters, mais il y a également des pics autour de 4 et 6 clusters.

    **Conclusion :** Le nombre optimal de clusters de 4 ou 5 semble être un bon compromis.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""J'ai identifié le nombre de clusters qui me semblent optimum (4 ou 5) voyons la répartition des clients dans ces groupes.""")
    return


@app.cell
def _(StandardScaler, rfm_df):
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_df)
    return X_scaled, scaler


@app.cell
def _(KMeans, X_scaled, plt):
    # Create a figure with 1 line and 2 columns
    figure = plt.figure(figsize=(25, 20))
    grid_spec = plt.GridSpec(1, 2, wspace=0.5, hspace=0.3)

    # Train the first model with 4 clusters
    kmeans_4_3vars_clusters = KMeans(n_clusters=4, random_state=42)
    kmeans_4_3vars_clusters.fit(X_scaled)
    # Define a 3D subplot
    axis_4_3vars_clusters = figure.add_subplot(grid_spec[0, 0], projection="3d")
    # Plot the values colored by labels
    axis_4_3vars_clusters.scatter(
        xs=X_scaled[:, 0], ys=X_scaled[:, 1], zs=X_scaled[:, 2], c=kmeans_4_3vars_clusters.labels_, cmap="Set1"
    )
    axis_4_3vars_clusters.set_xlabel("Récence", fontsize=15)
    axis_4_3vars_clusters.set_ylabel("Fréquence", fontsize=15)
    axis_4_3vars_clusters.set_zlabel("Montant", fontsize=15)
    plt.title("Représentation des clusters avec 4 groupes", fontweight="bold", fontsize=20)

    # Train the second model with 5 clusters
    kmeans_5_3vars_clusters = KMeans(n_clusters=5, random_state=42)
    kmeans_5_3vars_clusters.fit(X_scaled)
    # Define a 3D subplot
    axis_5_3vars_clusters = figure.add_subplot(grid_spec[0, 1], projection="3d")
    # Plot the values colored by labels
    axis_5_3vars_clusters.scatter(
        xs=X_scaled[:, 0], ys=X_scaled[:, 1], zs=X_scaled[:, 2], c=kmeans_5_3vars_clusters.labels_, cmap="Set1"
    )
    axis_5_3vars_clusters.set_xlabel("Récence", fontsize=15)
    axis_5_3vars_clusters.set_ylabel("Fréquence", fontsize=15)
    axis_5_3vars_clusters.set_zlabel("Montant", fontsize=15)
    plt.title("Représentation des clusters avec 5 groupes", fontweight="bold", fontsize=20)
    return kmeans_4_3vars_clusters, kmeans_5_3vars_clusters


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Observons les coeficcients de silhouettes de chaque groupes.""")
    return


@app.cell
def _(X_scaled, kmeans_4_3vars_clusters, mo, silhouette_visualizer):
    with mo.persistent_cache(name="plots_silhouettes_4clusters_rfm_cache"):
        _viz = silhouette_visualizer(kmeans_4_3vars_clusters, X_scaled, show=False, colors="yellowbrick")

    _viz.show()
    return


@app.cell
def _(X_scaled, kmeans_5_3vars_clusters, mo, silhouette_visualizer):
    with mo.persistent_cache(name="plots_silhouettes_5clusters_rfm_cache"):
        _viz = silhouette_visualizer(kmeans_5_3vars_clusters, X_scaled, show=False, colors="yellowbrick")
    _viz.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Avec 4 clusters, on remarque que les clusters sont plus homègenes qu'avec 5, dont notamment un avec un score de silhouette très élévé.

    La répartition avec 4 clusters semble être la plus pertinente.
    """
    )
    return


@app.cell
def _(plt, sns):
    def plot_cluster_stats(df, cluster_labels, characteristics, stats):
        """Function to plot cluster statistics.

        Parameters:
        - df: DataFrame containing the characteristics (recency, frequency, monetary, etc.)
        - cluster_labels: Cluster labels (array-like)
        - characteristics: List of characteristics to analyze
        - stats: List of statistics to calculate
        """
        # Create a copy of the DataFrame and add cluster labels
        rfm_named_df = df[characteristics].copy()
        rfm_named_df["Clusters"] = cluster_labels

        # Create aggregation dictionary
        agg_dict = dict.fromkeys(characteristics, stats)
        # Add count for one characteristic (it will be the same for all)
        agg_dict[characteristics[0]] = ["count"] + agg_dict[characteristics[0]]

        # Calculate statistics
        _stats_results = rfm_named_df.groupby("Clusters").agg(agg_dict).round().reset_index()

        # Flatten multi-index columns
        new_columns = ["Clusters", "Count"]
        for char in characteristics:
            for stat in stats:
                new_columns.append(f"{char}_{stat}")

        _stats_results.columns = new_columns

        fig0 = plt.figure(figsize=(5, 3))
        sns.barplot(x="Clusters", y="Count", data=_stats_results)
        plt.title("Count by Cluster")
        plt.show()

        # Create a figure with a grid of subplots
        fig = plt.figure(figsize=(15, 6))
        grid = plt.GridSpec(2, len(characteristics), wspace=0.4, hspace=0.4)

        # Create bar plots for each characteristic and statistic
        for i, char in enumerate(characteristics):
            for j, _ in enumerate(stats):
                ax = plt.subplot(grid[j, i])
                sns.boxplot(x="Clusters", y=rfm_named_df[char], data=rfm_named_df, ax=ax, showfliers=False)
                ax.set_title(f"{char.capitalize()} by Cluster")
                ax.set_xlabel("Cluster")
                ax.set_ylabel(f"{char.capitalize()}")

        # Adjust layout
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

        # Show plot
        plt.show()

        return _stats_results

    return (plot_cluster_stats,)


@app.cell
def _(final_df_filtered, kmeans_4_3vars_clusters, plot_cluster_stats):
    plot_cluster_stats(
        final_df_filtered, kmeans_4_3vars_clusters.labels_, ["recency", "frequency", "monetary"], stats=["mean"]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Les graphiques ci dessus montrent les  diffrérentes façons de consommer des groupes de clients. On a:

    - Groupe 0: Nouveaux clients qui n'ont pas beaucoup dépensé
    - Groupe 1: Clients récents qui ont beaucoup dépensé
    - Groupe 2: Clients qui ont utilisé la plateforme plus d'une fois
    - Groupe 3: Clients qui n'ont pas utilisé la plateforme depuis longtemps.

    Il serait interessant de visualiser les groupes clients créer en ajoutant les notes de satisfactions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## K-Means avec 4 variables""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""J'ajoute la note de satisfaction client aux cractéristiques clients.""")
    return


@app.cell
def _(final_df_filtered):
    rfm_4var_df = final_df_filtered[["recency", "log_frequency", "log_monetary", "average_review_score"]]
    return (rfm_4var_df,)


@app.cell
def _(kmeans_optimized, mo, plot_kmeans_optimized, rfm_4var_df):
    _clusters = range(2, 12)
    with mo.persistent_cache(name="Kmeans_silhouettes_rfm_4var_cache"):
        _inertia, _calinski, _silhouettes = kmeans_optimized(rfm_4var_df, _clusters, silhouettes_plot=True)

    plot_kmeans_optimized(_clusters, _inertia, _calinski, _silhouettes)
    return


@app.cell
def _():
    # _model=KMeans()
    # _visualizer = KElbowVisualizer(_model, k=(2,13))

    # _visualizer.fit(rfm_4var_scaled)        # Fit the data to the visualizer
    # _visualizer.show()        # Finalize and render the figure
    # _visualizer.elbow_value_ # Get elbow value
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Un nombre de cluster égale à 5,6 ou 8 semblent être les plus pertienents.""")
    return


@app.cell
def _(KMeans, StandardScaler):
    def train_kmeans_models(X, clusters):
        """Train KMeans models with different numbers of clusters.

        Parameters:
        X (array-like): Unnormalized data to cluster.
        clusters (list): List of numbers of clusters to test.

        Returns:
        dict: Dictionary containing the trained models.
        """
        models = {}

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for n_clusters in clusters:
            # Initialize and train the KMeans model
            model = KMeans(n_clusters=n_clusters, random_state=42)
            model.fit(X_scaled)

            # Stockage du modèle
            models[f"kmeans_{n_clusters}"] = model

        return models

    return (train_kmeans_models,)


@app.cell
def _(rfm_4var_df, scaler, train_kmeans_models):
    _clusters = [5, 6, 8]
    kmeans_models = train_kmeans_models(rfm_4var_df, _clusters)
    kmeans_5_4var = kmeans_models["kmeans_5"]
    kmeans_6_4var = kmeans_models["kmeans_6"]
    kmeans_8_4var = kmeans_models["kmeans_8"]

    # Standardize features
    rfm_4var_scaled = scaler.fit_transform(rfm_4var_df)
    return kmeans_5_4var, kmeans_6_4var, kmeans_8_4var, rfm_4var_scaled


@app.cell
def _(kmeans_5_4var, mo, rfm_4var_scaled, silhouette_visualizer):
    with mo.persistent_cache(name="plots_silhouettes_5clusters_4var_kmeans"):
        _viz = silhouette_visualizer(kmeans_5_4var, rfm_4var_scaled, show=False, colors="yellowbrick")

    _viz.show()
    return


@app.cell
def _(kmeans_6_4var, mo, rfm_4var_scaled, silhouette_visualizer):
    with mo.persistent_cache(name="plots_silhouettes_6clusters_4var_kmeans"):
        _viz = silhouette_visualizer(kmeans_6_4var, rfm_4var_scaled, show=False, colors="yellowbrick")

    _viz.show()
    return


@app.cell
def _(kmeans_8_4var, mo, rfm_4var_scaled, silhouette_visualizer):
    with mo.persistent_cache(name="plots_silhouettes_8clusters_4vars_kmeans"):
        _viz = silhouette_visualizer(kmeans_8_4var, rfm_4var_scaled, show=False, colors="yellowbrick")

    _viz.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Les groupes sont plus homogènes avec 5 clusters.

    Le score de silhouette moyen est de 0,32.

    Au delà de 5 groupes, je ne pense pas être capable de les analyser.
    """
    )
    return


@app.cell
def kmeans_5_4vars_plots(final_df_filtered, kmeans_5_4var, plot_cluster_stats):
    plot_cluster_stats(
        final_df_filtered,
        kmeans_5_4var.labels_,
        ["recency", "frequency", "monetary", "average_review_score"],
        stats=["mean"],
    )
    return


@app.cell
def _(TSNE, kmeans_5_4var, mo, pd, px, rfm_4var_scaled):
    with mo.persistent_cache(name="TSNE_kmeans6"):
        _X_embedded = TSNE(n_components=2).fit_transform(rfm_4var_scaled)
        _X_embedded = pd.DataFrame(_X_embedded, columns=["x_component", "y_component"])
        _X_embedded["Clusters"] = kmeans_5_4var.labels_.astype(str)
        _X_embedded = _X_embedded.sort_values(by="Clusters")

    _fig = px.scatter(_X_embedded, x="x_component", y="y_component", color="Clusters", size_max=60)
    _fig.update_layout(height=800)
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## DBSCAN""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) est un algorithme de clustering qui utilise deux paramètres principaux : eps (epsilon) et MinPts (nombre minimum de points).

    - **eps** : C'est le rayon autour d'un point pour chercher ses voisins. Si d'autres points se trouvent dans ce rayon, ils sont considérés comme voisins.

    - **MinPts** : C'est le nombre minimum de points nécessaires pour qu'un point soit considéré comme un point central (ou "core point") d'un cluster. Un point central doit avoir au moins MinPts voisins dans un rayon de eps.
    Une règle empirique qui fonctionne est d'utiliser 2 fois le nombre de features.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""L'algorithme nécessitant beaucoup de puissance de calculs, je vais tester l'algorithme sur un échantillon.""")
    return


@app.cell
def _(final_df_filtered, pd, scaler):
    # Create a sample of the dataframe
    final_df_filtered_sampled = pd.DataFrame(
        final_df_filtered,
        columns=["recency", "log_frequency", "frequency", "log_monetary", "monetary", "average_review_score"],
    ).sample(n=10000, random_state=42)

    rfm_4var_sampled = final_df_filtered_sampled[["recency", "log_frequency", "log_monetary", "average_review_score"]]

    # Standardized features
    rfm_4var_scaled_sampled = scaler.fit_transform(rfm_4var_sampled)
    return final_df_filtered_sampled, rfm_4var_scaled_sampled


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Pour déterminer la valeur optimale de eps, j'utilise la méthode du "k-distance graph". Voici comment cela fonctionne :

    Calcul des distances aux k plus proches voisins :

    - Pour chaque point, on trouve ses k voisins les plus proches. Par exemple, si MinPts est 8, on cherche les 8 points les plus proches.

    - Tri des distances :

    - On trace ces distances par ordre croissant. Le "coude" (ou "knee") de ce graphique indique une bonne valeur pour eps.
    """
    )
    return


@app.cell
def _(NearestNeighbors, np, plt, rfm_4var_scaled_sampled):
    # initialize the value of k for kNN which can be same as MinPts
    _k = 8

    # Compute k-nearest neighbors
    # you need to add 1 to k as this function also return
    # distance to itself (first column is zero)
    _nbrs = NearestNeighbors(n_neighbors=_k + 1).fit(rfm_4var_scaled_sampled)

    # get distances
    # Compute the distances and indices of our data
    distances, _ = _nbrs.kneighbors(rfm_4var_scaled_sampled)

    # Sort distances
    distances = np.sort(distances, axis=0)

    # drop first which is zero
    distances = distances[:, 1]

    # Plot the distances regarding the number of points
    plt.plot(distances)
    plt.ylim([0, 0.6])
    plt.title("Kneighbors-distance Graph", fontsize=20)
    plt.xlabel("Data Points", fontsize=14)
    plt.ylabel("Distance")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Le point d'inflexion de la courbe est pour eps égal à 0.1.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Après plusieurs essais, j'ai définis **eps=0.3** et **MinPts=8**

    Voyons le résultat de DBscans.
    """
    )
    return


@app.cell
def _(DBSCAN, mo, rfm_4var_scaled_sampled):
    with mo.persistent_cache(name="rfm_4var_dbscan"):
        # Apply the DBSCAN algorithm with the optimal values
        dbscan = DBSCAN(eps=0.35, min_samples=8, n_jobs=-1)
        # Train it and predict the labels on the dataframe
        dbscans_clusters = dbscan.fit_predict(rfm_4var_scaled_sampled)
    return dbscan, dbscans_clusters


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Analyse du model en calculant les coefficients de silhouettes.""")
    return


@app.cell
def _(dbscans_clusters, rfm_4var_scaled_sampled, silhouette_score):
    # Compute the silhouette score of the model
    silhouette_score(rfm_4var_scaled_sampled, dbscans_clusters)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Le coefficient de silhouette est proche de 0 ce qui indique des clusters mal différenciés.""")
    return


@app.cell
def _(dbscan, final_df_filtered_sampled, plot_cluster_stats):
    plot_cluster_stats(
        final_df_filtered_sampled,
        dbscan.labels_,
        ["recency", "frequency", "monetary", "average_review_score"],
        stats=["mean"],
    )
    return


@app.cell
def _(TSNE, pd, px):
    def plot_clusters_tsne(model, X_scaled):
        """Generates and displays a scatter plot of clusters using t-SNE for dimensionality reduction.

        Parameters:
        model: A pre-fitted model.
        X_scaled (array-like): Normalized data to be reduced to 2 dimensions.
        sample: Reduce dataframe to speed up compute time

        Returns:
        None
        """
        # Apply t-SNE to reduce dimensions to 2
        X_embedded = TSNE(n_components=2).fit_transform(X_scaled)

        # Create a DataFrame with t-SNE components
        X_embedded = pd.DataFrame(X_embedded, columns=["x_component", "y_component"])

        # Add cluster labels
        X_embedded["Clusters"] = model.labels_.astype(str)

        # Sort data by clusters
        X_embedded = X_embedded.sort_values(by="Clusters")

        # Create the scatter plot
        fig = px.scatter(X_embedded, x="x_component", y="y_component", color="Clusters", size_max=60)

        # Update the height of the plot
        fig.update_layout(height=800)

        # Display the plot
        return fig

    return (plot_clusters_tsne,)


@app.cell(disabled=True)
def _(dbscan, mo, plot_clusters_tsne, rfm_4var_scaled_sampled):
    with mo.persistent_cache(name="plot_dbscan"):
        _viz = plot_clusters_tsne(dbscan, rfm_4var_scaled_sampled)
    _viz.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    DBSCAN est plus adapté pour des données qui sont de formes sphériques ou convexes.
    Je ne vais pas conserver ce modèle pour mon étude.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Hierarchical clustering""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    J'utilise un autre type d'algorithme, l'algorithme hiérarchique.

    Le regroupement agglomératif est un algorithme de regroupement hiérarchique qui regroupe des points de données similaires.

    Il commence par chaque point de données comme un regroupement séparé puis combine ces regroupements de manière itérative en fonction de leur similarité jusqu'à ce que tous les points de données appartiennent à un seul regroupement.
    """
    )
    return


@app.cell
def _(AgglomerativeClustering, rfm_4var_scaled_sampled):
    # Initiate the Agglomerative Clustering algorithm with ward method to minimize the variance within clusters
    agglo = AgglomerativeClustering(n_clusters=5, linkage="ward", compute_distances=True)

    # Ajustement du modèle aux données
    clusters = agglo.fit_predict(rfm_4var_scaled_sampled)
    # Fit the data to the model and determine which clusters each data point belongs to

    return (agglo,)


@app.cell
def _(agglo, dendrogram, plt, rfm_4var_scaled_sampled):
    from scipy.cluster.hierarchy import linkage

    # Nous allons utiliser la méthode 'ward' pour le linkage
    Z = linkage(rfm_4var_scaled_sampled, "ward")

    plt.title("Hierarchical Clustering Dendrogram")
    # plot the 5 levels of the dendrogram and give individual color per cluster
    plt.xlabel("Number of customers per clusters")
    _threshold = 0
    # Make the dendrogram

    dendrogram(
        Z, labels=agglo.labels_, truncate_mode="lastp", p=5, color_threshold=_threshold, above_threshold_color="grey"
    )
    # dendrogram(Z, labels=agglo.labels_, truncate_mode = 'level', p=2 , color_threshold=_threshold, above_threshold_color='grey')
    # Add horizontal line.
    # plt.axhline(y=_threshold, c='grey', lw=1, linestyle='dashed')

    # Show the graph
    plt.show()
    return


@app.cell
def _(agglo, rfm_4var_scaled_sampled, silhouette_score):
    silhouette_score(rfm_4var_scaled_sampled, agglo.labels_)
    return


@app.cell
def _(agglo, final_df_filtered_sampled, plot_cluster_stats):
    plot_cluster_stats(
        final_df_filtered_sampled,
        agglo.labels_,
        ["recency", "frequency", "monetary", "average_review_score"],
        stats=["median"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""L'algorithme est long à process pour un résultat moins bon qu'avec le Kmeans.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Exploitation du modèle choisi""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    L'algorithme qui me semble le plus adapté au jeu de données est le Kmeans qui est rapide et donnes des clusters suffisement distincts.

    Je n'obtient pas de meilleurs résultats avec DBscans ou AgglomerativeClustering.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Analyse métier des groupes de clients""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    L'algorithme le plus performant est le Kmeans avec 5 clusters avec 4 caractéristiques clients (recency, frequency(log), monetary(log), average_review_score)

    Voyons les caractéristiques des clients par groupe.

    Pour chaque client j'ajoute le label du groupe correspondant.
    """
    )
    return


@app.cell
def _(final_df_filtered, kmeans_5_4var, plot_cluster_stats):
    plot_cluster_stats(
        final_df_filtered,
        kmeans_5_4var.labels_,
        ["recency", "frequency", "monetary", "average_review_score"],
        stats=["mean"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On observe différents type de clients qui partagent quand même des caractéristiques communes.

    - Groupe 0: 12% des clients qui sont insatisfaits du service de Olist et ne recommenderont probablement pas (Frequency =1)

    - Groupe 1: 20% de clients qui dépensent plus que la moyenne mais ne sont pas fidélisés (Monetary > 200 et Frequency =1)

    - Groupe 2: clients récurents (Frequency = 2) mais représentent moins de 1% de la totalité

    - Groupe 3: Anciens clients qui pourraient revenir car satisfait (review = 5)

    - Groupe 4: 30% des clients sont nouveaux et dépensent peu (recency < 200 et monetary = 69€)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Stabilité des clusters à l'initialisation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Pour évaluer la stabilité des clusters identifiés par l'algorithme, je vais comparer la constitution des groupes obtenus à travers plusieurs exécutions. 

    Pour ce faire, je vais fixer le nombre d'initialisations de l'algorithme à 1 (n_init=1) et supprimer le paramètre random_state afin d'introduire de la variabilité dans l'initialisation des centroïdes. 

    Si l'inertie des groupes reste similaire à chaque itération, cela pourrait indiquer que les groupes trouvés ont une constitution similaire et sont donc stables.
    """
    )
    return


@app.cell
def _(KMeans, plt, rfm_4var_scaled, sns):
    # Parameters
    n_initializations = 20
    n_clusters = 5

    # List to store inertias
    inertias = []
    for i in range(n_initializations):
        # Initialize i models with the same number of clusters
        kmeans = KMeans(n_clusters=n_clusters, n_init=1)
        # Train the model and make predictions
        kmeans.fit_predict(rfm_4var_scaled)
        # Store the inertias in the list
        inertias.append(kmeans.inertia_)

    # Plot a barplot of computed inertias
    sns.barplot(inertias)
    # Add a line with the mean of inertias
    sns.lineplot(x=[0, 20], y=sum(inertias) / len(inertias))
    plt.title("Inertie des clusters avec différentes initialisation", fontweight="bold")
    plt.xlabel("Itérations")
    plt.ylabel("Inertie")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Différentes initialisations de KMeans conduisent à des clusters similaires. 

    Ici, la plupart des initialisations ont des inerties similaires, ce qui suggère une certaine stabilité.
    """
    )
    return


@app.cell
def _(final_df_filtered):
    final_df_filtered.to_csv("data/processed/finale_df_filtered.csv")
    return


if __name__ == "__main__":
    app.run()
