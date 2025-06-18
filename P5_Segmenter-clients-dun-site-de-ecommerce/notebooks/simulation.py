import marimo

__generated_with = "0.13.15"
app = marimo.App(
    width="medium",
    app_title="P5 Simulation",
    auto_download=["ipynb", "html"],
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# P5 Simulation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Rappel
        type: info

    ///

    Dans le notebook précédent, j'ai pu établir une segmentation clients basés sur les caractéristiques clients.

    - **'customer_unique_id'**,
    - **'recency'** (nombre de jours depuis la dernière commande),
    - **'frequency'** (nombre d'achat effectuée),
    - **'monetary'** (somme total des commandes),
    - **'log_frequency'**, ➡️ A privilégier dans les modèles de ML
    - **'log_monetary'**, ➡️ A privilégier dans les modèles de ML
    - **'average_review_score'** (note moyenne de satisfaction),

    J'ai ainsi catégorisé les clients en 5 groupes.

    - Groupe 0: 12% des clients qui sont insatisfaits du service de Olist et ne recommanderont probablement pas (Frequency =1)
    - Groupe 1: 20% de clients qui dépensent plus que la moyenne mais ne sont pas fidélisés (Monetary > 200 et Frequency =1)
    - Groupe 2: clients récurrents (Frequency = 2) mais représentent moins de 1% de la totalité
    - Groupe 3: Anciens clients qui pourraient revenir car satisfait (review = 5)
    - Groupe 4: 30% des clients des clients sont nouveaux et dépensent peu (recency < 200 et monetary = 69€)

    L'objectif étant de cibler les groupes de clients par des campagnes marketing ciblées.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Les limites de ma segmentation c'est quelle est figée à un instant "t" et les groupes clients peuvent devenir obsolète. 

    La maintenance d'une segmentation client est cruciale pour plusieurs raisons :

    - Évolution des comportements : Les clients changent avec le temps, leurs préférences et comportements évoluent.
    - Nouveaux clients : L'arrivée de nouveaux clients peut modifier la structure des segments existants.
    - Précision des modèles : Les modèles de segmentation peuvent se dégrader avec le temps et doivent être mis à jour pour rester pertinents.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Dans ce notebook, je vais essayer d'estimer la période de temps à partir de laquelle une maintenance de ma modélisation précédente devra être effectuée.

    Pour cela, je vais utiliser le score ARI qui permet comparer deux segmentations clients.

    - ARI = 1 ➡️ segmentation identiques
    - ARI = 0 ➡️ segmentation complètement différentes
    - ARI = 0.5 ➡️ segmentation très mauvaise

    Je me fixe un seuil de **0.8** en dessous duquel, le modèle de ML n'est plus adapté aux informations clients et des ajustements sont nécessaires.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Afin d'évaluer la stabilité de ma segmentation client voici la méthodologie utilisée :

    - Diviser la base de données en période de de 1 mois
    - Appliquer le clustering à chacune des périodes
    - Comparer les segmentation une à une à l'aide de l'indice ARI
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
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.manifold import TSNE
    from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    return (
        ColumnTransformer,
        FunctionTransformer,
        KMeans,
        Pipeline,
        StandardScaler,
        TSNE,
        adjusted_rand_score,
        np,
        pd,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Chargement des données""")
    return


@app.cell
def _(pd):
    finale_df_filtered = pd.read_csv("data/processed/finale_df_filtered.csv")
    finale_df_filtered.describe()
    return (finale_df_filtered,)


@app.cell
def _(finale_df_filtered):
    # Load variables and sort by recency days
    rfm_4_var = finale_df_filtered[["recency", "frequency", "monetary", "average_review_score"]].sort_values(
        by="recency"
    )
    return (rfm_4_var,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Initie un StandardScaler pour normaliser les données et l'algorithme Kmeans.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Divise le jeu de données pour avoir une division de clients tous les mois.""")
    return


@app.cell
def _(rfm_4_var, sns):
    sns.histplot(rfm_4_var["recency"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Préparation du model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Etape 1: Pipeline et transformation des caractéristiques clients
    Je commence par définir un pipeline pour transformer les caractéristiques clients avant leur utilisation par le modèle.

    - Transformation logarithmique pour les caractéristiques (**frequency** et **monetary**)
    - Toutes les colonnes sont ensuite centrées-réduites avec un **StandardScaler**
    - Je définis **KMeans** comme model à utiliser et le nombre de cluster à **5**

    ```python
    def log_transform(X):
        # Apply log transformation to the input data
        return np.log1p(X)

    def detect_model_drift(t_0, rfm_df):
        ###
        # Apply log transformation to the 'frequency' and 'monetary' columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('log_freq', FunctionTransformer(log_transform), ['frequency']),
                ('log_mon', FunctionTransformer(log_transform), ['monetary'])
            ],
            remainder='passthrough'  # Pass through other columns without transformation
        )

        # Create the pipeline for the initial model
        # The pipeline includes preprocessing, scaling, and KMeans clustering
        initial_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=5, random_state=42))
        ])
        ###
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Etape 2: Entraîner un modèle initial
    Je commence par entraîner le modèle de clustering avec les informations clients tout au long de la première année d'existence de Olist.

    Pour cela j'utilise la colonne recency qui contient le nombre de jour écoulés entre aujourd'hui et la commmande passée.
    La valeur max de recency correspond à la plus ancienne commande par un client.

    Pour obtenir la commande 1 an après, je soustrait 365 jours au max de recency.


    /// admonition | Info
        type: info

        Utiliser 1 an de données me permet d'éviter les biais de saisonalités où on pourrait avoir moins de client par exemple.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Etape 3: Comparaison des clusters formés avec plus de données et plus récentes

    Ensuite, je vais entraîner d'autres modèles en partant de la même base de caractéristiques, mais en avançant dans le temps semaine par semaine. 

    La date de début est initialement définie comme
    ```python
    start_date = rfm_4_var_df["recency"].max() - t_0
    ```

    Pour chaque itération, j'ajoute 14 jours à la date de début pour analyser des périodes de temps de plus en plus récentes. Cela signifie que je commence à partir d'une date de référence et que je me déplace vers des données plus récentes.

    Je compare les clusters formés par le modèle initial (entraîné sur les données de la période de référence, la plus ancienne) avec les clusters formés par les modèles entraînés sur ces périodes de temps plus récentes.

    Pour cela, j'utilise le score **ARI** (Adjusted Rand Index), qui mesure la similarité entre deux partitions de données. 

    /// admonition | Rappel
    Une valeur **ARI** de 1 indique une similarité parfaite, tandis qu'une valeur de 0 indique une similarité aléatoire.
    ///
    J'itère plusieurs fois en avançant d'une semaine à chaque itération pour voir à partir de quelle période le modèle d'origine n'est plus adapté aux nouvelles données, ce qui indique une dérive du modèle et nécessite un réentraînement. 

    Une baisse significative de l'ARI suggère que les clusters formés par le modèle initial ne correspondent plus aux clusters formés par les nouveaux modèles, indiquant ainsi une dérive du modèle. 

    Cette dérive peut être due à des changements dans le comportement des clients ou à d'autres facteurs externes qui affectent les données.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("soutenance/Evaluation_stabilite_kmeans.jpg", width=500, height=500)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Adjusted Rand Score""")
    return


@app.cell
def _(mo):
    # Slider to try different number of additional days to consider at each iteration
    # Must be a multiple of 7 and value cannot be greater than one year in days (7*52)
    days_increment_slider = mo.ui.slider(start=7, stop=7 * 3, value=14, step=7, label="Recency window size")
    days_increment_slider
    return (days_increment_slider,)


@app.cell(hide_code=True)
def _(
    ColumnTransformer,
    FunctionTransformer,
    KMeans,
    Pipeline,
    StandardScaler,
    adjusted_rand_score,
    days_increment_slider,
    np,
    pd,
):
    def log_transform(X):
        # Apply log transformation to the input data
        return np.log1p(X)

    def detect_model_drift(t_0, rfm_df):
        # Parameters
        # Number of additional days to consider at each iteration, 7 by default
        days_increment = days_increment_slider.value

        # Calculate the start date based on the maximum recency value and t_0
        start_date = rfm_df["recency"].max() - t_0
        current_date = start_date  # Init starting date

        # Sort dataframe by recency
        rfm_df = rfm_df.sort_values(by="recency")

        # Calculate the initial DataFrame
        # Filter the DataFrame to include only rows with recency greater than or equal to start_date
        initial_df = rfm_df[rfm_df["recency"] >= start_date].copy()
        # Adjust the recency values by subtracting the start_date
        initial_df["recency"] -= start_date

        # Define the transformations for the preprocessor
        # Apply log transformation to the 'frequency' and 'monetary' columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("log_freq", FunctionTransformer(log_transform), ["frequency"]),
                ("log_mon", FunctionTransformer(log_transform), ["monetary"]),
            ],
            remainder="passthrough",  # Pass through other columns without transformation
        )

        # Create the pipeline for the initial model
        # The pipeline includes preprocessing, scaling, and KMeans clustering
        initial_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=5, random_state=42)),
        ])

        # Train the initial model
        initial_pipeline.fit(initial_df)

        # Dictionary to store Adjusted Rand Index (ARI) scores with week numbers
        ari_scores = {}
        current_week = 0  # Initiat week number

        # Lists to store predictions, DataFrames, and centroids
        predictions_list = []
        dataframes_list = []
        centroids_list = []

        # Loop over different time periods using a while loop
        while current_date > 0:
            # Filter the DataFrame to include only rows with recency greater than or equal to current_date
            current_df = rfm_df[rfm_df["recency"] >= current_date].copy()
            # Adjust the recency values by subtracting the current_date
            current_df["recency"] -= current_date

            # Create the pipeline for the current model
            # The pipeline includes preprocessing, scaling, and KMeans clustering
            current_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=5, random_state=42)),
            ])
            # Train the current model
            current_pipeline.fit(current_df)

            # Predict clusters for the current model
            real_clusters = current_pipeline.predict(current_df)
            # Predict clusters using the initial model
            predicted_clusters = initial_pipeline.predict(current_df)

            # Get the centroids from the current model
            current_centroids = current_pipeline.named_steps["kmeans"].cluster_centers_
            initial_centroids = initial_pipeline.named_steps["kmeans"].cluster_centers_

            # Calculate the Adjusted Rand Index (ARI) score
            ari_score = adjusted_rand_score(real_clusters, predicted_clusters)
            ari_scores[current_week] = ari_score

            # Store predictions, DataFrames, and centroids
            predictions_list.append({
                "week_number": current_week,
                "real_clusters": real_clusters,
                "predicted_clusters": predicted_clusters,
            })
            dataframes_list.append({"week_number": current_week, "dataframe": current_df})
            centroids_list.append({
                "week_number": current_week,
                "current_centroids": current_centroids,
                "initial_centroids": initial_centroids,
            })

            # Update the current date and week number
            current_date -= days_increment
            current_week += days_increment / 7  # Update week number

        # Convert the dictionary to a pandas DataFrame
        ari_scores_df = pd.DataFrame(list(ari_scores.items()), columns=["Week_Number", "ARI_Score"])

        # Return the list of ARI scores, predictions, DataFrames, and centroids
        return ari_scores_df, predictions_list, dataframes_list, centroids_list

    return (detect_model_drift,)


@app.cell(hide_code=True)
def _(plt, sns):
    def plot_ari_score(ari_scores_df, threshold=0.8):
        sns.set_theme()
        # Set the figure size with increased width
        plt.figure(figsize=(15, 6))

        # Create an array of week numbers to use as x-axis values
        plt.plot(ari_scores_df["Week_Number"], ari_scores_df["ARI_Score"], label="ARI Score", marker="o")
        plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ARI = {threshold}")

        # Find the first week where ARI score is below the threshold
        below_threshold = ari_scores_df[ari_scores_df["ARI_Score"] < threshold]

        if not below_threshold.empty:
            first_below_threshold = below_threshold["Week_Number"].iloc[0]
            plt.axvline(
                x=first_below_threshold,
                color="black",
                linestyle="--",
                label=f"First value below threshold (Week {int(first_below_threshold)})",
            )

        plt.title("Temporal Stability of K-Means Segmentation", fontsize=18, color="b")
        plt.xlabel("Number of Weeks")
        plt.ylabel("ARI Score")

        # Set x-ticks to display at regular intervals for better readability
        plt.xticks(ari_scores_df["Week_Number"])

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add grid lines for better readability
        plt.grid(visible=True, linestyle="--", alpha=0.7)

        plt.legend()
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.show()

    return (plot_ari_score,)


@app.cell
def _(detect_model_drift, plot_ari_score, rfm_4_var):
    ari_scores_df, predictions_list, dataframes_list, centroids_list = detect_model_drift(408, rfm_4_var)
    plot_ari_score(ari_scores_df)
    return centroids_list, dataframes_list, predictions_list


@app.cell
def _():
    return


@app.cell
def _(TSNE, pd, plt):
    def plot_clusters_with_tsne(predictions_list, dataframes_list, centroids_list, iteration_index, sample_size=1000):
        """Plot real and predicted clusters using t-SNE for dimensionality reduction.

        Parameters:
        - predictions_list: List of prediction dictionaries.
        - dataframes_list: List of dataframes.
        - centroids_list: List of centroids dictionaries.
        - iteration_index: Index of the iteration to plot.
        - sample_size: Size of the sample to use for t-SNE.
        """
        sample_size = 1000

        # Select a specific iteration
        initial_df = dataframes_list[0]["dataframe"].copy()
        current_df = dataframes_list[iteration_index]["dataframe"].copy()

        current_centroids = centroids_list[iteration_index]["current_centroids"]
        initial_centroids = centroids_list[iteration_index]["initial_centroids"]

        # Assign clusters to individuals
        initial_df["clusters"] = predictions_list[0]["predicted_clusters"]
        current_df["clusters"] = predictions_list[iteration_index]["predicted_clusters"]

        # Select only new data from _current_df compared to initial_df
        new_data_df = current_df.iloc[len(initial_df) :, :]

        # Select a Sample of 800 observation of initial_df and 200 from new data
        initial_df_sampled = initial_df.sample(n=int(sample_size * 0.8), random_state=42)
        new_data_sampled_df = new_data_df.sample(n=int(sample_size * 0.2), random_state=42)

        # Merge the DataFrames
        merged_df = pd.concat([initial_df_sampled, new_data_sampled_df], ignore_index=True)

        # Select a random sample of the data
        initial_df_sampled = initial_df.sample(n=sample_size, random_state=42)
        current_sampled_df = current_df.sample(n=sample_size, random_state=42)

        # Features for t-SNE
        features = merged_df.drop(columns=["clusters"])

        # Apply t-SNE to reduce dimensionality on the sample while preserving the clusters
        tsne = TSNE(n_components=2, random_state=42, perplexity=4)
        sampled_df_tsne = tsne.fit_transform(features)

        # Transform the centroids using the same t-SNE model
        current_centroids_tsne = tsne.fit_transform(current_centroids)
        initial_centroids_tsne = tsne.fit_transform(initial_centroids)

        # Plot the real and predicted clusters with centroids using t-SNE
        # plt.figure(figsize=(12, 6))

        # plt.subplot(1, 2, 1)
        # Plot the first 800 observations
        plt.scatter(
            sampled_df_tsne[:800, 0],
            sampled_df_tsne[:800, 1],
            c=merged_df["clusters"][:800],
            cmap="viridis",
            s=50,
            label="First 800 Observations",
        )

        # Plot the last 200 observations with 'x'
        plt.scatter(
            sampled_df_tsne[800:, 0],
            sampled_df_tsne[800:, 1],
            c=merged_df["clusters"][800:],
            cmap="viridis",
            s=50,
            marker="x",
            label="Last 200 Observations",
        )
        plt.scatter(
            initial_centroids_tsne[:, 0],
            initial_centroids_tsne[:, 1],
            c="red",
            s=200,
            alpha=0.75,
            marker="X",
            label="Centroids",
        )
        plt.title(f"Real Clusters - Week {predictions_list[iteration_index]['week_number']:.0f}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")

        plt.legend()
        plt.show()

    return (plot_clusters_with_tsne,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Voyons si je peux voir une difference dans les clusters formés en utilisant une projection à 2 dimensions.""")
    return


@app.cell
def _(
    centroids_list,
    dataframes_list,
    plot_clusters_with_tsne,
    predictions_list,
):
    iteration_index = 5

    plot_clusters_with_tsne(predictions_list, dataframes_list, centroids_list, iteration_index)
    return (iteration_index,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    L'analyse de la stabilité temporelle des clusters révèle que la similarité entre les segmentations commence à se dégrader significativement à partir de la **huitième semaine**, comme en témoigne la chute du score ARI en dessous du seuil de **0,8**. 

    Cette diminution de stabilité suggère qu'un nouvel entraînement du modèle serait nécessaire à partir de cette période pour maintenir une segmentation pertinente des clients, reflétant ainsi l'évolution de leur comportement.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Comparons la distribution des caractéristiques clients en fonction des périodes initiales par rapport à période_initiale + 6 semaines""")
    return


@app.cell
def _(dataframes_list, iteration_index):
    # initial_df = dataframes_list[0]["dataframe"]
    _current_df = dataframes_list[iteration_index]["dataframe"]
    _current_df
    return


@app.cell(hide_code=True)
def _(dataframes_list, iteration_index, mo):
    # Get initial DataFrame
    initial_df = dataframes_list[0]["dataframe"]

    # Get datframes when algorithm start to be obsolete
    t1_df = dataframes_list[iteration_index]["dataframe"]
    nb_new_customer = len(t1_df) - len(initial_df)
    mo.md(f"L'algorithme commence à devenir obsolète à partir de **{nb_new_customer}** nouveaux clients")
    return initial_df, t1_df


@app.cell
def _(initial_df, plt, sns, t1_df):
    # Créer une figure avec une grille de sous-graphiques
    fig = plt.figure(figsize=(8, 20))
    gs = plt.GridSpec(len(initial_df.columns), 1, figure=fig)

    # Boucle sur les colonnes des DataFrames
    for i, column in enumerate(initial_df.columns):
        # Sous-graphique pour initial_df
        ax1 = fig.add_subplot(gs[i])
        sns.histplot(data=initial_df, x=column, ax=ax1, fill=True, label="données initiale")
        sns.histplot(data=t1_df, x=column, ax=ax1, fill=True, label="données initiale + t1")
        ax1.set_title(f'Distribution KDE de "{column}" - Initial DataFrame')
        ax1.set_xlabel(column)
        ax1.set_ylabel("Densité")
        ax1.legend()
    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, sns):
    def plot_cluster_stats(df, cluster_labels, stats, title=None):
        """Function to plot cluster statistics.

        Parameters:
        - df: DataFrame containing the characteristics (recency, frequency, monetary, etc.)
        - cluster_labels: Cluster labels (array-like)
        - stats: List of statistics to calculate
        """
        # Create a copy of the DataFrame and add cluster labels
        rfm_named_df = df.copy()
        characteristics = rfm_named_df.columns

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

        fig0 = plt.figure(figsize=(5, 4))
        fig0.suptitle(title)
        sns.barplot(x="Clusters", y="Count", data=_stats_results)
        plt.title("Count by Cluster")
        plt.tight_layout()
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
def _(initial_df, plot_cluster_stats, predictions_list):
    plot_cluster_stats(
        initial_df,
        cluster_labels=predictions_list[0]["predicted_clusters"],
        stats=["mean"],
        title="Analyse groupes clients semaine 0",
    )
    return


@app.cell
def _(iteration_index, plot_cluster_stats, predictions_list, t1_df):
    _clusters = predictions_list[iteration_index]["predicted_clusters"]
    plot_cluster_stats(
        t1_df, cluster_labels=_clusters, stats=["mean"], title=f"Analyse groupes clients semaine {iteration_index * 2}"
    )
    return


@app.cell
def _(iteration_index, plot_cluster_stats, predictions_list, t1_df):
    _clusters = predictions_list[iteration_index]["real_clusters"]
    plot_cluster_stats(
        t1_df, cluster_labels=_clusters, stats=["mean"], title=f"Analyse groupes clients semaine {iteration_index * 2}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Les graphiques ci dessous montrent l'évolution de la distribution des informations clients.

    Après 8 semaines, les caractéristiques de chaque groupe ne sont pas franchemnt différentes mais l'assignation des clients dans les groupes adéquats sont inexacts.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Modélisation pour la soutenance""")
    return


@app.cell
def _(KMeans, np, plt):
    from sklearn.datasets import make_blobs

    # Générer des données aléatoires pour le clustering initial
    X_initial, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=0)

    # Créer une instance de KMeans
    kmeans_initial = KMeans(n_clusters=5, random_state=42)

    # Entraîner le modèle sur les données initiales
    kmeans_initial.fit(X_initial)

    # Prédire les clusters pour les données initiales
    initial_clusters = kmeans_initial.predict(X_initial)

    # Récupérer les centroïdes
    centroids_initial = kmeans_initial.cluster_centers_

    # Générer de nouvelles données
    X_new, _ = make_blobs(n_samples=100, centers=5, cluster_std=0.60, random_state=42)

    # Prédire les clusters pour les nouvelles données avec le modèle initial
    new_clusters_initial = kmeans_initial.predict(X_new)

    # Réentraîner le modèle avec les données initiales et les nouvelles données
    X_combined = np.vstack((X_initial, X_new))
    kmeans_retrained = KMeans(n_clusters=5, random_state=42)
    kmeans_retrained.fit(X_combined)

    # Prédire les clusters pour les données combinées avec le modèle réentraîné
    combined_clusters_retrained = kmeans_retrained.predict(X_combined)

    # Récupérer les nouveaux centroïdes
    centroids_retrained = kmeans_retrained.cluster_centers_

    # Tracer les résultats
    plt.figure(figsize=(8, 20))

    # Tracer les données initiales avec le modèle initial
    plt.subplot(4, 1, 1)
    plt.scatter(X_initial[:, 0], X_initial[:, 1], c=initial_clusters, s=50, cmap="viridis", label="Initial Data")
    plt.scatter(centroids_initial[:, 0], centroids_initial[:, 1], c="red", s=200, alpha=0.75, marker="X", label="Centroids")
    plt.title("Clustering Initial avec KMeans")
    plt.xlabel("Caractéristique 1")
    plt.ylabel("Caractéristique 2")
    plt.legend()

    # Tracer les données initiales et les nouvelles données avec le modèle initial
    plt.subplot(4, 1, 2)
    plt.scatter(X_initial[:, 0], X_initial[:, 1], c=initial_clusters, s=50, cmap="viridis", label="Initial Data")
    plt.scatter(X_new[:, 0], X_new[:, 1], c=new_clusters_initial, s=50, cmap="viridis", marker="x", label="New Data")
    plt.scatter(centroids_initial[:, 0], centroids_initial[:, 1], c="red", s=200, alpha=0.75, marker="X", label="Centroids")
    plt.title("Clustering avec Modèle Initial")
    plt.xlabel("Caractéristique 1")
    plt.ylabel("Caractéristique 2")
    plt.legend()

    # Tracer les données combinées avec le modèle réentraîné
    plt.subplot(4, 1, 3)
    plt.scatter(X_combined[:, 0], X_combined[:, 1], c=combined_clusters_retrained, s=50, cmap="viridis", label="Combined Data")
    plt.scatter(centroids_retrained[:, 0], centroids_retrained[:, 1], c="red", s=200, alpha=0.75, marker="X", label="Centroids")
    plt.title("Clustering avec Modèle Réentraîné")
    plt.xlabel("Caractéristique 1")
    plt.ylabel("Caractéristique 2")
    plt.legend()

    # Tracer les centroïdes initiaux et réentraînés pour comparaison
    plt.subplot(4, 1, 4)
    plt.scatter(centroids_initial[:, 0], centroids_initial[:, 1], c="blue", s=200, alpha=0.75, marker="X", label="Initial Centroids")
    plt.scatter(centroids_retrained[:, 0], centroids_retrained[:, 1], c="red", s=200, alpha=0.75, marker="X", label="Retrained Centroids")
    plt.title("Comparaison des Centroïdes")
    plt.xlabel("Caractéristique 1")
    plt.ylabel("Caractéristique 2")
    plt.legend()

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
