import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


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

    - Groupe 0: 12% des clients qui sont insatisfaits du service de Olist et ne recommenderont probablement pas (Frequency =1)
    - Groupe 1: 20% de clients qui dépensent plus que la moyenne mais ne sont pas fidélisés (Monetary > 200 et Frequency =1)
    - Groupe 2: clients récurents (Frequency = 2) mais représentent moins de 1% de la totalité
    - Groupe 3: Anciens clients qui pourraient revenir car satisfait (review = 5)
    - Groupe 4: 30% des clients des clients sont nouveaux et dépensent peu (recency < 200 et monetary = 69€)

    L'obejctif étant de cibler les groupes de clients par des campagnes marketing ciblées.
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

    Je me fixe un seuil de **0.8** en dessous duquel, le modèl de ML n'est plus adapté aux informations clients et des ajustements sont nécessaires.
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
    - Comparer les segementation une à une à l'aide de l'indice ARI
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
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.neighbors import NearestNeighbors

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import adjusted_rand_score

    # Clustering Analysis
    from yellowbrick.cluster import KElbowVisualizer, silhouette_visualizer
    return (
        ColumnTransformer,
        FunctionTransformer,
        KMeans,
        Pipeline,
        StandardScaler,
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
    rfm_4_var = finale_df_filtered[["recency", "frequency", "monetary", "average_review_score"]].sort_values(by="recency")
    return (rfm_4_var,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Initie un StandardScaler pour normaliser les données et l'algorithme Kmeans.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Divise le jeu de données pour avoir une dvision de clients tous les mois.""")
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
    - Toutes les colonnes sont ensuites centrées-réduites avec un **StandardScaler**
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
    ### Etape 2: Entrainer un modèle initial
    Je commence par entrainer le modèle de clustering avec les informations clients tout au long de la première année d'existence de Olist.

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
    days_increment_slider = mo.ui.slider(start=7, stop=7 * 3, value=7, step=7, label="Recency window size")
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
        current_date = start_date  # Initialisation de la date courante

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

            # Calculate the Adjusted Rand Index (ARI) score
            ari_score = adjusted_rand_score(real_clusters, predicted_clusters)
            ari_scores[current_week] = ari_score

            # Update the current date and week number
            current_date -= days_increment
            current_week += days_increment / 7  # Update week number

        # Convert the dictionary to a pandas DataFrame
        ari_scores_df = pd.DataFrame(list(ari_scores.items()), columns=["Week_Number", "ARI_Score"])

        # Return the list of ARI scores and the predicted clusters
        return ari_scores_df
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
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.legend()
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.show()
    return (plot_ari_score,)


@app.cell
def _(detect_model_drift, plot_ari_score, rfm_4_var):
    ari_scores_df = detect_model_drift(408, rfm_4_var)
    plot_ari_score(ari_scores_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    L'analyse de la stabilité temporelle des clusters révèle que la similarité entre les segmentations commence à se dégrader significativement à partir de la **sixième semaine**, comme en témoigne la chute du score ARI en dessous du seuil de **0,8**. 

    Cette diminution de stabilité suggère qu'un réentraînement du modèle serait nécessaire à partir de cette période pour maintenir une segmentation pertinente des clients, reflétant ainsi l'évolution de leur comportement.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Comparons la distribution des caractéristiques clients en fonction des périodes initiles par rapport à période_initiale + 6 semaines"""
    )
    return


@app.cell
def _(rfm_4_var):
    # Parameters
    t_0 = rfm_4_var["recency"].max() - 365
    t_1 = t_0 + (46 * 7)

    # Calculate the initial DataFrame
    # Filter the DataFrame to include only rows with recency greater than or equal to start_date
    initial_df = rfm_4_var[rfm_4_var["recency"] >= t_0].copy()

    # Filter the DataFrame to include only rows with recency greater than or equal to current_date
    t1_df = rfm_4_var[rfm_4_var["recency"] >= t_1].copy()
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
        sns.kdeplot(data=initial_df, x=column, ax=ax1, fill=True, label="données initiale")
        sns.kdeplot(data=t1_df, x=column, ax=ax1, fill=True, label="données initiale + t1")
        ax1.set_title(f'Distribution KDE de "{column}" - Initial DataFrame')
        ax1.set_xlabel(column)
        ax1.set_ylabel("Densité")
        ax1.legend()
    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Les graphiques ci dessous montrent l'évolution de la distribution des informations clients.

    La notation client est bien différente après 50 semaines que par rapport à la période d'entrainement du premier model.
    """
    )
    return


if __name__ == "__main__":
    app.run()
