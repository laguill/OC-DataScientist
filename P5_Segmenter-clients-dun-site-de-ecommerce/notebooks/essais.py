import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium", app_title="P5 Essais")


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

    La métrique choisit ici pour définir la performance des modèles est le coefficient de silhouette. 

    Ce coefficient mesure la cohérence interne des clusters en comparant:

    - la distance moyenne des points au sein de chaque cluster
    - la distance moyenne entre les clusters.

    A savoir que:  

    > Un coefficient de silhouette de 1 indique des clusters parfaitement distincts, tandis qu'un coefficient de -1 signale des clusters mal définis.

    > Des valeurs proches de 0 suggèrent des clusters qui se chevauchent, et des valeurs négatives indiquent généralement que certains points ont été mal classés, car ils sont plus proches d'un autre cluster.

    > Des valeurs négatives indiquent généralement que certains points ont été mal classés, car ils sont plus proches d'un autre cluster.

    > source: [scikit-learn_ silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score)

    Egalement lors de l'utilisation de l'algorithme des kmean, je vais mesurer la distorsion (la méthode du coude) afin d'identifier le nombre optimal de k voisins.

    > La méthode du coude aide à trouver le nombre optimal de clusters. Pour cela, on trace une courbe montrant comment la variance des points au sein des clusters diminue avec le nombre de clusters.
    > Le "coude" de la courbe indique le point où ajouter plus de clusters ne améliore plus significativement la séparation des données.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Algorithmes**

    Je vais tester les algorithmes de classification non supervisés à savoir K-MEANS et DBSCAN.
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

    import marimo as mo

    # Data viz
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    import seaborn as sns
    import sqlalchemy

    pio.renderers.default = "browser"

    # ML
    from scipy.cluster.hierarchy import dendrogram
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    # Clustering Analysis
    from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

    return KMeans, StandardScaler, mo, pd, plt


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Analyse de la segmentation RFM""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Lors de la phase d'exploration, j'ai appliqué une transformé logarthmique pour transformer la distribution des caractéristiques frequency et monetary. C'est celles que je vais conserver pour mon analyse."""
    )
    return


@app.cell
def _(final_df):
    rfm_df = final_df[["recency", "log_frequency", "log_monetary"]]
    return (rfm_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Définition de l'encodage des features""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""J'applique à présent le StandardScaler pour centrer-réduire les données""")
    return


@app.cell
def _(StandardScaler, rfm_df):
    # Define a StandardScaler object
    scaler = StandardScaler()
    # Use it on our dataframe to get standardized values
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Initialisation de l'algorithme K-Means avec 2 clusters""")
    return


@app.cell
def _(KMeans, StandardScaler, plt):
    def elbow_method(X, max_clusters=10):
        # Standardisation des données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Liste pour stocker les valeurs de WCSS
        wcss = []

        # Calcul de WCSS pour différents nombres de clusters
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        # Tracé de la courbe WCSS
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_clusters + 1), wcss, marker="o")
        plt.title("Méthode du Coude")
        plt.xlabel("Nombre de Clusters")
        plt.ylabel("WCSS")
        plt.show()

    return (elbow_method,)


@app.cell
def _(elbow_method, rfm_df):
    elbow_method(rfm_df, 12)
    return


if __name__ == "__main__":
    app.run()
