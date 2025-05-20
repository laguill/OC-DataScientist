from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from IPython.display import Markdown, display
from itables import options as opt
from itables import show
from scipy.stats import norm


def set_options() -> None:
    """Configure les options par défaut pour pandas et itables."""
    # Configurer pandas pour éviter l'écriture scientifique
    pd.set_option("display.float_format", "{:g}".format)

    # Configurer les options itables
    opt.scrollY = "200px"
    opt.scrollCollapse = True
    opt.paging = False
    opt.column_filters = "header"


def df_infos(df_title: str, df: pd.DataFrame) -> str:
    """Affiche des informations sur un DataFrame en Markdown.

    Parameters:
    df_title (str): Le titre du jeu de données.
    df (pd.DataFrame): Le DataFrame à analyser.

    Returns:
    str
    """
    info: str = f"""Nom du jeu de données: **{df_title}**<br />
    Doublons: **{df.duplicated().sum()}**<br />
    """
    return info


def df_shape(df: pd.DataFrame) -> str:
    """Markdown du nb ligne et nb colonnes du dataframe."""
    info: str = f"Le fichier  de données contient **{df.shape[0]}** lignes et **{df.shape[1]}** colonnes."
    return info


def correlation_graph(pca: list[str], x_y: tuple, features: list[str]) -> None:
    """Affiche le graphe des correlations.

    Positional arguments :
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """
    # Extrait x et y
    x, y = x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Pour chaque composante :
    for i in range(pca.components_.shape[1]):
        # Les flèches
        ax.arrow(
            0,
            0,
            pca.components_[x, i],
            pca.components_[y, i],
            head_width=0.07,
            head_length=0.07,
            width=0.02,
        )

        # Les labels
        plt.text(pca.components_[x, i] + 0.05, pca.components_[y, i] + 0.05, features[i])

    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color="grey", ls="--")
    plt.plot([0, 0], [-1, 1], color="grey", ls="--")

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel(f"F{x + 1} ({round(100 * pca.explained_variance_ratio_[x], 1)}%)")
    plt.ylabel(f"F{y + 1} ({round(100 * pca.explained_variance_ratio_[y], 1)}%)")

    # Titre du graphe
    plt.title(f"Cercle des corrélations (F{x + 1} et F{y + 1})")

    # Le cercle
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis("equal")
    plt.show(block=False)


def save_intermediate_data(df: pd.DataFrame, filename: str) -> None:
    """Enregistre un DataFrame en format Parquet dans le dossier /data/intermediate/.

    Crée le dossier s'il n'existe pas.

    :param df: DataFrame à enregistrer
    :param filename: Nom du fichier (sans extension)
    """
    path = Path("data/intermediate")
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / f"{filename}.parquet"
    df.to_parquet(file_path, index=False)
    print(f"Données enregistrées dans {file_path}")


def load_intermediate_data(filename: str) -> pd.DataFrame:
    """Charge un fichier Parquet depuis le dossier /data/intermediate/.

    :param filename: Nom du fichier (sans extension)
    :return: DataFrame chargé
    """
    file_path = Path("data/intermediate") / f"{filename}.parquet"

    if not file_path.exists():
        msg = "Fichier introuvable."
        raise FileNotFoundError(msg)

    return pd.read_parquet(file_path)


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Enregistre un DataFrame en format Parquet dans le dossier /data/intermediate/.

    Crée le dossier s'il n'existe pas.

    :param df: DataFrame à enregistrer
    :param filename: Nom du fichier (sans extension)
    """
    path = Path("data/processed")
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / f"{filename}.parquet"
    df.to_parquet(file_path, index=False)
    print(f"Données enregistrées dans {file_path}")


def load_processed_data(filename: str) -> pd.DataFrame:
    """Charge un fichier Parquet depuis le dossier /data/intermediate/.

    :param filename: Nom du fichier (sans extension)
    :return: DataFrame chargé
    """
    file_path = Path("data/processed") / f"{filename}.parquet"

    if not file_path.exists():
        msg = "Fichier introuvable."
        raise FileNotFoundError(msg)

    return pd.read_parquet(file_path)


def histogram_features(data: pd.DataFrame, features: list[str]) -> None:
    palette = iter(sns.color_palette("Set2"))

    sns.set_style("whitegrid")

    # Gestion d'un seul feature
    single_feature = len(features) == 1

    fig, axs = plt.subplots(
        nrows=len(features),
        ncols=1,
        sharex=False,
        figsize=(12, 5 * len(features)),
        layout="constrained",
    )

    # Si un seul graphique, on s'assure que axs est une liste
    if single_feature:
        axs = [axs]

    fig.suptitle(
        "Distribution des valeurs de nutriments",
    )

    for i, feature in enumerate(features):
        serie = data[feature].dropna()
        mean = serie.mean()
        median = serie.median()
        skewness = serie.skew()
        kurtosis = serie.kurtosis()

        sns.histplot(
            serie,
            kde=True,
            ax=axs[i],
            bins=50,
            color=next(palette),
        )

        sns.kdeplot(
            data=serie,
            bw_adjust=0.2,
            ax=axs[i],
        )

        # Ajouter la courbe de distribution normale
        mu, std = norm.fit(serie)
        xmin = serie.min()
        xmax = serie.max()
        x = np.linspace(xmin, xmax, len(serie))

        # Récupérer la largeur des bins et le nombre total de points
        bin_width = (xmax - xmin) / 50  # 50 étant le nombre de bins par défaut de histplot
        total_count = len(serie)

        # Ajuster la courbe gaussienne en count
        normal_dist = norm.pdf(x, mu, std) * total_count * bin_width

        axs[i].plot(
            x,
            normal_dist,
            "--",
            color="black",
            label="Ajustement gaussien",
        )

        axs[i].axvline(
            x=mean,
            color="r",
            linestyle=":",
            linewidth=3,
            label=f"Mean: {mean:.2f}",
        )
        axs[i].axvline(
            x=median,
            color="b",
            linestyle=":",
            linewidth=3,
            label=f"Median: {median:.2f}",
        )

        # Ajouter le texte légèrement en dessous de la légende
        axs[i].annotate(
            text=f"Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}",
            xy=(0.99, 0.75),  # Décalage vers le bas
            xycoords="axes fraction",
            fontsize=10,
            ha="right",
            va="top",
            bbox={
                "facecolor": "white",
                "edgecolor": "lightgrey",
                "boxstyle": "round",
                "pad": 0.3,
            },
        )

        axs[i].legend(loc="upper right", title="Legende:")


def matrix_manquants(data: pd.DataFrame, columns: list[str], title: str) -> None:
    """Visualise les valeurs manquantes dans les colonnes spécifiées à l'aide de missingno.

    Paramètres :
    - data (pd.DataFrame) : Le DataFrame contenant les données des produits.
    - columns (list) : La liste des colonnes pour visualiser pour les valeurs manquantes.
    - title (str): plot title.
    """
    ax = msno.matrix(
        df=data[columns],
        filter="top",
        sort="descending",
        figsize=(8, 6),
        color=(0.49, 0.69, 0.90),
        fontsize=10,
        width_ratios=(8, 3),
    )
    ax.set_title(title)
    plt.ylabel("Row index")
    plt.text(
        x=10,
        y=14,
        s="Sparkline\n nombre de données par ligne",
        ha="center",
        fontsize=10,
    )

    plt.show()


def compte_diff_lignes(
    nb_df1: int,
    nb_df2: int,
) -> None:
    """Calcule la différence de lignes entre 2 dataframe ou series.

    :param nb_df1: Nb du ligne du dataframe1 avant filtrage.
    :param nb_df2: Nb du ligne du dataframe2 après filtrage.

    """
    difference = nb_df1 - nb_df2
    pourcentage_difference = (difference / nb_df1) if nb_df1 > 0 else 0

    msg = f"""**Nb lignes difference:** <br>
    **df1:** {nb_df1} <br>
    **df2:** {nb_df2} <br>
    **Diff.:** {difference} <br>
    **Perc.:** {pourcentage_difference:.2%}
    """
    display(Markdown(msg))


def detect_outliers(df: pd.DataFrame, feature: str, seuil: float) -> pd.DataFrame:
    q1: float = df[feature].quantile(0.25)
    q3: float = df[feature].quantile(0.75)

    iqr: float = q3 - q1
    lower_bound: float = q1 - seuil * iqr
    upper_bound: float = q3 + seuil * iqr

    return df.query(f"`{feature}` < {lower_bound} or `{feature}` > {upper_bound}")


def matrice_correlation(data: pd.DataFrame, columns: list[str], title: str) -> None:
    # Calcul de la matrice de corrélation
    correlation_matrix: pd.DataFrame = data[columns].corr()

    # Mask du triangle supérieur de la matrice
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.set_style("whitegrid")

    # Set up the matplotlib figure
    fig, ax = plt.subplots(
        figsize=(11, 9),
    )

    # Affichage de la matrice de corrélation
    heat_map = sns.heatmap(
        data=correlation_matrix,
        annot=True,
        fmt=".2f",
        mask=mask,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    # Rotation des labels des axes
    heat_map.set_xticklabels(
        heat_map.get_xticklabels(),
        rotation=45,
        horizontalalignment="right",
    )
    heat_map.set(
        xlabel=None,
        ylabel=None,
    )

    # Titre et dimensions de la figure
    ax.set_title(
        title,
    )

    plt.show()
