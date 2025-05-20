from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import missingno as msno
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

    _ = alt.data_transformers.enable("vegafusion")


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


def display_factorial_planes(
    X_projected, x_y, pca=None, labels=None, clusters=None, alpha=1, figsize=[10, 8], marker="."
):
    """Affiche la projection des individus.

    Positional arguments :
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments :
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8]
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """
    # Transforme X_projected en np.array
    X_ = np.array(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize:
        figsize = (7, 6)

    # On gère les labels
    if labels is None:
        labels = []
    try:
        len(labels)
    except Exception as e:
        raise e

    # On vérifie la variable axis
    if not len(x_y) == 2:
        raise AttributeError("2 axes sont demandées")
    if max(x_y) >= X_.shape[1]:
        raise AttributeError("la variable axis n'est pas bonne")

    # on définit x et y
    x, y = x_y

    # Initialisation de la figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # On vérifie s'il y a des clusters ou non
    c = None if clusters is None else clusters

    # Les points
    # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha,
    #                     c=c, cmap="Set1", marker=marker)
    sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c)

    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe
    if pca:
        v1 = str(round(100 * pca.explained_variance_ratio_[x])) + " %"
        v2 = str(round(100 * pca.explained_variance_ratio_[y])) + " %"
    else:
        v1 = v2 = ""

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f"F{x + 1} {v1}")
    ax.set_ylabel(f"F{y + 1} {v2}")

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() * 1.1
    y_max = np.abs(X_[:, y]).max() * 1.1

    # On borne x et y
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom=-y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color="grey", alpha=0.8)
    plt.plot([0, 0], [-y_max, y_max], color="grey", alpha=0.8)

    # Affichage des labels des points
    if len(labels):
        # j'ai copié collé la fonction sans la lire
        for i, (_x, _y) in enumerate(X_[:, [x, y]]):
            plt.text(_x, _y + 0.05, labels[i], fontsize="14", ha="center", va="center")

    # Titre et display
    plt.title(f"Projection des individus (sur F{x + 1} et F{y + 1})")
    plt


def display_factorial_planes(
    X_projected, x_y, pca=None, labels=None, clusters=None, alpha=1, figsize=[10, 8], marker="."
):
    """Affiche la projection des individus.

    Positional arguments :
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments :
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8]
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """
    # Transforme X_projected en un df
    X_ = pd.DataFrame(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize:
        figsize = (7, 6)

    # On vérifie la variable axis
    if not len(x_y) == 2:
        raise AttributeError("2 axes sont demandées")
    if max(x_y) >= X_.shape[1]:
        raise AttributeError("la variable axis n'est pas bonne")

    # On définit x et y
    x, y = x_y

    # Initialisation de la figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # On rajoute la color, les clusters et les labels à X_
    X_["clusters"] = clusters if clusters is not None else "None"
    X_["labels"] = labels if labels is not None else "None"
    c_unique_list = X_["clusters"].sort_values().unique()
    c_dict = {j: i + 1 for i, j in enumerate(c_unique_list)}
    X_["colors"] = X_["clusters"].apply(lambda i: c_dict[i])

    # Pour chaque couleur / cluster
    for c in sorted(X_.clusters.unique()):
        # On selectionne le sous DF
        sub_X = X_.loc[X_.clusters == c, :]

        # Clusters and color
        cluster = sub_X.clusters.iloc[0]
        color = sub_X.colors.iloc[0]

        # On affiche les points
        ax.scatter(sub_X.iloc[:, x], sub_X.iloc[:, y], alpha=alpha, label=cluster, cmap="Set1", marker=marker)

    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe
    if pca:
        v1 = str(round(100 * pca.explained_variance_ratio_[x])) + " %"
        v2 = str(round(100 * pca.explained_variance_ratio_[y])) + " %"
    else:
        v1 = v2 = ""

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f"F{x + 1} {v1}")
    ax.set_ylabel(f"F{y + 1} {v2}")

    # Valeur x max et y max
    x_max = np.abs(X_.iloc[:, x]).max() * 1.1
    y_max = np.abs(X_.iloc[:, y]).max() * 1.1

    # On borne x et y
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom=-y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color="grey", alpha=0.8)
    plt.plot([0, 0], [-y_max, y_max], color="grey", alpha=0.8)

    # Affichage des labels des points
    if labels:
        # j'ai copié collé la fonction sans la lire
        for i, (_x, _y) in enumerate(X_[:, [x, y]]):
            plt.text(_x, _y, labels[i], fontsize="14", ha="center", va="center")

    # Titre, legend et display
    plt.title(f"Projection des individus (sur F{x + 1} et F{y + 1})")
    if clusters is not None:
        plt.legend()
    plt


def proportion_valeurs_manquantes(data) -> pd.DataFrame:
    """Calcule la proportion de valeurs manquantes par colonne dans un DataFrame.

    Paramètres :
    - data : DataFrame pandas contenant les données à analyser.

    Retourne :
    - valeurs_manquantes_df : DataFrame contenant les colonnes et leurs proportions de valeurs manquantes.
    """
    taux_valeurs_manquantes = data.isna().mean() * 100
    valeurs_manquantes_df = taux_valeurs_manquantes.reset_index()
    valeurs_manquantes_df.columns = ["Colonne", "Proportion"]

    show(valeurs_manquantes_df, "Proportion de valeurs manquantes par colonne")

    return valeurs_manquantes_df


def visualiser_valeurs_manquantes(data, threshold=None):
    """Visualise la proportion de valeurs manquantes par colonne avec un graphique à barres interactif.

    Paramètres :
    - data : DataFrame pandas contenant les données à analyser.
    - threshold : Seuil pour la ligne de référence sur le graphique (optionnel).

    Retourne :
    - chart : Objet Altair Chart représentant le graphique.
    """
    valeurs_manquantes_df = proportion_valeurs_manquantes(data)

    bars = (
        alt.Chart(valeurs_manquantes_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Colonne:N",
                sort="-y",
                title="Colonnes",
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y(
                "Proportion:Q",
                title="Proportion de valeurs manquantes",
                scale=alt.Scale(domain=(0, 100)),
            ),
            color=alt.Color("Proportion:Q")
            .title("Taux valeurs manquantes")
            .scale(
                domain=(0, 100),
                range=["#66c2a5", "#fc8d62"],
            ),
            tooltip=["Colonne", alt.Tooltip("Proportion:Q", format=".2f")],
        )
        .properties(
            title="Proportions de valeurs manquantes par colonne",
            width=500,
            height=150,
        )
        .interactive()
    )

    if threshold is not None:
        threshold_line = (
            alt.Chart()
            .mark_rule(
                color="black",
                strokeDash=[4, 4],
            )
            .encode(y=alt.Y(datum=threshold))
        )

        label = threshold_line.mark_text(
            x="width",
            dx=-2,
            align="right",
            baseline="bottom",
            text=f"Manquants < {threshold}%",
        )

        chart = bars + threshold_line + label
    else:
        chart = bars

    chart.show()


def visualiser_quartiles(data: pd.DataFrame, colonnes: list[str] | None = None, titre: str | None = None):
    """Crée et affiche un boxplot horizontal pour les colonnes numériques spécifiées d'un DataFrame.

    Cette fonction utilise Altair pour générer un graphique interactif avec des tooltips affichant
    le nom de la colonne et la valeur. Si aucune colonne n'est spécifiée, toutes les colonnes
    numériques du DataFrame seront utilisées.

    Paramètres :
    - data (pd.DataFrame) : Le DataFrame contenant les données à visualiser.
    - colonnes (list[str] | None) : La liste des noms de colonnes à inclure dans le boxplot.
                                       Si None, toutes les colonnes numériques seront utilisées.
    - titre (str | None) : Le titre du graphique sinon "Boxplot des colonnes numériques" par défaut.

    Retourne :
    - None : Affiche directement le graphique.
    """
    # Si aucune colonne n'est spécifiée, utiliser toutes les colonnes numériques
    if colonnes is None:
        colonnes = data.select_dtypes(include="number").columns.tolist()

    if titre is None:
        titre = "Boxplot"

    # Fondre les données pour Altair
    data_melted = data[colonnes].melt(var_name="Colonnes", value_name="Valeur")

    # Créer un boxplot horizontal
    boxplot = (
        alt.Chart(data_melted)
        .mark_boxplot()
        .encode(
            x=alt.X("Valeur:Q"),
            y=alt.Y("Colonnes:N"),
            color=alt.Color("Colonnes:N")
            .legend(None)
            .scale(
                scheme="category10",
            ),
        )
        .properties(
            width=600,
            height=400,
            title=titre,
        )
    )

    boxplot.show()


def boxplot_nutriments(data: pd.DataFrame, features: list[str]):
    boxplot: alt.RepeatChart = (
        alt.Chart(data)
        .mark_boxplot()
        .encode(
            x=alt.X(
                alt.repeat("repeat"),
                type="quantitative",
            ),
            y=alt.Y(
                "pnns_groups_1:N",
                title="Groupes de produits",
            ),
            color=alt.Color("pnns_groups_1:N").scale(scheme="category10"),
        )
        .properties(
            width=300,
            height=300,
        )
        .repeat(
            repeat=features,
            columns=2,
        )
        .resolve_scale(x="independent")
    )
    boxplot.show()


def save_intermediate_data(df: pd.DataFrame, filename: str):
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


def save_processed_data(df: pd.DataFrame, filename: str):
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


def histogram_features(data: pd.DataFrame, features: list[str]):
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


def matrix_manquants(data: pd.DataFrame, columns: list[str], title: str):
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


def prepare_data_with_missing_values(df, listkNN, missing_fraction=0.25, sample_fraction=0.25, random_state=10):
    """Prepare a DataFrame by introducing missing values and sampling data for training.

    Parameters:
    - df: pandas DataFrame containing the data.
    - listkNN: List of column names to consider for missing values.
    - missing_fraction: Fraction of missing values to introduce in each column.
    - sample_fraction: Fraction of data to sample for training.
    - random_state: Seed for reproducibility.

    Returns:
    - sample_datas: Sampled DataFrame with missing values.
    """
    # Filtrer les données pour ne garder que les lignes sans valeurs manquantes dans listkNN
    df_cleaned = df.dropna(subset=listkNN).copy()

    # Introduire des valeurs manquantes dans une fraction des données pour chaque colonne de listkNN
    for col in listkNN:
        sample_indices = df_cleaned.sample(frac=missing_fraction, random_state=np.random.randint(1, 100)).index
        df_cleaned.loc[sample_indices, col] = np.nan

    # Échantillonner une fraction des données pour l'entraînement
    sample_datas = df_cleaned[listkNN].sample(frac=sample_fraction, random_state=random_state)

    return sample_datas
