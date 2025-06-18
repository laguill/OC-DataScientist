import marimo

__generated_with = "0.13.15"
app = marimo.App(
    width="medium",
    app_title="P5 Exploration",
    sql_output="pandas",
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Exploration""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Introduction""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Vous êtes consultant pour [Olist](https://olist.com/), une entreprise brésilienne qui propose une solution de vente sur les marketplaces en ligne.

    **But de la mission :** fournir à l'équipe Marketing d'Olist une segmentation des clients basée sur leur comportement et leurs données personnelles.

    Cette segmentation doit être exploitable et facile d'utilisation pour les équipes Marketing afin de mieux cibler leurs campagnes de communication.

    **Détail de la Mission**

    **Analyse Exploratoire des Données (EDA) :**

    Explorer les données fournies pour comprendre les différentes variables disponibles (historique des commandes, produits achetés, commentaires de satisfaction, localisation des clients).<br>
    Identifier les variables pertinentes pour la segmentation des clients.

    **Modélisation Non Supervisée :**

    Utiliser des techniques de clustering (comme K-Means, DBSCAN, ou Hierarchical Clustering) pour regrouper les clients en segments distincts.<br>
    Évaluer la qualité des clusters obtenus en utilisant des métriques.

    **Description des Segments :**

    Analyser les caractéristiques des différents segments obtenus.<br>
    Fournir une description actionable de chaque segment pour les équipes Marketing.

    **Proposition de Contrat de Maintenance :**

    Réaliser une simulation pour déterminer la fréquence nécessaire de mise à jour du modèle de segmentation afin qu'il reste pertinent.<br>
    Analyser la stabilité des segments au cours du temps pour proposer une recommandation de fréquence de mise à jour.

    **Livrables**

    - Un notebook avec des essais des différentes approches de modélisation
    - Un notebook de simulation pour déterminer la fréquence nécessaire de mise à jour du modèle de segmentation, afin que celui-ci reste pertinent 
    - Une présentation pour un collègue afin d’obtenir ses retours sur votre approche

    ⚠️ Pour information, le code fourni doit respecter la convention PEP8, pour être utilisable par Olist.
    Ce qui signifie :

    - Respecter une indentation de 4 espaces
    - Les lignes de code ne dépassent pas 79 caractères
    - Les imports sont déclarés au début du script
    - Les commentaires sont rédigés en anglais

    La liste des règles est non exhaustive. Pour me faciliter la tâche, j’utilise le linter ruff.
    """  # noqa: RUF001
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Des informations plus détaillées de la base de données sont disponibles sur Kaggle : [https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Le texte sur les connecteurs indique les clés primaires et étrangères qui relient les tables de la base de données.""")
    return


@app.cell(hide_code=True)
def _(mo):
    _src = "notebooks/public/relation_db.png"
    mo.image(src=_src, rounded=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Initialisation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Importation des librairies""")
    return


@app.cell
def _():
    # Import librairies
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    import seaborn as sns
    import sqlalchemy

    pio.renderers.default = "browser"

    from ydata_profiling import ProfileReport

    # load seaborn
    sns.set_theme()
    return Path, ProfileReport, np, pd, plt, px, sns, sqlalchemy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Connexion à la base de données""")
    return


@app.cell
def _(sqlalchemy):
    # Etablishing connection
    DATABASE_URL = "sqlite:///data/raw/olist.db"
    engine = sqlalchemy.create_engine(DATABASE_URL)
    return (engine,)


@app.cell
def _(engine, mo, sqlite_master):
    # Stocking table list in a variable for further use
    tables_df = mo.sql(
        """
        SELECT name FROM sqlite_master
        WHERE type='table'
        """,
        engine=engine,
    )
    # Store tables names in a list
    tables_list = tables_df["name"].to_list()
    return (tables_list,)


@app.cell(disabled=True)
def _(engine, mo, tables_list):
    # Dict to store tables dataframes
    dataframes = {}

    # Store each tables in a dataframe
    for _table in tables_list:
        query = f"SELECT * FROM '{_table}'"  # noqa: S608
        dataframes[_table] = mo.sql(
            query,
            output=False,
            engine=engine,
        )
    return (dataframes,)


@app.cell
def _(mo):
    button = mo.ui.run_button("warn", tooltip="Click to run expensive cells")
    button
    return (button,)


@app.cell
def _(ProfileReport, button, dataframes, mo):
    # Explore tables using ydata-profiling
    mo.stop(not button.value, "Click the button to continue")

    for _table in dataframes:
        _profile = ProfileReport(
            dataframes[_table],
            title=f"{_table} Profiling Report",
            explorative=True,
        )
        _profile.to_file(f"notebooks/public/ydata_reports/{_table}_report.html")
    return


@app.cell(hide_code=True)
def _(Path, mo):
    # List all files in public/data/
    data_dir = Path("notebooks/public/ydata_reports/")
    files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix == ".html"]

    md_links = "\n".join(f"- [{file.name}](notebooks/public/ydata_profiles/{file.name})" for file in files)
    mo.md(md_links)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    L'analyse exploratoires des tables de la base données me permet pour le moment d'obtenir ces informations:

    - 9 tables dans la base de données
    - 99 441 commandes (table **customers**)
        - pas de doublons dans les commandes (customer_id)
        - pas de valeurs manquantes
        - customer_unique_id = 96096. Des clients ont repassé commande sur la plateforme
        - présence de doublons dans customer_unique_id
        - Un client à commandé 17 fois sur la plateforme
    - types de payments
        - **orders_pymts**
        - 5 types (card debit/credit, voucher, etc...)
    - tables **orders**
        - delivered_carrier_date
        - delivered_customer_date
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Table customers""")
    return


@app.cell
def _(customers, engine, mo):
    _df = mo.sql(
        f"""
        WITH duplicate_ids AS (
            SELECT customer_unique_id
            FROM customers
            GROUP BY customer_unique_id
            HAVING COUNT(*) > 1
        )
        SELECT c.*
        FROM customers c
        JOIN duplicate_ids d ON c.customer_unique_id = d.customer_unique_id
        ORDER BY c.customer_unique_id;
        """,
        engine=engine
    )
    return


@app.cell
def _(customers, engine, mo):
    _df = mo.sql(
        f"""
        -- get number of time a customer used Olist
        SELECT
            customer_unique_id,
            COUNT(*) as count
        FROM
            customers
        GROUP BY
            customer_unique_id
        HAVING
            COUNT(*) > 1
        ORDER BY
            count DESC
        LIMIT
            5
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Table order_pymts""")
    return


@app.cell
def _(engine, mo, order_pymts):
    _df = mo.sql(
        f"""
        WITH
            duplicate_orders AS (
                SELECT
                    order_id
                FROM
                    order_pymts
                GROUP BY
                    order_id
                HAVING
                    COUNT(*) > 1
            )
        SELECT
            o.*
        FROM
            order_pymts o
        WHERE
            o.order_id IN (
                SELECT
                    order_id
                FROM
                    duplicate_orders
            )
        ORDER BY
            o.order_id,
            payment_sequential
        LIMIT
            15
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Préparation des données""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    L'objectif est d'analyser le comportements des clients qui interagissent avec la plateforme.

    Je commence par définir une base de temps pour commencer mon étude.

    ### Définir une Référence Temporelle

    La date de référence est cruciale pour plusieurs calculs :

    - **Commandes des 6 derniers mois** : Concentre l'analyse sur les données récentes.
    - **Clients avec 3 mois d’ancienneté ou plus** : Permet de cibler les clients fidèles.
    - **Recency dans RFM** : Mesure la récence des commandes pour chaque client.
    """  # noqa: RUF001
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""On calcule la dernière date de commande dans la table orders.""")
    return


@app.cell(hide_code=True)
def _(engine, mo, orders):
    # Store last order date in a variable
    last_order_date_df = mo.sql(
        """
        WITH
            LastOrderDate AS (
                SELECT
                    MAX(order_purchase_timestamp) AS last_order_date
                FROM
                    orders
            )
        SELECT
            last_order_date
        FROM
            LastOrderDate
        """,
        engine=engine,
    )
    last_order_date = last_order_date_df.iloc[0, 0]
    return (last_order_date,)


@app.cell
def _(engine, mo, orders):
    _df = mo.sql(
        f"""
        WITH
            LastOrderDate AS (
                SELECT
                    MAX(order_purchase_timestamp) AS last_order_date
                FROM
                    orders
            )
        SELECT
            last_order_date
        FROM
            LastOrderDate
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""J'identifie les commandes des 6 derniers mois.""")
    return


@app.cell
def _(engine, last_order_date, mo):
    last_6_month_orders_df = mo.sql(
        f"""
        WITH
            Last6MonthOrders AS (
                SELECT
                    *
                FROM
                    orders
                WHERE
                    order_purchase_timestamp >= DATE('{last_order_date}', "-6 months")
            )
        SELECT
            *
        FROM
            Last6MonthOrders;
        """,
        engine=engine
    )
    return


@app.cell
def _(engine, mo):
    _df = mo.sql(
        f"""
        WITH
            -- get last order date
            LastOrderDate AS (
                SELECT
                    MAX(order_purchase_timestamp) AS last_order_date
                FROM
                    orders
            ),
            -- get last 6 month orders
            Last6MonthOrders AS (
                SELECT
                    *
                FROM
                    orders
                WHERE
                    order_purchase_timestamp >= DATE(
                        (
                            SELECT
                                *
                            FROM
                                LastOrderDate
                        ),
                        "-6 months"
                    )
            )
        SELECT
            *
        FROM
            Last6MonthOrders
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Liste les clients qui utilsent la plateforme depuis 3 mois ou plus ➡️ clients qui ont passé commande il y a plus de 3 mois.""")
    return


@app.cell
def _(engine, mo):
    oldest_client_df = mo.sql(
        f"""
        WITH
            -- get last order date
            LastOrderDate AS (
                SELECT
                    MAX(order_purchase_timestamp) AS last_order_date
                FROM
                    orders
            )

        SELECT
            c.customer_unique_id,
            o.customer_id,
            MIN(o.order_purchase_timestamp) as date_inscription
        FROM
            orders AS o
        JOIN
            customers AS c ON o.customer_id = c.customer_id
        GROUP BY
            c.customer_unique_id
        HAVING
            date_inscription <= DATE(
                (
                    SELECT
                        last_order_date
                    FROM
                        LastOrderDate
                ),
                "-3 months"
            )
        ORDER BY
            date_inscription DESC
        """,
        engine=engine
    )
    return (oldest_client_df,)


@app.cell
def _(oldest_client_df, pd, plt, sns):
    oldest_client_df["date_inscription"] = pd.to_datetime(oldest_client_df["date_inscription"])

    # Extract year and month from signin date
    oldest_client_df["year_month"] = oldest_client_df["date_inscription"].dt.to_period("M")

    # Count number of client per mont/year
    aggregated_data = oldest_client_df["year_month"].value_counts().sort_index()

    # Create a new dataframe
    aggregated_df = aggregated_data.reset_index()
    aggregated_df.columns = ["year_month", "count"]

    # display plot
    sns.barplot(x="year_month", y="count", data=aggregated_df)
    plt.xticks(rotation=45)
    plt.title("Nombre de clients par mois et année d'inscription")
    plt.xlabel("Mois et Année d'inscription")
    plt.ylabel("Nombre de clients")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Le calcul de la **RFM** (**Récence**, **Fréquence**, **Montant**) d'une base de données client permet de segmenter les clients en fonction de leur comportement d'achat pour cibler les campagnes marketing de manière plus efficace.

    - **Récence** : Temps écoulé depuis le dernier achat du client.
    - **Fréquence** : Nombre total d'achats effectués par le client.
    - **Montant** : Valeur totale dépensée par le client.

    J'ai besoin:

    - **Date de la dernière commande** : Utilisation d'une CTE sur la table **orders** avec la colonne **order_purchase_timestamp**.
    - **Recency** : Calcul des jours depuis la dernière commande pour chaque client, utilisant les tables **orders** et customers avec les colonnes **customer_unique_id** et **order_purchase_timestamp**.
    - **Frequency** : Calcul du nombre total de commandes par client, utilisant les tables **orders** et **customers** avec les colonnes **customer_unique_id** et **order_id**.
    - **Monetary** : Calcul du montant total dépensé par chaque client, utilisant les tables **orders**, **customers**, et **order_pymts** avec les colonnes **customer_unique_id**, **order_id**, et **payment_value**.
    - **Combinaison des résultats** : Fusion des CTEs pour obtenir le tableau RFM trié par fréquence décroissante, utilisant **customer_unique_id**, **recency**, **frequency**, et **monetary**.

    J'itère en écrivant d'abord séparement les requêtes pour ensuite les fusionner dans une requête finale.
    """
    )
    return


@app.cell
def _(customers, engine, mo):
    _df = mo.sql(
        f"""
        -- Get total number of orders per client
        WITH frequence AS (
            SELECT
                customer_unique_id,
                COUNT(*) AS total_orders
            FROM
                customers
            GROUP BY
                customer_unique_id
        )

        -- Join result to original table
        SELECT
            c.customer_unique_id,
            f.total_orders
        FROM
            customers c
        JOIN
            frequence AS f
        ON
            c.customer_unique_id = f.customer_unique_id;
        """,
        engine=engine
    )
    return


@app.cell
def _(customers, engine, mo, order_pymts, orders):
    rfm_df = mo.sql(
        f"""
        WITH
            -- get last order date
            LastOrderDate AS (
                SELECT
                    MAX(order_purchase_timestamp) AS last_order_date
                FROM
                    orders
            ),
            -- Calculate recency (nombre de jours depuis le dernier achat)
            Recency AS (
                SELECT
                    c.customer_unique_id,
                    ROUND(
                        JULIANDAY (
                            (
                                SELECT
                                    last_order_date
                                FROM
                                    LastOrderDate
                            )
                        ) - JULIANDAY (MAX(o.order_purchase_timestamp))
                    ) AS recency
                FROM
                    orders o
                    JOIN customers c ON o.customer_id = c.customer_id
                GROUP BY
                    c.customer_unique_id
            ),
            -- Calculate frequency (nombre total de commandes)
            Frequency AS (
                SELECT
                    c.customer_unique_id,
                    COUNT(o.order_id) AS frequency
                FROM
                    orders o
                    JOIN customers c ON o.customer_id = c.customer_id
                GROUP BY
                    c.customer_unique_id
            ),
            -- Calculate monetary (montant total dépensé)
            Monetary AS (
                SELECT
                    c.customer_unique_id,
                    SUM(op.payment_value) AS monetary
                FROM
                    orders o
                    JOIN customers c ON o.customer_id = c.customer_id
                    JOIN order_pymts op ON o.order_id = op.order_id
                GROUP BY
                    c.customer_unique_id
            )
            -- Combine results of CTEs to get RFM's table
        SELECT
            r.customer_unique_id,
            r.recency,
            f.frequency,
            m.monetary
        FROM
            Recency r
            JOIN Frequency f ON r.customer_unique_id = f.customer_unique_id
            JOIN Monetary m ON r.customer_unique_id = m.customer_unique_id
        ORDER BY
            f.frequency DESC
        """,
        engine=engine
    )
    return (rfm_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Analyse de la segmentation RFM""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Voyons la distribution des caractéristiques de ce nouveau dataframe.
    Pour simplifier la tâche je crée une fonction.
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    def feature_analysis(dataframe, feature):
        """Analyse statistique d'une colonne numérique d'un DataFrame."""
        # Check if the feature exists in the dataframe
        if feature not in dataframe.columns:
            msg = f"La colonne '{feature}' n'existe pas dans le dataframe."
            raise ValueError(msg)
        # Check if the feature is numeric, otherwise we can't get the statistics
        if not pd.api.types.is_numeric_dtype(dataframe[feature]):
            return f"La colonne '{feature}' doit contenir des données numériques."

        # Create the list of statistics
        statistics = {
            "Min": dataframe[feature].min(),
            "Max": dataframe[feature].max(),
            "Mode": dataframe[feature].mode()[0],
            "Moyenne": dataframe[feature].mean(),
            "Mediane": dataframe[feature].median(),
            "Variance": dataframe[feature].var(ddof=1),
            "Ecart type": dataframe[feature].std(),
            "Skewness": dataframe[feature].skew(),
            "Kurtosis": dataframe[feature].kurtosis(),
        }

        # Convert the list to DataFrame
        return pd.DataFrame.from_dict(statistics, orient="index", columns=["Value"])
    return (feature_analysis,)


@app.cell
def _(feature_analysis, rfm_df):
    feature_analysis(rfm_df, "recency")
    return


@app.cell
def _(sns):
    # Définir une palette de couleurs
    colors = sns.color_palette("Set2", 3)
    return (colors,)


@app.cell(hide_code=True)
def _(colors, plt, rfm_df, sns):
    # Sélectionner la feature
    _feature = "recency"
    _sub_df = rfm_df[_feature]
    _color = colors[0]

    # Créer une figure avec 3 subplots
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 5))

    # Histogramme
    sns.histplot(_sub_df, bins=15, color=_color, ax=_axs[0])
    _axs[0].set_title("Histogramme")
    _axs[0].set_xlabel(_feature)
    _axs[0].set_ylabel("Fréquence")

    # KDE Plot
    sns.kdeplot(_sub_df, fill=True, color=_color, ax=_axs[1])
    _axs[1].set_title("KDE Plot")
    _axs[1].set_xlabel(_feature)
    _axs[1].set_ylabel("Densité")

    # Boxplot et Stripplot
    # sns.boxenplot(_sub_df, color=_color, fill=False, orient="h", ax=_axs[2])
    # sns.boxplot(_sub_df, color=_color, fill=False, orient="h", ax=_axs[2])
    # sns.stripplot(_sub_df, color=_color, orient="h", ax=_axs[2], alpha=0.9, jitter=0.3)
    sns.violinplot(_sub_df, color=_color, orient="h", ax=_axs[2], inner="quart", bw_adjust=0.5)
    _axs[2].set_title("Violin Plot")
    _axs[2].set_xlabel(_feature)
    _axs[2].set_ylabel("Valeurs")

    # Ajouter un titre global
    _fig.suptitle(f"Analyse de la feature: {_feature}", fontsize=16)

    # Ajuster les espaces entre les subplots
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Afficher la figure
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Le site est actif: Commandes dans les 330 derniers jours
    - Progression du nombre de commande depuis les 360 derniers jours et se maintient.
    """
    )
    return


@app.cell
def _(feature_analysis, rfm_df):
    feature_analysis(rfm_df, "frequency")
    return


@app.cell(hide_code=True)
def _(colors, plt, rfm_df, sns):
    # Sélectionner la feature
    _feature = "frequency"
    _sub_df = rfm_df[_feature]
    _color = colors[1]

    # Créer une figure avec 3 subplots
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 5))

    # Histogramme
    sns.histplot(_sub_df, bins=15, color=_color, stat="probability", ax=_axs[0])
    _axs[0].set_title("Histogramme")
    _axs[0].set_xlabel(_feature)
    _axs[0].set_ylabel("Fréquence")

    # KDE Plot
    sns.kdeplot(_sub_df, fill=True, color=_color, ax=_axs[1])
    _axs[1].set_title("KDE Plot")
    _axs[1].set_xlabel(_feature)
    _axs[1].set_ylabel("Densité")

    # Boxplot et Stripplot
    # sns.boxenplot(_sub_df, color=_color, fill=False, orient="h", ax=_axs[2])
    # sns.boxplot(_sub_df, color=_color, fill=False, orient="h", ax=_axs[2])
    # sns.stripplot(_sub_df, color=_color, orient="h", ax=_axs[2], alpha=0.9, jitter=0.3)
    sns.violinplot(_sub_df, color=_color, orient="h", ax=_axs[2], inner="quart", bw_adjust=0.5)
    _axs[2].set_title("Violin Plot")
    _axs[2].set_xlabel(_feature)
    _axs[2].set_ylabel("Valeurs")

    # Ajouter un titre global
    _fig.suptitle(f"Analyse de la feature: {_feature}", fontsize=16)

    # Ajuster les espaces entre les subplots
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Afficher la figure
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Les clients semblent passer commande qu'une seule fois.""")
    return


@app.cell(hide_code=True)
def _(mo, rfm_df):
    # Get total number of customer that ordered more than once
    _clients_plusieurs_commandes = rfm_df.query("frequency > 1").shape[0]

    # Get total number of customer
    _total_clients = rfm_df.shape[0]

    # Get ratio in percent
    _pourcentage = (_clients_plusieurs_commandes / _total_clients) * 100

    mo.md(f"""Nombre de client qui ont commandé plus d'une fois sur Olist: **{_pourcentage:.0f}%**""")
    return


@app.cell
def _(np, rfm_df, sns):
    # log transform to minimize importance of high values
    rfm_df["log_frequency"] = np.log(rfm_df["frequency"] + 1)

    sns.histplot(rfm_df["log_frequency"], bins=17)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Peu de fidélité dans les utilisateurs de la plateforme.""")
    return


@app.cell
def _(feature_analysis, rfm_df):
    feature_analysis(rfm_df, "monetary")
    return


@app.cell(hide_code=True)
def _(colors, plt, rfm_df, sns):
    # Sélectionner la feature
    _feature = "monetary"
    _sub_df = rfm_df[_feature]
    _color = colors[2]

    # Créer une figure avec 3 subplots
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 5))

    # Histogramme
    sns.histplot(_sub_df, bins=15, color=_color, ax=_axs[0])
    _axs[0].set_title("Histogramme")
    _axs[0].set_xlabel(_feature)
    _axs[0].set_ylabel("Fréquence")

    # KDE Plot
    sns.kdeplot(_sub_df, fill=True, color=_color, ax=_axs[1])
    _axs[1].set_title("KDE Plot")
    _axs[1].set_xlabel(_feature)
    _axs[1].set_ylabel("Densité")

    # Boxplot et Stripplot
    # sns.boxenplot(_sub_df, color=_color, fill=False, orient="h", ax=_axs[2])
    # sns.boxplot(_sub_df, color=_color, fill=False, orient="h", ax=_axs[2])
    # sns.stripplot(_sub_df, color=_color, orient="h", ax=_axs[2], alpha=0.9, jitter=0.3)
    sns.violinplot(_sub_df, color=_color, orient="h", ax=_axs[2], inner="quart", bw_adjust=0.5)
    _axs[2].set_title("Violin Plot")
    _axs[2].set_xlabel(_feature)
    _axs[2].set_ylabel("Valeurs")

    # Ajouter un titre global
    _fig.suptitle(f"Analyse de la feature: {_feature}", fontsize=16)

    # Ajuster les espaces entre les subplots
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Afficher la figure
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Le panier médian est d'environ 105€.""")
    return


@app.cell(hide_code=True)
def _(np, plt, rfm_df, sns):
    # log transform to minimize importance of high values
    rfm_df["log_monetary"] = np.log(rfm_df["monetary"] + 1)

    sns.histplot(rfm_df["log_monetary"], bins=15)
    plt.title("Montant total dépensé par client (log)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Appliquer les deux transformations logarithmique (**frequence**, **montant_moyen**) permettent de centrer la distribution des données.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Satisfaction client""")
    return


@app.cell(hide_code=True)
def _(customers, engine, mo, order_reviews, orders):
    avg_review_df = mo.sql(
        f"""
        WITH OrderCustomer AS (
            -- Join btwn orders and customers to get customer_unique_id et order_id
            SELECT
                o.order_id,
                c.customer_unique_id
            FROM
                orders o
            JOIN
                customers c ON o.customer_id = c.customer_id
        ),

        ReviewScores AS (
            -- Join with order_reviews to get review_score
            SELECT
                oc.customer_unique_id,
                orv.review_score
            FROM
                OrderCustomer oc
            JOIN
                order_reviews orv ON oc.order_id = orv.order_id
        )

        -- Calculate avg review score per customer_unique_id
        -- Also get count of reviews per customer_unique_id
        SELECT
            customer_unique_id,
            ROUND(AVG(review_score)) AS average_review_score,
            COUNT(review_score) AS nb_reviews
        FROM
            ReviewScores
        GROUP BY
            customer_unique_id;
        """,
        engine=engine
    )
    return (avg_review_df,)


@app.cell(hide_code=True)
def _(avg_review_df, np, plt, sns):
    sns.histplot(avg_review_df["average_review_score"], discrete=True, stat="probability", cumulative=True)
    plt.xticks(np.arange(1, 6, 1))
    plt.title("Note moyenne de statisfaction par client")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Presque **80%** des clients de Olist ont donnée une note de 4 ou 5 lors de l'enquête de satisfaction.

    Les clients sont satisfait de la plateforme.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Info details
        type: info

    Je constate que le nombre des reviews est inférieur aux nombres de commandes.
    En cas de jointure il faudra que je pense à imputer ou écarter des données de mon étude.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Géographie""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""J'analyse la destination des colis envoyés par Olist.""")
    return


@app.cell
def _(engine, geoloc, mo):
    geoloc_df = mo.sql(
        f"""
        SELECT
            *
        FROM
            geoloc
        GROUP BY
            geolocation_zip_code_prefix
        """,
        engine=engine
    )
    return (geoloc_df,)


@app.cell
def _(geoloc_df, px):
    _geoloc_df = geoloc_df.sort_values(by="geolocation_state")
    fig = px.scatter_map(
        _geoloc_df,
        lat="geolocation_lat",
        lon="geolocation_lng",
        color="geolocation_state",
        center={"lat": _geoloc_df["geolocation_lat"].mean(), "lon": _geoloc_df["geolocation_lng"].mean()},
        zoom=0,
        height=500,
        map_style="carto-positron",
        title="Carte des Etats receptionant les colis Olist",
    )
    fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    La majorité des colis sont expédiés au Brésil.

    Certains clients se trouvent également au Portugal et au Cap Vert.

    Je pense que l'information du customer_zip_code_prfix peut être intéressante pour segmenter les clients de Olist.

    J'utiliserai l'information de la table customers.
    """
    )
    return


@app.cell
def _(customers, engine, mo, orders):
    customer_zip_code_df = mo.sql(
        f"""
        -- get latest order for each customer
        WITH LatestOrder AS (
            SELECT
                c.customer_unique_id,
                c.customer_zip_code_prefix,
                MAX(o.order_purchase_timestamp) as latest_order_timestamp
            FROM
                customers c
            JOIN
                orders o ON c.customer_id = o.customer_id
            GROUP BY
                c.customer_unique_id
        )

        -- filter latest order table to get only customer_unique_id and customer_zip_code_prefix
        SELECT
            customer_unique_id,
            customer_zip_code_prefix AS zip_code
        FROM
            LatestOrder;
        """,
        engine=engine
    )
    return (customer_zip_code_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Fusion des jeux de données et exportation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    L'exploration de la base de donnée m'a permis d'identifier des caractéristiques clients qui me semblent utile pour segmenter les clients de Olist.

    - **Comportements**
        - recency
        - frequency
        - monetary
        - log(frequency)
        - log(monetary)

    - **Satisfaction**
        - nb_reviews
        - average_review_score

    - **Geographie**
        - zip_code
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Je fusionne les dataframes pandas avant de les enregistrer dans une base de données""")
    return


@app.cell
def _(avg_review_df, customer_zip_code_df, pd, rfm_df):
    # Merge btwn rfm_df and avg_review_df
    temp_df = pd.merge(rfm_df, customer_zip_code_df, on="customer_unique_id", how="left")

    # Second merge btwn previous result and avg_review_df
    db_finale = pd.merge(temp_df, avg_review_df, on="customer_unique_id", how="left")

    # display final df
    db_finale
    return (db_finale,)


@app.cell
def _(db_finale, plt, sns):
    sns.barplot(db_finale[["recency", "frequency", "monetary", "average_review_score"]].isna().sum())
    plt.xticks(rotation=45)
    plt.show()
    return


@app.cell
def _(db_finale):
    # Save df to csv for latter use
    db_finale.to_csv("data/processed/finale_df.csv")
    return


if __name__ == "__main__":
    app.run()
