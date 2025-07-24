import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium", app_title="P7 Modelisation")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Implémenter un modèle de scoring""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Introduction
    Définition du scope:

    - **Entreprise** : Prêt à dépenser
    - **Activité** : Proposition de crédit à la consommation (Personne ayant peu ou pas du tout d’historique de prêt)
    Fonction : Data Scientist
    - **Mission** :

    Mettre en place un outil de "scoring crédit"

    - Objectif :

    - Calculer la probabilité qu’un client rembourse.
    - Classifier la demande en crédit accordé ou refusé.
    - Classifier la demande de crédit en type accordée ou refusée (binaire)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **MLflow** propose une API et une interface web pour suivre les expérimentations, enregistrer les paramètres, métriques, artefacts et versions de code. 

    Cela permet de comparer les performances des modèles, de gérer les versions et de déployer les modèles via un Model Registry centralisé

    **Streamlit** offre une interface interactive locale pour tester l’API de prédiction, complémentaire à un déploiement cloud gratuit sur Render.

    Le **MLOps** (Machine Learning Operations) est une discipline visant à industrialiser la gestion du cycle de vie des modèles de machine learning, du développement à la production, en intégrant la collaboration, la reproductibilité, la gestion des versions et le déploiement automatisé.
    """
    )
    return


@app.cell
def _():
    # === Built-in ===
    import os
    import shutil
    import time
    import warnings

    from datetime import datetime
    from pathlib import Path

    import marimo as mo

    # === Suppression des warnings ===
    warnings.filterwarnings("ignore")

    # === Data manipulation ===
    import pandas as pd
    import numpy as np

    # === Visualization ===
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", palette="Set2")
    pal = sns.color_palette("Set2")

    # === Scikit-learn ===
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

    # === Imbalanced-learn ===
    from sklearnex import patch_sklearn

    patch_sklearn()

    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline

    # === MLflow ===
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    # === Optuna ===
    import optuna
    from optuna.integration.mlflow import MLflowCallback

    # === SHAP ===
    import shap

    # === LightGBM ===
    from lightgbm import LGBMClassifier

    # === Scipy ===
    from scipy.sparse import csr_matrix
    from scipy.stats import randint, uniform

    # === Evidently ===
    # from evidently.future.datasets import Dataset
    # from evidently.future.datasets import DataDefinition
    # from evidently.future.report import Report
    return LabelEncoder, OneHotEncoder, Path, mlflow, mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Configuration de MLFlow""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Démarrage de MLFlow
        type: Info

        Pour démarrer MLFlow, j'ai créer un script bash et ajouter le script comme dépendance de mon projet avec uv.

        ```bash
        #!/bin/bash
        uv run mlflow server \
        --host 0.0.0.0 \
        --port 5000 \
        --backend-store-uri sqlite:///data/mlflow/db/mlflow.db \
        --default-artifact-root ./data/mlflow/artifacts
        ```

        Le script est rendu exécutable avec la commande 
        ```bash
        chmod +x scripts/start_mlflow.sh
        ```

    ///

    /// admonition | Utilisation de just
        type: Info

        Just est une alternative à Make pour automatiser l'execution de commandes bash.
        Dans un premier temps, il faut s'assurer que just est installé sur le systeme.
        Puis un créer un fichier justfile.

        ```bash
        # justfile

        start-mlflow:
            uv run bash scripts/start_mlflow.sh
        ```

        Exécuter le script MLFlow via just :

        ```bash
        just start-mlflow
        ```

        Cela permet d'exécuter rapidement et proprement le serveur MLFlow dans le bon environnement Python géré par uv.
    ///
    """
    )
    return


@app.cell
def _(Path, mlflow):
    # Chemin du répertoire où vous souhaitez stocker la base de données
    db_directory = Path("data/mlflow/db")

    # Créer le répertoire s'il n'existe pas
    db_directory.mkdir(exist_ok=True)

    # Chemin complet vers la base de données
    db_path = db_directory / "mlruns.db"

    # Définir le répertoire racine des artefacts
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    # Définir le nom de l'experience
    # mlflow.set_experiment("credit_scoring")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Chargement et Fusion des données""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Analyse des données""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Objectif** :

    - Comprendre du mieux possible les données
    - Développer une premiere stratégie de modélisation

    1. **Checklist de base**

    - **Analyse de Forme** :
        - variable target : TARGET binaire, 1 si le client à rencontré des difficultées de remboursement du prêt
        - Données déjà séparée en 2 fichiers train et test
        - lignes et colonnes : train (307511, 122) et test (48744, 121)
        - types de variables : qualitatives : 16, quantitatives : 105

    - **Analyse des valeurs manquantes** :
        - beaucoup de NaN (un peu moins de la moitié des variables > 30% de NaN)
        - Après suppression des colonnes avec plus de 30% de valeurs manquantes: qualitatives: 11, quantitatives: 60

    2. **Analyse de Fond** :

    - **Visualisation de la target** :

        Répartition déséquilibré de la Target 91,9% = 0 et 8,1% = 1.

        Utilisation de SMOTE pour réquilibré lors de l'entrainement du model.

    - **Signification des variables** :


    3. **Relation Variables / Target** :


    4. **Analyse plus détaillée**
        - **Relation Variables / Variables** :


    4. **hypotheses nulle (H0)**:


        H0 =
    """
    )
    return


@app.cell
def _(Path):
    # Affichage du contenu des fichiers du projet
    data_path = Path("data/raw/")
    file_names = [f.name for f in data_path.iterdir() if f.is_file()]
    file_names
    return (data_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Il y a un total de 10 fichiers voir (https://www.kaggle.com/c/home-credit-default-risk/data) :

    Les fichiers CSV chargés représentent diverses sources de données, incluant des informations sur les applications de crédit, les bureaux de crédit, les soldes, et les paiements. Ces données offrent une vue complète des comportements et des historiques financiers des clients.

    - Fichiers principaux **application_train.csv** et **application_test.csv**:

        Ce sont les fichiers principaux. **application_train.csv** contient les données d'entraînement avec la variable cible (TARGET), tandis que **application_test.csv** est utilisé pour tester les modèles sans la variable cible.
    Chaque ligne représente un prêt dans notre échantillon de données.


    - Fichiers Secondaires: fournissent des informations complémentaires sur chaque prêt.

        - **HomeCredit_columns_description.csv** : Décris l'utilité de chaque colonnes présents dans les csv.

        - **bureau.csv** :

            Informations globales sur les prêts passés d'un client dans d'autres institutions financières (via le bureau de crédit)

        - **bureau_balance.csv** :

            Historique mensuel détaillé de l’état de ces prêts, ligne par ligne, sur plusieurs mois.

        - **POS_CASH_balance.csv** :

            Contient des instantanés mensuels des soldes des prêts précédents sur point de vente (POS) et des prêts en espèces.
            Utilisé pour analyser les comportements de remboursement spécifiques à ces types de prêts.

        - **credit_card_balance.csv** :

            Fournit des instantanés mensuels des soldes des cartes de crédit précédentes des clients.
            Utilisé pour analyser les comportements de gestion de crédit et de remboursement des cartes de crédit.


        - **previous_application.csv** :

            Contient toutes les demandes de prêt précédentes faites par les clients auprès de Home Credit.
            Utilisé pour comprendre l'historique des demandes de prêt et évaluer les comportements de demande de crédit.

        - **installments_payments.csv** :

            Contient l'historique des remboursements pour les crédits précédemment décaissés.
            Utilisé pour analyser les habitudes de remboursement et identifier les tendances de paiement.
    """
    )
    return


@app.cell
def _(data_path, pd):
    _df = pd.read_csv(data_path / "HomeCredit_columns_description.csv", encoding="latin1")
    _df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Voyons le contenu des fichiers applications application_train et application_test.""")
    return


@app.cell
def _(pd):
    # Chargement des données d'entraînement depuis le fichier CSV
    app_train_df = pd.read_csv("data/raw/application_train.csv")

    print("Taille des données d'entraînement : ", app_train_df.shape)

    app_train_df.head()
    return (app_train_df,)


@app.cell
def _(pd):
    # Charger les données de test
    app_test_df = pd.read_csv("data/raw/application_test.csv")

    print("Taille des données de test : ", app_test_df.shape)

    app_test_df.head()
    return (app_test_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Les fichiers d'entraînement et de test partagent des colonnes identiques, à l'exception du fichier d'entraînement qui inclut une colonne supplémentaire nommée **TARGET**.

    Voici une description détaillée des colonnes :

    **SK_ID_CURR** : Cette colonne contient l'identifiant unique attribué à chaque demande de prêt. Elle permet de distinguer chaque dossier de manière individuelle.

    **TARGET** : Il s'agit de la colonne cible, cruciale pour l'analyse. Elle indique le résultat du prêt :

    0 : Signifie que le prêt a été remboursé sans problème.
    1 : Indique que l'emprunteur a rencontré des difficultés pour rembourser le prêt.
    Cette colonne "**TARGET**" est la variable que l'on cherche à prédire à travers les modèles d'analyse.

    Colonnes de caractéristiques : Les fichiers comportent également de multiples autres colonnes qui décrivent les caractéristiques des emprunteurs et de leurs demandes de prêt. 

    Ces caractéristiques peuvent être classées en plusieurs catégories :

    **Données démographiques** : Elles incluent des informations telles que l'âge, le sexe, la situation familiale, etc.

    **Données financières** : Ces informations couvrent divers aspects financiers comme le revenu, voiture de fonction, locataire etc.

    **Informations liées à la demande de prêt** : Documents fournis, première demande de prêt, etc.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Fusion des données""")
    return


@app.cell
def _(data_path, pd):
    bureau_df = pd.read_csv(data_path / "bureau.csv")
    bureau_balance_df = pd.read_csv(data_path / "bureau_balance.csv")
    pos_cash_balance_df = pd.read_csv(data_path / "POS_CASH_balance.csv")
    credit_card_balance_df = pd.read_csv(data_path / "credit_card_balance.csv")
    previous_application_df = pd.read_csv(data_path / "previous_application.csv")
    installments_payments_df = pd.read_csv(data_path / "installments_payments.csv")
    return (
        bureau_balance_df,
        bureau_df,
        credit_card_balance_df,
        installments_payments_df,
        pos_cash_balance_df,
        previous_application_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Agrégation du **bureau_balance_df** par SK_ID_BUREAU (identifiant unique d’un prêt connu par un bureau de crédit) :

    - months_min : le mois le plus ancien enregistré (ex : -60)

    - months_max : le mois le plus récent (ex : 0)

    - status_last : dernier statut connu du prêt (ex : "0", "1", "C", "X"…)

    Jointure de cette agrégation avec le fichier **bureau_df**, pour enrichir les informations globales du prêt avec son historique mensuel.
    """
    )
    return


@app.cell
def _(bureau_balance_df, bureau_df):
    bb_agg = (
        bureau_balance_df.groupby("SK_ID_BUREAU")
        .agg(months_min=("MONTHS_BALANCE", "min"), months_max=("MONTHS_BALANCE", "max"), status_last=("STATUS", "last"))
        .reset_index()
    )
    bureau = bureau_df.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""➡️ chaque prêt à maintenant des infos sur sa durée et son dernier statut mensuel.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Créer pour chaque client (SK_ID_CURR) un ensemble de caractéristiques agrégées sur ses anciens crédits auprès d'autres institutions (via **bureau.csv**)""")
    return


@app.cell
def _(app_test_df, app_train_df, bureau_df):
    _bureau_agg = (
        bureau_df.groupby("SK_ID_CURR")
        .agg({
            "AMT_CREDIT_SUM": ["mean", "sum"],  # Montant total des crédits (moyenne et somme)
            "AMT_CREDIT_SUM_OVERDUE": ["mean", "sum"],  # Retards de paiement
            "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],  # Dette restante
            "CREDIT_ACTIVE": lambda s: (s == "Active").sum(),  # Nombre de crédits actifs
            "CREDIT_DAY_OVERDUE": ["max"],  # Nombre maximal de jours de retard
            "AMT_ANNUITY": ["mean", "sum"],  # Annuities (échéances)
            "DAYS_CREDIT": ["mean", "min", "max"],  # Depuis combien de jours le crédit a été enregistré
            "CREDIT_TYPE": "nunique",  # Nombre de types de crédits différents
        })
        .reset_index()
    )

    # Nettoyage des noms de colonnes
    # e.g. ('AMT_CREDIT_SUM', 'mean') → 'AMT_CREDIT_SUM_mean'
    _bureau_agg.columns = ["_".join(col).strip() for col in _bureau_agg.columns.to_numpy()]

    # Correction du nom de colonne SK_ID_CURR apres le join
    _bureau_agg = _bureau_agg.rename(columns={"SK_ID_CURR_": "SK_ID_CURR"})

    # fusion
    app_train_merged = app_train_df.merge(_bureau_agg, on="SK_ID_CURR", how="left")
    app_test_merged = app_test_df.merge(_bureau_agg, on="SK_ID_CURR", how="left")
    return app_test_merged, app_train_merged


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""➡️ Datasets enrichis avec pour chaque client les informations sur leurs crédits passés chez d'autres prêteurs.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Collecte des demandes de prêts de chaque client (nombre de prêts accordés/refusés, montants)""")
    return


@app.cell
def _(app_test_merged, app_train_merged, previous_application_df):
    _PA_agg = (
        previous_application_df.groupby("SK_ID_CURR")
        .agg({
            "SK_ID_PREV": "count",  # Nombre total de demandes de prêt passées pour ce client
            "NAME_CONTRACT_STATUS": lambda s: (s == "Approved").sum(),  # Nombre de prêts approuvés (Approved)
            "AMT_APPLICATION": "mean",  # Montant moyen des demandes de prêts
            "AMT_CREDIT": "mean",  # Montant moyen des prêts accordés
        })
        .reset_index()
    )

    # fusion
    app_train_merged2 = app_train_merged.merge(_PA_agg, on="SK_ID_CURR", how="left")
    app_test_merged2 = app_test_merged.merge(_PA_agg, on="SK_ID_CURR", how="left")
    return app_test_merged2, app_train_merged2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ➡️ Ajout de features basées sur l’historique de demandes de prêts pour mieux évaluer les risques:

    - Un client qui a fait de nombreuses demandes dans le passé est peut-être en situation d’endettement risquée

    - Un client avec peu ou aucune demande approuvée peut être perçu comme plus risqué

    - Des montants trop élevés demandés systématiquement peuvent aussi signaler un profil à risque
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calcul de la différence de paiement et les jours de retard pour chaque paiement.""")
    return


@app.cell
def _(app_test_merged2, app_train_merged2, installments_payments_df):
    # Nouvelle colonne Client à trop ou pas assez payé ?
    installments_payments_df["PAY_DIFF"] = (
        installments_payments_df["AMT_PAYMENT"] - installments_payments_df["AMT_INSTALMENT"]
    )
    # Nouvelle colonne Client à payé en avance ou en retard ?
    installments_payments_df["PAY_DELAY_DAYS"] = (
        installments_payments_df["DAYS_ENTRY_PAYMENT"] - installments_payments_df["DAYS_INSTALMENT"]
    )

    _IP_agg = (
        installments_payments_df.groupby("SK_ID_CURR").agg(
            INST_CNT=("SK_ID_PREV", "count"),  # Nombre total d’échéances
            INST_PAY_DIFF_MEAN=("PAY_DIFF", "mean"),  # Écart moyen payé vs dû
            INST_PAY_DIFF_MIN=("PAY_DIFF", "min"),  # Écart minimum
            INST_PAY_DIFF_MAX=("PAY_DIFF", "max"),  # Écart maximum
            INST_DELAY_MEAN=("PAY_DELAY_DAYS", "mean"),  # Retard moyen
            INST_DELAY_MAX=("PAY_DELAY_DAYS", "max"),  # Retard max
            INST_TOTAL_PAID=("AMT_PAYMENT", "sum"),  # Total payé
            INST_TOTAL_DUE=("AMT_INSTALMENT", "sum"),  # Total dû
        )
    ).reset_index()

    # fusion
    app_train_merged3 = app_train_merged2.merge(_IP_agg, on="SK_ID_CURR", how="left")
    app_test_merged3 = app_test_merged2.merge(_IP_agg, on="SK_ID_CURR", how="left")
    return app_test_merged3, app_train_merged3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ➡️ Ajout de features basées sur la ponctualité des paiement pour mieux évaluer les risques:

    - Détecter les bons payeurs (ponctualité, paiement complet)

    - Repérer les risques (retards chroniques, paiements partiels)

    - Estimer la capacité de remboursement
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Nombre de prêts et retard par client""")
    return


@app.cell
def _(app_test_merged3, app_train_merged3, pos_cash_balance_df):
    _pos_agg = (
        pos_cash_balance_df.groupby("SK_ID_CURR")
        .agg({
            "SK_ID_PREV": "nunique",  # Nombre de prêts POS différents par client
            "SK_DPD": "max",  # Maximum de jours de retard sur les prêts POS
        })
        .reset_index()
    )

    # fusion
    app_train_merged4 = app_train_merged3.merge(_pos_agg, on="SK_ID_CURR", how="left")
    app_test_merged4 = app_test_merged3.merge(_pos_agg, on="SK_ID_CURR", how="left")
    return app_test_merged4, app_train_merged4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ➡️ Ajout de features basées sur le nombre de prêts pour mieux évaluer les risques:

    - Identifier les clients sur-enndettées
    - Repérer les risques d'insolvabilité
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Solde des cartes de crédit""")
    return


@app.cell
def _(app_test_merged4, app_train_merged4, credit_card_balance_df):
    _cc_agg = (
        credit_card_balance_df.groupby("SK_ID_CURR")
        .agg({
            "AMT_BALANCE": "mean",  # Solde moyen sur carte de crédit
            "SK_DPD": "mean",  # Délai moyen de retard de paiement
        })
        .reset_index()
    )

    # fusion
    app_train_merged5 = app_train_merged4.merge(_cc_agg, on="SK_ID_CURR", how="left")
    app_test_merged5 = app_test_merged4.merge(_cc_agg, on="SK_ID_CURR", how="left")
    return app_test_merged5, app_train_merged5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ➡️ Ajout de features basées sur les soldes de carte  pour mieux évaluer les risques:

    - Identifier les clients qui ont des difficultés dans les dépenses du quotidient
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ➡️ En résumé

    | Source                 | Feature Engineering                      |
    | ---------------------- | ---------------------------------------- |
    | `bureau`               | Dettes, montants, type crédits, statut   |
    | `bureau_balance`       | Historique des statuts de paiements      |
    | `previous_application` | Comptage demandes, taux d’approbation    |
    | `installments`         | Retard/avance de paiements, montants dus |
    | `pos_cash_balance`     | Retard max sur prêts à la consommation   |
    | `credit_card_balance`  | Moyenne des soldes & retards carte       |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Exploration des données""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Type des colonnes""")
    return


@app.cell
def _(app_train_merged5):
    # Nombre de chaque type de colonne
    app_train_merged5.dtypes.value_counts()
    return


@app.cell
def _(app_test_merged5):
    app_test_merged5.dtypes.value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Valeurs manquantes""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Je commence par écrire une fonction qui résume le contenu des jeux de données afin de m'assurer de l'absence de doublons ou de valeurs manquantes.""")
    return


@app.cell
def _(pd):
    def show_missings(df, threshold=0):
        # Calculer le pourcentage de valeurs manquantes par colonne
        pourcentage_manquant = round(df.isna().mean() * 100)

        # Créer un DataFrame pour les statistiques
        stats = pd.DataFrame({
            "Pourcentage de valeurs manquantes": pourcentage_manquant,
        })

        # Filtrer les colonnes avec des valeurs manquantes
        mis_val_table_ren_columns = stats[stats["Pourcentage de valeurs manquantes"] > threshold]

        # Trier les résultats par ordre décroissant
        stats_sorted = stats.sort_values(by="Pourcentage de valeurs manquantes", ascending=False)

        # Affichage de quelques informations récapitulatives
        print(f"""Votre DataFrame sélectionné contient {stats.shape[0]} colonnes.
    Il y a {mis_val_table_ren_columns.shape[0]} colonnes qui contiennent plus de {threshold}% de valeurs manquantes.""")

        return stats_sorted
    return (show_missings,)


@app.cell
def _(plt, sns):
    def plot_missing_data_distribution(
        stats_sorted,
        figsize=(20, 12),
    ):
        """Affiche la distribution des valeurs manquantes par variable dans un DataFrame."""
        plt.figure(figsize=figsize)
        sns.barplot(
            x=stats_sorted.index,
            y=stats_sorted["Pourcentage de valeurs manquantes"],
            label="Pourcentage de valeurs manquantes",
        )
        plt.title("Distribution des valeurs manquantes par variable", fontsize=16, fontweight="bold")
        plt.ylabel("Pourcentage de valeurs manquantes", fontsize=14)
        plt.xlabel("Variables", fontsize=14)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title="Légende", loc="upper right", fontsize=14)
        plt.tight_layout()
        plt.show()
    return (plot_missing_data_distribution,)


@app.cell
def _(app_train_merged5, plot_missing_data_distribution, show_missings):
    _missings_df = show_missings(app_train_merged5, 30)
    plot_missing_data_distribution(_missings_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Note
        type: warning

        Il manque des données dans plusieurs colonnes. 

        Le nombre de colonne étant important, je vais supprimer les colonnes avec plus de 30% de valeurs manquantes. J'étudirais ces colonnes si j'ai besoin d'améliorer les performances du model.

        **Ne pas oublier d'imputer les données lors de la création des modèles.**

    ///
    """
    )
    return


@app.cell
def _(app_test_merged5, app_train_merged5):
    # Calculer le seuil de 30% du nombre total de lignes
    threshold = len(app_train_merged5) * 0.7

    # Conserver uniquement les colonnes qui ont moins de 30% de valeurs manquantes
    app_train_df_filtered = app_train_merged5.dropna(axis="columns", thresh=threshold)

    print("Train shape")
    print(app_train_merged5.shape)
    print(app_train_df_filtered.shape)

    # Supprime les meme colonnes dans le test
    columns_to_keep = app_train_df_filtered.columns
    columns_to_keep = columns_to_keep.drop("TARGET")

    # Appliquer les mêmes colonnes au test set
    app_test_df_filtered = app_test_merged5[columns_to_keep]

    print("Test shape")
    print(app_test_merged5.shape)
    print(app_test_df_filtered.shape)
    return app_test_df_filtered, app_train_df_filtered


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Ainsi, j'ai diminué le nombre de colonne de mon dataset et il reste 97 colonnes.""")
    return


@app.cell
def _(app_train_df_filtered):
    # Nombre de chaque type de colonne
    app_train_df_filtered.dtypes.value_counts()
    return


@app.cell
def _(app_train_df_filtered, plot_missing_data_distribution, show_missings):
    _missings_df = show_missings(app_train_df_filtered)
    plot_missing_data_distribution(_missings_df)
    return


@app.cell
def _(app_test_df_filtered, app_train_df_filtered):
    print(f"Doublons Trainset: {app_train_df_filtered.duplicated().sum()}")
    print(f"Doublons Testset: {app_test_df_filtered.duplicated().sum()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Info
        type: warning

        Les jeu de données ne contiennent pas doublons.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Variables categorielles""")
    return


@app.cell
def _(app_train_df_filtered, pd):
    app_train_df_filtered.select_dtypes("object").apply(pd.Series.nunique, axis=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Info
        type: warning

        Pour les colonnes catgéorielles, j'utiliserai un Label Encoder si le nombre de catégorie est inférieur à 2 sinon j'utiliserai un One-Hot Encoder pour éviter les biais arbitraire.

    ///
    """
    )
    return


@app.cell
def _(plt, sns):
    def plot_categorical_distributions(df, feature, label=False):
        """Cette fonction génère un graphique de distribution pour chaque colonne de type 'object' dans un DataFrame.

        Paramètres:
        df (DataFrame): Le DataFrame contenant les données à visualiser.
        """
        # Calculer l'ordre des catégories basé sur la fréquence
        order = df[feature].value_counts().index

        # Créer une nouvelle figure pour chaque variable
        plt.figure(figsize=(20, 10))
        _ax = sns.countplot(x=feature, data=df, stat="percent", order=order)

        # Arrondir les valeurs et les afficher sur les barres
        if label:
            for _label in _ax.containers:
                _ax.bar_label(_label, fmt="%.1f%%", label_type="edge")

        # Incliner les étiquettes de l'axe des x de 45 degrés
        plt.xticks(rotation=90)
        _ax.set_ylabel("Count")
        _ax.set_title(f"Distribution de {feature}", fontsize=16)

        # Ajuster la mise en page et afficher
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
    return (plot_categorical_distributions,)


@app.cell
def _(app_train_df_filtered, plot_categorical_distributions):
    plot_categorical_distributions(app_train_df_filtered, "ORGANIZATION_TYPE")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A première vue, les colonnes catégorielles du jeu de données ne semblent pas contenir de valeurs aberrantes.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Colonnes quantitatives""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Suppression de valeurs aberrantes documentées""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Les notebooks de la compétition Kaggle indique la présence de valeurs aberrantes.

    Dans la colonne comptabilisant des jours en activité, il ya la présence de 365243. Cela équivaudrait à **1000 ans**...

    Je vais les remplacer par des valeurs manquantes (NAN).
    """
    )
    return


@app.cell
def _(plt, sns):
    def plot_distribution(df, feature):
        plt.figure(figsize=(10, 6))
        plt.title(f"Distribution de {feature}", fontsize=16, fontweight="bold")
        sns.histplot(df[feature], kde=True, bins=100, edgecolor="black")
        plt.xlabel(feature, fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
    return (plot_distribution,)


@app.cell
def _(app_train_df_filtered, plot_distribution):
    plot_distribution(app_train_df_filtered, "DAYS_EMPLOYED")
    return


@app.cell
def _(app_test_df_filtered, app_train_df_filtered, np):
    app_train_df_filtered["DAYS_EMPLOYED"] = app_train_df_filtered["DAYS_EMPLOYED"].replace(365243, np.nan)
    app_test_df_filtered["DAYS_EMPLOYED"] = app_test_df_filtered["DAYS_EMPLOYED"].replace(365243, np.nan)

    print("Remplacé 365243 par NaN dans les colonnes : DAYS_EMPLOYED")
    return


@app.cell
def _(app_train_df_filtered, plot_distribution):
    plot_distribution(app_train_df_filtered, "DAYS_EMPLOYED")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Il y a peu de client qui ont un emploi stable depuis plusieurs années et qui font des demandes de crédit à la consommation.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Encodage des variables catégorielles""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Problème : Les modèles de machine learning ne peuvent pas traiter directement les variables catégorielles.

    Solutions :

    - Encodage par étiquette (Label Encoding) : Assignation d'un entier à chaque catégorie unique. Utile pour les variables avec 2 catégories.

    - Encodage à chaud (One-Hot Encoding) : Création d'une colonne pour chaque catégorie unique. Préconisé pour plus de 2 catégories pour éviter les biais d'ordre arbitraire.

    1. Utilisation de l'encodage par étiquette pour les variables avec 2 catégories et de l'encodage à chaud pour celles avec plus de 2 catégories.

    2. Outils : LabelEncoder de Scikit-Learn pour l'encodage par étiquette et OneHotEncoder de Scikit-Learn pour l'encodage à chaud.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Une fois les variables catégorielles encodés, il faut s'assurer que le données d'entrainement et de test correspondent:

    - Objectif : S'assurer que les caractéristiques (colonnes) sont identiques dans les données d'entraînement et de test.

    - Problème : L'encodage à chaud a créé plus de colonnes dans les données d'entraînement car certaines variables catégorielles ont des catégories non représentées dans les données de test.

    - Solution : Aligner les dataframes pour supprimer les colonnes présentes dans les données d'entraînement mais absentes dans les données de test.

    1. Extraction de la cible : Extraire la colonne cible (TARGET) des données d'entraînement avant l'alignement.

    2. Alignement : Utiliser align avec axis=1 pour aligner les dataframes basés sur les colonnes, et non sur les lignes.
    """
    )
    return


@app.cell
def _(
    LabelEncoder,
    OneHotEncoder,
    app_test_df_filtered,
    app_train_df_filtered,
    pd,
):
    # Encoder les variables binaires avec LabelEncoder, sinon utiliser OneHotEncoder (scikit-learn)
    categorical_cols = app_train_df_filtered.select_dtypes(include="object").columns

    app_train_encoded = app_train_df_filtered.copy()
    app_test_encoded = app_test_df_filtered.copy()

    for col in categorical_cols:
        if app_train_encoded[col].nunique() == 2:
            # Label Encoding pour les colonnes binaires
            le = LabelEncoder()
            le.fit(app_train_encoded[col].astype(str))
            app_train_encoded[col] = le.transform(app_train_encoded[col].astype(str))
            app_test_encoded[col] = le.transform(app_test_encoded[col].astype(str))
        else:
            # One-hot encoding avec OneHotEncoder de scikit-learn
            ohe = OneHotEncoder(sparse_output=False)
            ohe.fit(app_train_encoded[[col]])

            train_ohe = pd.DataFrame(
                ohe.transform(app_train_encoded[[col]]),
                columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                index=app_train_encoded.index,
            )
            test_ohe = pd.DataFrame(
                ohe.transform(app_test_encoded[[col]]),
                columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                index=app_test_encoded.index,
            )

            app_train_encoded = pd.concat([app_train_encoded.drop(columns=[col]), train_ohe], axis=1)
            app_test_encoded = pd.concat([app_test_encoded.drop(columns=[col]), test_ohe], axis=1)

    print(f""""Données avant alignement :
    Train shape : {app_train_encoded.shape}
    Test shape : {app_test_encoded.shape}
    """)

    # Alignement des colonnes train/test (important pour éviter les erreurs de dimensions)
    # Extraire la colonne cible avant alignement
    train_labels = app_train_encoded["TARGET"]
    app_train_encoded = app_train_encoded.drop(columns=["TARGET"])

    # Aligner les colonnes (garder uniquement les colonnes communes)
    app_train_encoded, app_test_encoded = app_train_encoded.align(app_test_encoded, join="inner", axis=1)

    # Restaurer la cible
    app_train_encoded["TARGET"] = train_labels

    print(f""""Données alignées :
    Train shape : {app_train_encoded.shape}
    Test shape : {app_test_encoded.shape}
    """)
    return (app_train_encoded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Correlation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Objectif : Identifier les relations entre les caractéristiques et la cible (TARGET) en calculant le coefficient de corrélation de Pearson.

    - Méthode : Utiliser la méthode .corr sur le DataFrame pour obtenir les corrélations entre chaque variable et la cible.

    - Interprétation des valeurs absolues :

    0.00–0.19 : Très faible

    0.20–0.39 : Faible

    0.40–0.59 : Modérée

    0.60–0.79 : Forte

    0.80–1.00 : Très forte

    Limitation : Le coefficient de corrélation ne représente pas toujours parfaitement la pertinence d'une caractéristique, mais il donne une première indication des relations possibles dans les données.
    """
    )
    return


@app.cell
def _(app_train_df_encoded):
    app_train_df_encoded["DAYS_BIRTH"].head()
    return


@app.cell
def _(app_train_encoded):
    # Trouver les corrélations avec la cible et trier
    correlations = app_train_encoded.corr()["TARGET"].sort_values().round(2)

    # Afficher les corrélations
    print("Corrélations les plus positives :\n", correlations.tail(10))
    print("\nCorrélations les plus négatives :\n", correlations.head(10))
    return (correlations,)


@app.cell
def _(correlations, plt):
    plt.figure(figsize=(8, 5))

    # stem function
    plt.stem(correlations.drop("TARGET").values)
    plt.ylim(-0.3, 0.3)

    # Ajouter des titres et des labels
    plt.title("Corrélation des variables avec la cible (Zoom)")
    plt.xlabel("Variables")
    plt.ylabel("Valeurs de corrélation")

    # Afficher le graphique
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Les variables **EXT_SOURCE_2, et EXT_SOURCE_3** représentent des scores normalisés provenant de sources de données externes.

    Ces scores sont utilisés pour évaluer la solvabilité ou le risque d'un client à partir d'informations externes, comme des données de bureau de crédit ou d'autres bases de données financières.

    Ces scores sont souvent agrégés ou calculés à partir de plusieurs facteurs externes et fournissent une indication globale du risque.

    - Corrélation significative : La variable DAYS_BIRTH montre la corrélation positive la plus forte (hors TARGET, car une variable est toujours corrélée à 1 avec elle-même).

    - Interprétation : DAYS_BIRTH représente l'âge du client en jours négatifs. Une corrélation positive signifie que les clients plus âgés sont moins susceptibles de faire défaut sur leur prêt (TARGET == 0).
    """
    )
    return


@app.cell
def _(correlations):
    # Effet de l'Âge sur le Remboursement
    _pearson = correlations["DAYS_BIRTH"]

    print(f"Coefficient de correlation entre l'âge du client et la TARGET: {_pearson:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Corelation Négative : Il existe une relation linéaire négative entre l'âge du client et la cible (TARGET), indiquant que les clients plus âgés ont tendance à rembourser leurs prêts à temps plus souvent.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Analyse de la Variable : Commençons par examiner cette variable.

    Histogramme de l'Âge : Nous allons tracer un histogramme de l'âge, en utilisant l'axe des x en années pour rendre le graphique plus compréhensible.
    """
    )
    return


@app.cell
def _(app_train_encoded, plt, sns):
    # Tracer la distribution des âges en années
    sns.histplot(app_train_encoded['DAYS_BIRTH'] / 365, edgecolor='k', bins=25)
    plt.title('Âge du Client')
    plt.xlabel('Âge (années)')
    plt.ylabel('Nombre')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Un simple histograme ne permet pas d'analyser le lien de la catégorie avec la cible.
    Je vais tracer les courbes KDE en fonction de la TARGET

    - Distribution d'Âge : La courbe pour target == 1 (prêts non remboursés) est plus concentrée chez les jeunes.

    - Corrélation Faible : Bien que la corrélation soit faible (-0.07), l'âge est probablement utile pour les modèles de machine learning.

    - Analyse par Tranche d'Âge :

    - Création de Bins : Diviser l'âge en tranches de 5 ans.

    - Calcul de la Moyenne : Calculer le taux moyen de défaut de paiement par tranche d'âge.
    """
    )
    return


@app.cell
def _(app_train_encoded, plt, sns):
    # Tracer le KDE des prêts remboursés à temps
    sns.kdeplot(app_train_encoded.loc[app_train_encoded["TARGET"] == 0, "DAYS_BIRTH"] / 365, label="target == 0")

    # Tracer le KDE des prêts non remboursés à temps
    sns.kdeplot(app_train_encoded.loc[app_train_encoded["TARGET"] == 1, "DAYS_BIRTH"] / 365, label="target == 1")

    # Étiqueter le graphique
    plt.xlabel("Âge (années)")
    plt.ylabel("Densité")
    plt.title("Distribution des Âges")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Les clients jeunes être plus souvent en défaut de remboursement de leurs prêts.""")
    return


@app.cell
def _(app_train_encoded, np, pd, plt):
    # Créer un DataFrame séparé pour les informations sur l'âge
    age_data = app_train_encoded[["TARGET", "DAYS_BIRTH"]]

    # Convertir les jours de naissance en années
    age_data["YEARS_BIRTH"] = age_data["DAYS_BIRTH"] / 365

    # Diviser les données d'âge en tranches
    age_data["YEARS_BINNED"] = pd.cut(age_data["YEARS_BIRTH"], bins=np.linspace(20, 70, num=11))

    # Grouper par tranche d'âge et calculer les moyennes
    age_groups = age_data.groupby("YEARS_BINNED").mean()

    # Tracer les tranches d'âge et la moyenne de la cible sous forme de diagramme en barres
    plt.bar(age_groups.index.astype(str), 100 * age_groups["TARGET"])

    # Étiqueter le graphique
    plt.xticks(rotation=75)
    plt.xlabel("Groupe d'Âge (années)")
    plt.ylabel("Taux de Défaut de Paiement (%)")
    plt.title("Taux de Défaut de Paiement par Groupe d'Âge")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Le graphique ci dessus confirme que les clients jeunes 20-25 sont plus souvent en défaut de paiement de leurs crédits.

    Le taux de défaut est supérieur à 10%.

    Pour réduire ce risque, le client pourrait proposer une formation et accompagnement à ces clients.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Correlation positive**

    Sources Externes : Les variables EXT_SOURCE_2, et EXT_SOURCE_3.

    Elles représentent un score normalisé provenant de sources de données externes, possiblement une évaluation de crédit cumulative.

    Ces variables montrent les corrélations négatives les plus fortes avec la cible (TARGET).
    """
    )
    return


@app.cell
def _(correlations):
    # Extraire les variables EXT_SOURCE et calculer les corrélations
    correlations[["TARGET","EXT_SOURCE_2","EXT_SOURCE_3","DAYS_BIRTH"]]
    return


@app.cell
def _(app_train_encoded):
    ext_data = app_train_encoded[["TARGET", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH"]]
    ext_data_corrs = ext_data.corr().sort_values(by="TARGET")
    ext_data_corrs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Analyse de la variable cible""")
    return


@app.cell
def _(app_train_df_filtered, plt):
    target_counts = app_train_df_filtered["TARGET"].value_counts(normalize=True)

    # Définir l'effet "explode": chaque valeur représente la distance de la tranche par rapport au centre
    explode = [0.05] * len(target_counts)

    plt.figure(figsize=(7, 7))
    plt.pie(target_counts, labels=target_counts.index, autopct="%1.1f%%", startangle=90, explode=explode)
    plt.legend()
    plt.title("Target Distribution", fontsize=16, weight="bold")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Le graphique à barres ci-dessus révèle que la majorité des clients remboursent effectivement leurs prêts. 
    On observe un déséquilibre important dans la distribution de la variable cible. 

    Cette disparité est un facteur crucial à considérer lors du choix d'une méthode d'évaluation pour le modèle. 

    En effet, il est possible d'obtenir de bonnes performances prédictives simplement parce que le modèle néglige les prédictions pour les clients à risque. 

    Dans un tel cas, le modèle perdrait son utilité, car il ne remplirait pas son objectif principal d'identifier correctement les clients à risque.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Métriques d'évaluation
    Pour mieux répondre aux besoins de l'entreprise, nous allons créer une fonction pour calculer un "score métier". Ce score nous aidera à comprendre l'impact financier des décisions de crédit.

    Il est important de noter que les erreurs de prédiction n'ont pas le même coût. Par exemple, accorder un crédit à un client à risque (faux négatif) coûte bien plus cher que refuser un crédit à un bon client (faux positif). 
    En fait, un faux négatif coûte environ 10 fois plus qu'un faux positif.

    Voici comment nous calculons le "gain" pour la société :

    - Vrais positifs : Ce sont les bons clients qui rapportent de l'argent. Leur impact est positif et est noté +1.
    - Faux positifs : Ce sont les bons clients refusés à tort. La société perd un gain potentiel, donc leur impact est noté -1.
    - Faux négatifs : Ce sont les mauvais clients qui reçoivent un crédit par erreur. Cela entraîne une perte financière importante, notée -10.
    - Vrais négatifs : Ces cas n'affectent pas financièrement la société et sont donc ignorés dans notre calcul.

    Note : Ces coefficients de pondération pourront être ajustés en fonction des résultats observés lors de l'entraînement des modèles.
    """
    )
    return


if __name__ == "__main__":
    app.run()
