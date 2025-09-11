import marimo

__generated_with = "0.15.1"
app = marimo.App(
    width="medium",
    app_title="P7 Modelisation",
    layout_file="layouts/notebook_modelisation.slides.json",
)


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
    from sklearn.metrics import (
        make_scorer,
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
        confusion_matrix,
        ConfusionMatrixDisplay,
        RocCurveDisplay,
        PrecisionRecallDisplay,
        classification_report,
    )
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate, cross_val_predict
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split

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
    from optuna.pruners import SuccessiveHalvingPruner
    from optuna.samplers import TPESampler
    import logging
    import sys

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # === SHAP ===
    import shap

    # === LightGBM ===
    from lightgbm import LGBMClassifier

    # === Scipy ===
    from scipy.sparse import csr_matrix
    from scipy.stats import randint, uniform

    # === Data Drift ===
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    # === Model export API ===
    import pickle
    return (
        DataDriftPreset,
        DummyClassifier,
        LGBMClassifier,
        LabelEncoder,
        LogisticRegression,
        OneHotEncoder,
        Path,
        Pipeline,
        RandomUnderSampler,
        Report,
        SMOTE,
        SimpleImputer,
        StandardScaler,
        StratifiedKFold,
        SuccessiveHalvingPruner,
        TPESampler,
        confusion_matrix,
        cross_val_predict,
        cross_validate,
        datetime,
        infer_signature,
        logging,
        make_scorer,
        mlflow,
        mo,
        np,
        optuna,
        pd,
        pickle,
        plt,
        precision_recall_curve,
        shap,
        sns,
        train_test_split,
    )


@app.cell
def _(pd):
    def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
        """
        Down‑cast float64→float32 and int64→int32/16 to cut RAM ~50 %.
        Does NOT affect object / category columns.
        """
        for col in df.columns:
            t = df[col].dtype
            if t.kind in "iuf":
                df[col] = pd.to_numeric(df[col], downcast="float" if t.kind == "f" else "integer")
        return df
    return (reduce_mem,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Configuration de MLFlow""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Utilisation de just pour démarrer mlflow-server
        type: Info

        Just est une alternative à Make pour automatiser l'execution de commandes bash.
        Dans un premier temps, il faut s'assurer que just est installé sur le systeme.
        Puis un créer un fichier justfile.

        ```bash    
        # start mlflow server
        [group('ml-tools')]
        start-mlflow:
            uv run mlflow server \
            --backend-store-uri sqlite:///data/mlflow/db/mlflow.db \
            --default-artifact-root file:data/mlflow/artifacts
            --host 127.0.0.1 \
            --port 5000
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
    # Constantes
    # MLFLOW
    _MLFLOW_DB_PATH = Path("data/mlflow/db/mlflow.db")
    _ARTIFACTS_PATH = Path("data/mlflow/artifacts")
    ARTIFACT_LOCATION = str(_ARTIFACTS_PATH.resolve())
    _TRACKING_URI = f"sqlite:///{_MLFLOW_DB_PATH.resolve()}"

    # Optuna
    OPTUNA_DIR = Path("data/optuna/db/")

    # Crée les dossiers si besoin
    _MLFLOW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(_TRACKING_URI)
    return ARTIFACT_LOCATION, OPTUNA_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Dans cette section, on définit une fonction utilitaire pour gérer les expériences MLflow. L'objectif est de s'assurer que chaque modèle et ses paramètres associés sont enregistrés de manière organisée et traçable.

    La fonction `get_or_create_experiment` est utilisée pour récupérer l'ID d'une expérience MLflow existante ou pour en créer une nouvelle si elle n'existe pas encore. Cela permet de centraliser les informations relatives à chaque expérience, facilitant ainsi la gestion et la reproductibilité des tests et des résultats.

    - **Objectif** : Assurer la traçabilité et l'organisation des expériences de modélisation.
    - **Fonctionnalité** : Vérifie l'existence d'une expérience par son nom et la crée si nécessaire.
    """
    )
    return


@app.cell
def _(mlflow):
    def get_or_create_experiment(experiment_name, artifact_location):
        """Récupère l'ID d'une expérience MLflow existante ou en crée une nouvelle si elle n'existe pas."""

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    return (get_or_create_experiment,)


@app.cell(hide_code=True)
def _(mo):
    test_mlflow_button = mo.ui.run_button(kind="warn", full_width=True, label="Click to test mlflow db")
    test_mlflow_button
    return (test_mlflow_button,)


@app.cell
def _(
    ARTIFACT_LOCATION,
    Path,
    get_or_create_experiment,
    mlflow,
    mo,
    test_mlflow_button,
):
    mo.stop(not test_mlflow_button.value, mo.md("Click 👆 to test mlflow db").callout(kind="info"))
    # Create or get the experiment
    _experiment_id = get_or_create_experiment("Testing mlflow", artifact_location=ARTIFACT_LOCATION)

    # Start MLflow run
    with mlflow.start_run(experiment_id=_experiment_id):
        mlflow.log_param("param1", 42)
        mlflow.log_metric("metric1", 0.99)

        # Log an artifact and clean up
        sample_file = Path("sample.txt")
        sample_file.write_text("artifact content")
        mlflow.log_artifact(str(sample_file))
        sample_file.unlink()
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
    """
    )
    return


@app.cell
def _(Path):
    # Affichage du contenu des fichiers du projet
    data_path = Path("data/raw/")
    _file_names = [f.name for f in data_path.iterdir() if f.is_file()]
    _file_names
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
def _(data_path, pd):
    # Chargement des données d'entraînement depuis le fichier CSV
    _app_train_df = pd.read_csv(data_path / "application_train.csv")

    print(f"Taille des données d'entraînement : {_app_train_df.shape}")

    _app_train_df.head()
    return


@app.cell
def _(pd):
    # Charger les données de test
    _app_test_df = pd.read_csv("data/raw/application_test.csv")

    print(f"Taille des données de test : {_app_test_df.shape}")

    _app_test_df.head()
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Optimisation du code
        type: Info

        - Pour réduire l'utilisation de la mémoire dans marimo, la logique de chargement et d'agrégation de vos données sont encapsulées dans des fonctions , pour ne garder en mémoire que les variables réellement nécessaires pour un traitement ultérieur. 

        - Après la fusion, les dataframes intermédiaires sont supprimés. 

        - Des variables locales (préfixées par un trait de soulignement) sont utilisées éviter de polluer l'espace de noms global et pour aider le modèle d'exécution réactif de marimo à gérer la mémoire plus efficacement.

    ///
    """
    )
    return


@app.function
def merge_with_app(app_df, agg_df):
    """Fusionne les agrégations avec les données d'application."""
    return app_df.merge(agg_df, on="SK_ID_CURR", how="left")


@app.cell
def _(pd, reduce_mem):
    def load_application_data(data_path):
        """Charge les données d'application."""
        app_train_df = reduce_mem(pd.read_csv(data_path / "application_train.csv"))
        app_test_df = reduce_mem(pd.read_csv(data_path / "application_test.csv"))
        return app_train_df, app_test_df
    return (load_application_data,)


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
def _(pd, reduce_mem):
    def load_and_aggregate_bureau(data_path):
        """Charge et agrège les données de bureau."""
        bureau_df = reduce_mem(pd.read_csv(data_path / "bureau.csv"))
        _bureau_balance_df = reduce_mem(pd.read_csv(data_path / "bureau_balance.csv"))

        _bb_agg = (
            _bureau_balance_df.groupby("SK_ID_BUREAU")
            .agg(months_min=("MONTHS_BALANCE", "min"), months_max=("MONTHS_BALANCE", "max"), status_last=("STATUS", "last"))
            .reset_index()
        )
        bureau = bureau_df.merge(_bb_agg, on="SK_ID_BUREAU", how="left")

        return bureau, bureau_df
    return (load_and_aggregate_bureau,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""➡️ chaque prêt à maintenant des infos sur sa durée et son dernier statut mensuel.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Créer pour chaque client (SK_ID_CURR) un ensemble de caractéristiques agrégées sur ses anciens crédits auprès d'autres institutions (via **bureau.csv**)""")
    return


@app.function
def aggregate_bureau_features(bureau_df):
    """Agrège les caractéristiques des données de bureau."""
    _bureau_agg = (
        bureau_df.groupby("SK_ID_CURR")
        .agg(
            AMT_CREDIT_SUM_mean=("AMT_CREDIT_SUM", "mean"),  # Montant total des crédits (moyenne et somme)
            AMT_CREDIT_SUM_sum=("AMT_CREDIT_SUM", "sum"),
            AMT_CREDIT_SUM_OVERDUE_mean=("AMT_CREDIT_SUM_OVERDUE", "mean"),  # Retards de paiement
            AMT_CREDIT_SUM_OVERDUE_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
            AMT_CREDIT_SUM_DEBT_mean=("AMT_CREDIT_SUM_DEBT", "mean"),  # Dette restante
            AMT_CREDIT_SUM_DEBT_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
            CREDIT_ACTIVE_count_active=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),  # Nombre de crédits actifs
            CREDIT_DAY_OVERDUE_max=("CREDIT_DAY_OVERDUE", "max"),  # Nombre maximal de jours de retard
            AMT_ANNUITY_mean=("AMT_ANNUITY", "mean"),  # Annuities (échéances)
            DAYS_CREDIT_mean=("DAYS_CREDIT", "mean"),  # Depuis combien de jours le crédit a été enregistré
            DAYS_CREDIT_min=("DAYS_CREDIT", "min"),
            DAYS_CREDIT_max=("DAYS_CREDIT", "max"),
            CREDIT_TYPE_nunique=("CREDIT_TYPE", "nunique"),  # Nombre de types de crédits différents
        )
        .reset_index()
    )

    return _bureau_agg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""➡️ Datasets enrichis avec pour chaque client les informations sur leurs crédits passés chez d'autres prêteurs.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Collecte des demandes de prêts de chaque client (nombre de prêts accordés/refusés, montants)""")
    return


@app.cell
def _(pd, reduce_mem):
    def process_previous_applications(data_path):
        """Traite les données d'applications précédentes."""
        _previous_application_df = reduce_mem(pd.read_csv(data_path / "previous_application.csv"))
        _PA_agg = (
            _previous_application_df.groupby("SK_ID_CURR")
            .agg({
                "SK_ID_PREV": "count",  # Nombre total de demandes de prêt passées pour ce client
                "NAME_CONTRACT_STATUS": lambda s: (s == "Approved").sum(),  # Nombre de prêts approuvés (Approved)
                "AMT_APPLICATION": "mean",  # Montant moyen des demandes de prêts
                "AMT_CREDIT": "mean",  # Montant moyen des prêts accordés
            })
            .reset_index()
        )

        return _PA_agg
    return (process_previous_applications,)


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
def _(pd, reduce_mem):
    def process_installments_payments(data_path):
        """Traite les données des paiements d'échéanciers."""
        # Chargement du fichier
        _installments_payments_df = reduce_mem(pd.read_csv(data_path / "installments_payments.csv"))

        # Nouvelle colonne: Client à trop ou pas assez payé ?
        _installments_payments_df["PAY_DIFF"] = (
            _installments_payments_df["AMT_PAYMENT"] - _installments_payments_df["AMT_INSTALMENT"]
        )

        # Nouvelle colonne: Client à payé en avance ou en retard ?
        _installments_payments_df["PAY_DELAY_DAYS"] = (
            _installments_payments_df["DAYS_ENTRY_PAYMENT"] - _installments_payments_df["DAYS_INSTALMENT"]
        )
        _IP_agg = (
            _installments_payments_df.groupby("SK_ID_CURR").agg(
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

        return _IP_agg
    return (process_installments_payments,)


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
def _(pd, reduce_mem):
    def process_pos_cash_balance(data_path):
        """Traite les données de balance de trésorerie POS."""
        # Charge le fichier
        _pos_cash_balance_df = reduce_mem(pd.read_csv(data_path / "POS_CASH_balance.csv"))

        _pos_agg = (
            _pos_cash_balance_df.groupby("SK_ID_CURR")
            .agg({
                "SK_ID_PREV": "nunique",  # Nombre de prêts à la conso différents par client
                "SK_DPD": "max",  # Maximum de jours de retard de remboursement sur les prêts
            })
            .reset_index()
        )

        return _pos_agg
    return (process_pos_cash_balance,)


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
def _(pd, reduce_mem):
    def process_credit_card_balance(data_path):
        """Traite les données de balance de carte de crédit."""
        # Charge le fichier
        _credit_card_balance_df = reduce_mem(pd.read_csv(data_path / "credit_card_balance.csv"))

        _cc_agg = (
            _credit_card_balance_df.groupby("SK_ID_CURR")
            .agg({
                "AMT_BALANCE": "mean",  # Solde moyen sur carte de crédit
                "SK_DPD": "mean",  # Délai moyen de retard de paiement
            })
            .reset_index()
        )

        return _cc_agg
    return (process_credit_card_balance,)


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


@app.cell
def merge_dataset(
    data_path,
    load_and_aggregate_bureau,
    load_application_data,
    mo,
    process_credit_card_balance,
    process_installments_payments,
    process_pos_cash_balance,
    process_previous_applications,
):
    with mo.persistent_cache(name="merge_dataset"):
        # Charger les données d'application
        _app_train_df, _app_test_df = load_application_data(data_path)

        # Charger et agréger les données de bureau
        _bureau, _bureau_df = load_and_aggregate_bureau(data_path)
        _bureau_agg = aggregate_bureau_features(_bureau_df)
        _bureau_df = None  # Libérer la mémoire

        # Fusion avec les données d'application
        _app_train_merged = merge_with_app(_app_train_df, _bureau_agg)
        _app_test_merged = merge_with_app(_app_test_df, _bureau_agg)
        _bureau_agg = None  # Libérer la mémoire
        _app_train_df = None
        _app_test_df = None

        # Traiter les applications précédentes
        _previous_applications_agg = process_previous_applications(data_path)
        _app_train_merged2 = merge_with_app(_app_train_merged, _previous_applications_agg)
        _app_test_merged2 = merge_with_app(_app_test_merged, _previous_applications_agg)
        _previous_applications_agg = None  # Libérer la mémoire
        _app_train_merged = None
        _app_test_merged = None

        # Traiter les paiements d'échéanciers
        _installments_agg = process_installments_payments(data_path)
        _app_train_merged3 = merge_with_app(_app_train_merged2, _installments_agg)
        _app_test_merged3 = merge_with_app(_app_test_merged2, _installments_agg)
        _installments_agg = None  # Libérer la mémoire
        _app_train_merged2 = None
        _app_test_merged2 = None

        # Traiter les balances de trésorerie POS
        _pos_agg = process_pos_cash_balance(data_path)
        _app_train_merged4 = merge_with_app(_app_train_merged3, _pos_agg)
        _app_test_merged4 = merge_with_app(_app_test_merged3, _pos_agg)
        _pos_agg = None  # Libérer la mémoire
        _app_train_merged3 = None
        _app_test_merged3 = None

        # Traiter les balances de carte de crédit
        _credit_card_agg = process_credit_card_balance(data_path)
        app_train_merged5 = merge_with_app(_app_train_merged4, _credit_card_agg)
        app_test_merged5 = merge_with_app(_app_test_merged4, _credit_card_agg)
        _credit_card_agg = None  # Libérer la mémoire
        _app_train_merged4 = None
        _app_test_merged4 = None
    return app_test_merged5, app_train_merged5


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
    _threshold = len(app_train_merged5) * 0.7

    # Conserver uniquement les colonnes qui ont moins de 30% de valeurs manquantes
    app_train_df_filtered = app_train_merged5.dropna(axis="columns", thresh=_threshold)

    print("Train shape")
    print(app_train_merged5.shape)
    print(app_train_df_filtered.shape)

    # Supprime les meme colonnes dans le test
    _columns_to_keep = app_train_df_filtered.columns
    _columns_to_keep = _columns_to_keep.drop("TARGET")

    # Appliquer les mêmes colonnes au test set
    app_test_df_filtered = app_test_merged5[_columns_to_keep]

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
    # Conversion explicite des colonnes en float
    app_train_df_filtered.loc[:, "DAYS_EMPLOYED"] = app_train_df_filtered["DAYS_EMPLOYED"].astype(float)
    app_test_df_filtered.loc[:, "DAYS_EMPLOYED"] = app_test_df_filtered["DAYS_EMPLOYED"].astype(float)

    # Remplacement des valeurs
    app_train_df_filtered.loc[:, "DAYS_EMPLOYED"] = app_train_df_filtered["DAYS_EMPLOYED"].replace(365243, np.nan)
    app_test_df_filtered.loc[:, "DAYS_EMPLOYED"] = app_test_df_filtered["DAYS_EMPLOYED"].replace(365243, np.nan)

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Mais tout d'abord, j'ai noté des noms de colonnes catégorielles avec des caractères spéciaux, je crée une fonction pour les supprimer dans un pipeline.""")
    return


@app.cell
def _(
    LabelEncoder,
    OneHotEncoder,
    app_test_df_filtered,
    app_train_df_filtered,
    pd,
    pickle,
):
    _categorical_cols = app_train_df_filtered.select_dtypes(include="object").columns
    _CAT_THRESHOLD = 2

    # Sauvegarde des encoders
    _encoders = {}

    # Encoder les variables catégorielles
    app_train_encoded = app_train_df_filtered.copy()
    app_test_encoded = app_test_df_filtered.copy()

    for _col in _categorical_cols:
        if app_train_encoded[_col].nunique() == _CAT_THRESHOLD:
            # Label Encoding pour les colonnes binaires
            le = LabelEncoder()
            le.fit(app_train_encoded[_col].astype(str))
            app_train_encoded[_col] = le.transform(app_train_encoded[_col].astype(str))
            app_test_encoded[_col] = le.transform(app_test_encoded[_col].astype(str))
            _encoders[_col] = le
        else:
            # One-hot encoding
            ohe = OneHotEncoder(sparse_output=False)
            combined_data = pd.concat([app_train_encoded[[_col]], app_test_encoded[[_col]]])
            ohe.fit(combined_data)

            # Transformation des données
            train_ohe = pd.DataFrame(
                ohe.transform(app_train_encoded[[_col]]),
                columns=[f"{_col}_{cat}" for cat in ohe.categories_[0]],
                index=app_train_encoded.index,
            )
            test_ohe = pd.DataFrame(
                ohe.transform(app_test_encoded[[_col]]),
                columns=[f"{_col}_{cat}" for cat in ohe.categories_[0]],
                index=app_test_encoded.index,
            )
            app_train_encoded = pd.concat([app_train_encoded.drop(columns=[_col]), train_ohe], axis=1)
            app_test_encoded = pd.concat([app_test_encoded.drop(columns=[_col]), test_ohe], axis=1)
            _encoders[_col] = ohe

    # Suppression des caractères spéciaux créés après le One Hot encoder
    app_train_encoded.columns = app_train_encoded.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
    app_test_encoded.columns = app_test_encoded.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)

    # Affichage des dimensions avant alignement
    print(f"""Données avant alignement :
    Train shape : {app_train_encoded.shape}
    Test shape : {app_test_encoded.shape}
    """)

    # Alignement des colonnes train/test
    _train_labels = app_train_encoded["TARGET"]
    app_train_encoded = app_train_encoded.drop(columns=["TARGET"])

    app_train_encoded, app_test_encoded = app_train_encoded.align(app_test_encoded, join="inner", axis=1)
    app_train_encoded["TARGET"] = _train_labels

    # Sauvegarde encoders
    with open("model/encoders.pkl", "wb") as _f:
        pickle.dump(_encoders, _f)

    # Sauvegarde features
    with open("model/model_features.pkl", "wb") as _f:
        pickle.dump(app_train_encoded.drop(columns=["TARGET"]).columns.tolist(), _f)

    # Affichage des dimensions après alignement
    print(f"""Données alignées :
    Train shape : {app_train_encoded.shape}
    Test shape : {app_test_encoded.shape}
    """)
    return app_test_encoded, app_train_encoded


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
    sns.histplot(-app_train_encoded["DAYS_BIRTH"] / 365, edgecolor="k", bins=25)
    plt.title("Âge du Client")
    plt.xlabel("Âge (années)")
    plt.ylabel("Nombre")
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
    sns.kdeplot(-app_train_encoded.loc[app_train_encoded["TARGET"] == 0, "DAYS_BIRTH"] / 365, label="target == 0")

    # Tracer le KDE des prêts non remboursés à temps
    sns.kdeplot(-app_train_encoded.loc[app_train_encoded["TARGET"] == 1, "DAYS_BIRTH"] / 365, label="target == 1")

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
    # Créer une copie explicite du DataFrame
    _age_data = app_train_encoded[["TARGET", "DAYS_BIRTH"]].copy()

    # Convertir les jours de naissance en années en utilisant .loc
    _age_data.loc[:, "YEARS_BIRTH"] = -_age_data["DAYS_BIRTH"] / 365

    # Diviser les données d'âge en tranches
    _age_data.loc[:, "YEARS_BINNED"] = pd.cut(_age_data["YEARS_BIRTH"], bins=np.linspace(20, 70, num=11))

    # Grouper par tranche d'âge et calculer les moyennes
    _age_groups = _age_data.groupby("YEARS_BINNED", observed=True).mean()

    # Tracer les tranches d'âge et la moyenne de la cible sous forme de diagramme en barres
    plt.bar(_age_groups.index.astype(str), 100 * _age_groups["TARGET"])

    # Étiqueter le graphique
    plt.xticks(rotation=75)
    plt.xlabel("Groupe d'Âge (années)")
    plt.ylabel("Taux de Défaut de Paiement (%)")
    plt.title("Taux de Défaut de Paiement par Groupe d'Âge")

    plt.show()
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


@app.cell(hide_code=True)
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
def _(app_train_encoded):
    _ext_data = app_train_encoded[["TARGET", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH"]]
    _ext_data_corrs = _ext_data.corr().sort_values(by="TARGET")
    _ext_data_corrs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Corrélations Négatives : Les trois variables EXT_SOURCE sont négativement corrélées avec la cible (TARGET), ce qui indique qu'une augmentation de leur valeur est associée à une probabilité plus élevée de remboursement du prêt.

    - Corrélation Positive avec l'Âge : DAYS_BIRTH est positivement corrélé avec EXT_SOURCE_1, suggérant que l'âge pourrait être un facteur pris en compte dans ce score.
    """
    )
    return


@app.cell
def _(app_train_encoded, plt, sns):
    # Définir la taille de la figure
    plt.figure(figsize=(10, 12))

    # Itérer à travers les variables EXT_SOURCE
    for _i, _col in enumerate(["EXT_SOURCE_2", "EXT_SOURCE_3"]):
        # Créer un sous-graphe pour chaque variable
        plt.subplot(3, 1, _i + 1)

        # Tracer la distribution des prêts remboursés (TARGET == 0)
        sns.kdeplot(app_train_encoded.loc[app_train_encoded["TARGET"] == 0, _col], label="target = 0")

        # Tracer la distribution des prêts non remboursés (TARGET == 1)
        sns.kdeplot(app_train_encoded.loc[app_train_encoded["TARGET"] == 1, _col], label="target = 1")

        # Étiqueter les graphiques
        plt.title(f"Distribution de {_col} selon la valeur de TARGET")
        plt.xlabel(_col)
        plt.legend()
        plt.ylabel("Densité")

    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout(h_pad=2.5)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""EXT_SOURCE_3 : Cette variable 1. montre la plus grande différence entre les valeurs de la target, indiquant une relation avec la probabilité de remboursement d'un prêt, bien que cette relation soit faible.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Utilisation d'un grid plot pour visualiser la valeurs de la Target individuellement par rapport aux variables étudiées précédemment.""")
    return


@app.cell
def _(app_train_encoded, plt, sns):
    _target_counts = app_train_encoded["TARGET"].value_counts(normalize=True)

    # Calculez les poids pour chaque ligne
    # L'idée est d'inverser les probabilités pour donner plus de poids aux valeurs sous-représentées
    _weights = app_train_encoded["TARGET"].apply(lambda x: 1 / _target_counts[x])

    # Échantillonnez en utilisant les poids calculés
    _balanced_sample = app_train_encoded.sample(n=50000, weights=_weights, random_state=42)

    # Utilisez seaborn pour le pairplot
    sns.pairplot(
        _balanced_sample[["TARGET", "DAYS_BIRTH", "EXT_SOURCE_2", "EXT_SOURCE_3"]],
        hue="TARGET",
        corner=True,
    )

    plt.suptitle("Séparation visuelle des classes TARGET sur un échantillon équilibré", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Analyse de la variable cible""")
    return


@app.cell
def _(app_train_encoded, plt):
    _target_counts = app_train_encoded["TARGET"].value_counts(normalize=True)

    # Définir l'effet "explode": chaque valeur représente la distance de la tranche par rapport au centre
    _explode = [0.05] * len(_target_counts)

    plt.figure(figsize=(7, 7))
    plt.pie(_target_counts, labels=_target_counts.index, autopct="%1.1f%%", startangle=90, explode=_explode)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Métriques d'évaluation
    Pour mieux répondre aux besoins de l'entreprise, nous allons créer une fonction pour calculer un "score métier". Ce score nous aidera à comprendre l'impact financier des décisions de crédit.

    Il est important de noter que les erreurs de prédiction n'ont pas le même coût. Par exemple, accorder un crédit à un client à risque (faux négatif) coûte bien plus cher que refuser un crédit à un bon client (faux positif). 
    En fait, un faux négatif coûte environ 10 fois plus qu'un faux positif.

    /// admonition 
    Voici comment on calcule le "gain" pour la société :

    Vrais positifs (TP) : Bons clients à qui l'on accorde un crédit. Cela rapporte de l'argent à la société. Impact : +1.

    Faux négatifs (FN) : Bons clients à qui on refuse un crédit. La société perd un gain potentiel. Impact : -1.

    Faux positifs (FP) : Mauvais clients à qui on accorde un crédit. Cela entraîne une perte financière importante. Impact : -10.

    Vrais négatifs (TN) : Mauvais clients à qui on refuse un crédit. Pas d'impact financier. Non pris en compte dans le calcul.

    Note : Ces coefficients de pondération pourront être ajustés en fonction des résultats observés lors de l'entraînement des modèles.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Dans ce projet, l'objectif est d'évaluer un modèle de prédiction du risque de crédit. Étant donné que nous travaillons avec des classes déséquilibrées (c'est-à-dire que le nombre de clients bons est beaucoup plus élevé que celui des clients à risque), certaines métriques classiques comme la précision ou l'exactitude ne sont pas suffisantes pour évaluer correctement la performance du modèle. Nous devons donc choisir des métriques adaptées au contexte:

    -  **AUC-ROC (Area Under the Curve - Receiver Operating Characteristic):** L'AUC-ROC est une métrique très importante, surtout dans les cas où les classes sont déséquilibrées. Elle mesure la capacité du modèle à différencier les classes positives et négatives à différents seuils de classification. Une AUC proche de 1 indique que le modèle est performant à distinguer les deux classes, tandis qu'une AUC proche de 0.5 suggère que le modèle ne fait pas mieux qu'un choix aléatoire.
    L'AUC-ROC est particulièrement utile dans le cadre de classes déséquilibrées, car elle n'est pas influencée par le choix du seuil de classification.

    - **Précision (Precision):** La précision mesure la proportion de prédictions positives correctes parmi toutes les prédictions positives. Autrement dit, elle nous dit combien de fois le modèle a correctement identifié un bon client parmi ceux qu'il a classés comme bons. Dans notre cas, la précision est importante pour éviter des faux positifs, c'est-à-dire accorder un crédit à un mauvais client (ce qui a un coût faible mais reste à éviter).

    Ne pas avoir à dire, je n'aurais pas du validé ce client.

    - **Rappel (Recall):** Le rappel mesure la proportion de vrais positifs capturés parmi tous les vrais positifs possibles. En d'autres termes, il nous indique combien de mauvais clients (faux négatifs) le modèle a bien identifiés. Le rappel est particulièrement important dans notre contexte, car le coût d'un faux négatif est élevé (accorder un crédit à un mauvais client représente un risque financier majeur). Un bon modèle devrait donc maximiser le rappel pour minimiser les faux négatifs.

      Est ce que j'ai raté de bon clients ?

    **F1-Score:** Le F1-score est la moyenne harmonique entre la précision et le rappel. Il est particulièrement utile lorsqu'on souhaite un compromis entre les deux. Dans notre cas, un F1-score élevé signifie que le modèle équilibre bien les erreurs de faux positifs et de faux négatifs.
    Cela est utile si on cherche un modèle robuste qui ne favorise ni les faux positifs ni les faux négatifs.

    **Matrice de Confusion:** La matrice de confusion est un outil de visualisation qui permet de comprendre le type d'erreurs faites par le modèle :

    - Vrais Positifs (TP) : Nombre de fois où le modèle a correctement identifié un bon client.
    - Faux Positifs (FP) : Le modèle dit "risqué", mais le client est fiable.
    - Faux Négatifs (FN) : Le modèle dit "fiable", mais le client est risqué.
    - Vrais Négatifs (TN) : Nombre de fois où le modèle a correctement identifié un mauvais client.

    Cette matrice permet de visualiser directement où le modèle fait ses erreurs, ce qui est très utile pour ajuster les seuils de classification et améliorer la prise de décision.
    """
    )
    return


@app.cell
def _(confusion_matrix, np):
    def normalized_business_score(y_true, y_pred, cost_fn: int = -10, cost_fp: int = -1, gain_tp: int = 1) -> float:
        """
        Calcule un score métier normalisé entre 0 (pire) et 1 (meilleur),
        basé sur les gains des vrais positifs et les coûts des erreurs.

        - TP = client risqué bien identifié (gain)
        - FN = client risqué non détecté (coût élevé)
        - FP = client non risqué identifié à tort comme risqué (coût modéré)

        Paramètres :
        -----------
        - y_true : Liste ou tableau des vraies classes (0 ou 1)
        - y_pred : Liste ou tableau des classes prédites (0 ou 1)
        - cost_fn : Coût d'un faux négatif (FN) défaut -10
        - cost_fp : Coût d'un faux positif (FP) défaut -1
        - gain_tp : Gain d'un vrai positif (TP) défaut 1

        Retourne :
        --------
        float
            Le score normalisé entre 0 (pire) et 1 (meilleur).
        """

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Gain total obtenu avec ce modèle
        gain_reel = tp * gain_tp + fp * cost_fp + fn * cost_fn

        total_tp = tp + fn  # tous les clients à risque
        total_neg = tn + fp  # tous les clients non risqués

        # Score idéal : tous les TP correctement détectés, aucune erreur
        score_max = total_tp * gain_tp

        # Pire score possible : on rate tous les TP et on fait le max de FP
        score_min = total_tp * cost_fn + total_neg * cost_fp

        # Normalisation entre 0 et 1
        if score_max == score_min:
            return 0.0  # Cas limite improbable

        score_normalise = (gain_reel - score_min) / (score_max - score_min)
        return score_normalise


    _y_true = np.array([1, 1, 0, 1])
    _y_pred = np.array([1, 0, 0, 0])

    score = normalized_business_score(_y_true, _y_pred, cost_fn=-10, cost_fp=-1, gain_tp=1)
    print("Score normalisé:", score)
    return (normalized_business_score,)


@app.cell
def _(make_scorer, normalized_business_score):
    # Création du scoring
    scoring = {
        "rocauc": "roc_auc",  # Performance globale du modèle indépendamment du seuil
        "precision": "precision",  # Importance pour éviter d’accorder un crédit à un mauvais client
        "recall": "recall",  # Critique pour minimiser les mauvais crédits accordés (FN)
        "f1_score": "f1",  # Compromis entre précision et rappel
        "business": make_scorer(
            normalized_business_score, greater_is_better=True
        ),  # Évaluation basée sur le coût des erreurs
    }
    return (scoring,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Re-équilibrage""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Comme spécifié, nous sommes face à un déséquilibre important des classes, nous devrons mettre en oeuvre une méthode de rééquilibrage afin d'élaborer un modèle pertinent.

    Méthode utilisée: Combinaison d'oversampling et d'undersampling.

    - **Oversampling avec SMOTE** pour créer des doublons extrapoler de la classe minoritaire pour atteindre une répartition toujours déséquilibré mais seulement de 20/80.
    - Ensuite Undersampling pour réduire de moitié le nombre d'échantillon de la classe majoritaire.

    ➡️ Cette méthodologie permet de reéquilibrer la Target à 33/66 mais réduit le nombre d'individus du jeu de données de moitié.
    C'est un bon compromis pour réduire les biais liés à l'échantillonage.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Info: 
        type: Info

        Afin d'éviter tout risque de data leakage, on implementera le ré-équilibreur SMOTE dans un pipeline lors de la définition de nos modèles.

    ///
    """
    )
    return


@app.cell
def _(Pipeline, RandomUnderSampler, SMOTE, SimpleImputer):
    # Define our oversampling and undersampling objects
    oversampler = SMOTE(sampling_strategy=0.2, k_neighbors=5, random_state=42)
    undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

    # Create a pipeline to manage balancing
    balancer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("oversampler", oversampler),
        ("undersampler", undersampler),
    ])
    return balancer, oversampler, undersampler


@app.cell
def _(app_train_encoded, balancer):
    # Split features and target
    _df = app_train_encoded

    _X_test = _df.drop("TARGET", axis=1)
    _y_test = _df["TARGET"]

    # Rebalance classes
    _X_test_balanced, _y_test_balanced = balancer.fit_resample(_X_test, _y_test)

    # Print number of values for each classes
    print("Balance target AVANT re-équilibrage:")
    print(_y_test.value_counts(normalize=True))
    print(f"NB individus: {_y_test.count()}")
    print()
    print("Balance target APRES re-équilibrage:")
    print(_y_test_balanced.value_counts(normalize=True))
    print(f"NB individus: {_y_test_balanced.count()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Modélisation

    Pour trouver le modèle présentant le plus d'intérêt et donc celui à optimiser, nous allons en mettre 3 à l'épreuve :

    - tout d'abord un DummyClassifier afin d'avoir une baseline
    - un LogisticRegression
    - un LGBMClassifier

    Pour chacun de ces modèles, nous définirons un pipeline comme suit :

    - Rééquilibrage des classes
    - Modèle

    Le modèle sera évalué sur le scoring métier, néanmoins, nous calculerons l'aire sous la courbe ROC ainsi que le temps d'entrainement.

    Pour cela, nous utiliserons une fonction qui nous permettra de faire une cross validation à l'aide d'un StratifiedKFold afin d'avoir plusieurs échantillons d'entrainement et de test.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Dans un contexte de données déséquilibrées, il est crucial de veiller à ce que les proportions des différentes classes (ici, TARGET) soient représentées de manière équitable dans les ensembles d'entraînement et de test. Si nous ne faisons pas attention, le modèle pourrait ne pas apprendre correctement à prédire la classe minoritaire, ce qui nuirait à la performance du modèle.

    Pour cela, nous utilisons la stratification lors de la séparation des données. La stratification permet de garantir que les proportions de chaque classe dans le jeu de données d'entraînement (train) et de test (test) sont identiques à celles du jeu de données original.

    Cela se fait facilement avec la fonction train_test_split de sklearn, où l'argument stratify permet de spécifier que la séparation doit se faire de manière à respecter la répartition des classes dans la variable cible (TARGET).
    """
    )
    return


@app.cell
def _(app_train_encoded, train_test_split):
    # Séparer features et cible
    _X_full = app_train_encoded.drop("TARGET", axis=1)
    _y_full = app_train_encoded["TARGET"]

    # Split équilibré : 50k train / 10k validation (≈ 60k sur 300k au total)
    X_train, X_val, y_train, y_val = train_test_split(
        _X_full, _y_full, train_size=50000, test_size=10000, stratify=_y_full, random_state=42
    )
    return X_train, X_val, y_train


@app.cell
def _(y_train):
    print("Balance target échantillonnées:")
    print(y_train.value_counts(normalize=True))
    print(f"NB individus: {y_train.count()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Models et pipelines""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### **Dummy classifier "stratified"**

    Le DummyClassifier avec la stratégie "stratified" est un modèle de référence qui ne cherche pas à apprendre à partir des données. Il génère des prédictions aléatoires tout en respectant la distribution des classes observée dans l’ensemble d’entraînement.

    Par exemple, si 75 % des observations appartiennent à la classe A et 25 % à la classe B, il prédira la classe A avec une probabilité de 75 % et la classe B avec 25 %.

    Ce modèle est utilisé comme baseline : il permet de vérifier si un modèle plus complexe apporte une vraie valeur ajoutée. Si un modèle ne fait pas mieux que ce classificateur “naïf”, cela peut révéler des problèmes dans les données ou l’approche choisie.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### **Régression logistique**

    La régression logistique est un modèle de classification binaire simple et robuste. Elle prédit la probabilité qu’une observation appartienne à la classe positive, en appliquant une fonction sigmoïde à une combinaison linéaire des variables explicatives.

    La sortie est une probabilité entre 0 et 1, et un seuil (souvent 0.5) est utilisé pour prendre une décision de classification.

    Ce modèle est particulièrement apprécié pour :

    - sa simplicité d’implémentation,

    - sa vitesse d’entraînement,

    - et sa facilité d’interprétation des coefficients (ce qui est utile pour comprendre l’impact de chaque variable).

    Il constitue souvent un bon point de départ dans un projet de classification.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### **LightGBM**

    LightGBM (Light Gradient Boosting Machine) est un algorithme de gradient boosting développé par Microsoft, réputé pour sa rapidité et son efficacité mémoire.

    Contrairement à d'autres implémentations comme XGBoost, LightGBM construit ses arbres par feuille ("leaf-wise"), ce qui permet d’obtenir de meilleures performances sur des données complexes.

    Ses principaux atouts sont :

    - une excellente vitesse d’exécution, même sur de gros volumes de données,

    - une consommation mémoire optimisée

    LightGBM est particulièrement adapté aux projets nécessitant des performances élevées et une scalabilité importante.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Création de séquence d'entrainement pour chacun des models.

    /// admonition | Rappel
        type: warning

        Contrairement à régression, logistique, LightGBM ne nécessite pas d'encoder les variables catégorielles.

        Je vais définir des Columns Transformers différents. 

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    | Étape                    | Dummy Classifier         | Régression Logistique | LightGBM (avec SMOTE) | LightGBM (sans SMOTE) |
    | ------------------------ | ------------------------ | --------------------- | --------------------- | --------------------- |
    | **Imputation (médiane)** | ✅                        | ✅                     | ✅                     | ❌                     |
    | **SMOTE (oversampling)** | ✅                        | ✅                     | ✅                     | ❌                     |
    | **RandomUnderSampler**   | ✅                        | ✅                     | ✅                     | ❌                     |
    | **Standardisation**      | ❌                        | ✅ *(StandardScaler)*  | ❌ *(pas nécessaire)*  | ❌ *(pas nécessaire)*  |
    | **Modèle utilisé**       | DummyClassifier (strat.) | LogisticRegression    | LGBMClassifier        | LGBMClassifier        |
    """
    )
    return


@app.cell
def _(
    DummyClassifier,
    LGBMClassifier,
    LogisticRegression,
    Pipeline,
    SimpleImputer,
    StandardScaler,
    oversampler,
    undersampler,
):
    # Création de séquences d'entrainements pour chacun des modèles
    dummy_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("oversampler", oversampler),
        ("undersampler", undersampler),
        ("model", DummyClassifier(strategy="stratified", random_state=42)),
    ])

    log_reg_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("oversampler", oversampler),
        ("undersampler", undersampler),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ])

    lgbm_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("oversampler", oversampler),
        ("undersampler", undersampler),
        ("model", LGBMClassifier()),
    ])

    lgbm_model_sans_smote = Pipeline([
        ("model", LGBMClassifier()),
    ])
    return dummy_model, lgbm_model, lgbm_model_sans_smote, log_reg_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Optimisation des hyperparamètres""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Même si les algorithmes utilisés sont des modèles d'apprentissage automatique, des ajustement des hyperparamètres sont nécessaires pour optimiser leurs prédictions.

    Pour mes projets précédents, j'avais utilisé GridSearchCV pour itérer sur chaques hyperparamètres.

    Ici, je vais utiliser une méthode plus avancé avec Optuna.

    Optuna permet une recherche efficace dans des espaces d’hyperparamètres complexes. Contrairement à GridSearchCV qui teste systématiquement toutes les combinaisons (et devient vite trop lent), Optuna s’adapte aux résultats précédents pour orienter intelligemment la recherche, ce qui est crucial avec des modèles comme LightGBM ayant beaucoup de paramètres.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    | Modèle                 | Hyperparamètre      | Type             | Rôle / Explication                                   |
    | ---------------------- | ------------------- | ---------------- | ---------------------------------------------------- |
    | **DummyClassifier**    | `random_state`      | Fixe             | Reproductibilité                                     |
    | **LogisticRegression** | `C`                 | Float (log)      | Inverse de la régularisation, contrôle la complexité |
    |                        | `penalty`           | Categorical      | Type de régularisation : L1, L2, ElasticNet          |
    |                        | `l1_ratio`          | Float (0–1)      | Poids entre L1 et L2 si `penalty="elasticnet"`       |
    |                        | `max_iter`          | Fixe             | Nombre d'itérations max pour convergence             |
    |                        | `solver`            | Fixe (`saga`)    | Optimiseur, requis pour ElasticNet                   |
    |                        | `n_jobs`            | Fixe (-1)        | Utilisation de tous les cœurs CPU                    |
    | **LGBMClassifier**     | `boosting_type`     | Categorical      | Type de boosting : gbdt, dart, rf                    |
    |                        | `learning_rate`     | Float (0.01–0.3) | Vitesse d’apprentissage                              |
    |                        | `num_leaves`        | Int (15–100)     | Nombre de feuilles max (contrôle la complexité)      |
    |                        | `max_depth`         | Int (3–10)       | Profondeur max des arbres                            |
    |                        | `min_child_samples` | Int (10–50)      | Minimum d’échantillons par feuille                   |
    |                        | `subsample`         | Float (0.5–1.0)  | Ratio de lignes utilisées (bagging)                  |
    |                        | `colsample_bytree`  | Float (0.5–1.0)  | Ratio de colonnes utilisées (feature bagging)        |
    |                        | `reg_alpha`         | Float (0.0–1.0)  | Régularisation L1                                    |
    |                        | `reg_lambda`        | Float (0.0–1.0)  | Régularisation L2                                    |
    |                        | `class_weight`      | Fixe             | Gère le déséquilibre des classes                     |
    |                        | `random_state`      | Fixe             | Reproductibilité                                     |
    |                        | `n_jobs`            | Fixe (-1)        | Utilisation des CPU                                  |
    |                        | `verbosity`         | Fixe (-1)        | Silence les logs                                     |
    """
    )
    return


@app.cell
def _():
    # Hyperparamètres à optimiser pour chacun des modèles
    dummy_params = {
        "model__random_state": lambda trial: 42,
    }
    log_reg_params = {
        "model__random_state": lambda trial: 42,
        "model__n_jobs": lambda trial: -1,
        "model__C": lambda trial: trial.suggest_float("model__C", 1e-4, 10.0, log=True),
        "model__penalty": lambda trial: trial.suggest_categorical("model__penalty", ["l2", "l1", "elasticnet"]),
        "model__l1_ratio": lambda trial: trial.suggest_float("model__l1_ratio", 0.0, 1.0),
        "model__max_iter": lambda trial: 2000,
        "model__solver": lambda trial: "saga",
    }
    lgbm_params = {
        "model__random_state": lambda trial: 42,
        "model__n_jobs": lambda trial: -1,
        "model__verbosity": lambda trial: -1,
        "model__class_weight": lambda trial: "balanced",
        "model__boosting_type": lambda trial: trial.suggest_categorical("model__boosting_type", ["gbdt", "dart", "rf"]),
        "model__learning_rate": lambda trial: trial.suggest_float("model__learning_rate", 0.01, 0.3),
        "model__num_leaves": lambda trial: trial.suggest_int("model__num_leaves", 15, 100),
        "model__max_depth": lambda trial: trial.suggest_int("model__max_depth", 3, 10),
        "model__min_child_samples": lambda trial: trial.suggest_int("model__min_child_samples", 10, 50),
        "model__subsample": lambda trial: trial.suggest_float("model__subsample", 0.5, 1.0),
        "model__colsample_bytree": lambda trial: trial.suggest_float("model__colsample_bytree", 0.5, 1.0),
        "model__reg_alpha": lambda trial: trial.suggest_float("model__reg_alpha", 0.0, 1.0),
        "model__reg_lambda": lambda trial: trial.suggest_float("model__reg_lambda", 0.0, 1.0),
    }
    return dummy_params, lgbm_params, log_reg_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Pour rendre mes expérimentations reproductibles et automatisées, j’ai créé une structure de dictionnaire models_dict qui regroupe chaque modèle, son pipeline d’entraînement, et les hyperparamètres à optimiser. 

    Cette structure me permet de passer chaque modèle à Optuna de manière transparente.
    """
    )
    return


@app.cell
def _(
    dummy_model,
    dummy_params,
    lgbm_model,
    lgbm_model_sans_smote,
    lgbm_params,
    log_reg_model,
    log_reg_params,
):
    # Créer un dictionnaire pour stocker les models et leurs paramètres
    models_dict = {
        "DummyClassifier": {
            "model": dummy_model,
            "params": dummy_params,
        },
        "LGBMClassifier": {
            "model": lgbm_model,
            "params": lgbm_params,
        },
        "LGBMClassifier sans SMOTE": {
            "model": lgbm_model_sans_smote,
            "params": lgbm_params,
        },
        "LogisticRegression": {
            "model": log_reg_model,
            "params": log_reg_params,
        },
    }
    return (models_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Optuna""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Afin de suivre plus efficacement la progression de l’optimisation par Optuna, on utilise une fonction de callback personnalisée, appelée **champion_callback**. 

    Cette fonction est exécutée à la fin de chaque essai et permet d’afficher uniquement les configurations d’hyperparamètres qui améliorent les performances du modèle par rapport aux essais précédents. 

    Concrètement, elle détecte si un essai bat le « champion » actuel (le meilleur score observé jusqu'à présent), calcule le pourcentage d'amélioration, puis l’affiche proprement dans les logs. Ce mécanisme rend l’optimisation plus lisible et permet de mieux suivre les progrès sans être noyé dans les résultats intermédiaires.
    """
    )
    return


@app.cell
def _(optuna):
    # Override optuna's default logging to CRITICAL only
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)


    # define a logging callback that will report on only new challenger parameter configurations if a
    # trial has usurped the state of 'best conditions'
    def champion_callback(study, frozen_trial):
        """Logging callback.

        Logging callback that will report when a new trial iteration improves upon existing
        best trial values.

        Note: This callback is not intended for use in distributed computing systems such as Spark
        or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
        workers or agents.
        The race conditions with file system state management for distributed trials will render
        inconsistent values with this callback.
        """
        winner = study.user_attrs.get("winner", None)

        if study.best_value and winner != study.best_value:
            study.set_user_attr("winner", study.best_value)
            if winner:
                improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
                print(
                    f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                    f"{improvement_percent: .4f}% improvement"
                )
            else:
                print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")
    return (champion_callback,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Cette fonction joue un rôle central dans l’entraînement, l’évaluation, et la journalisation des modèles, tout en intégrant l’optimisation par Optuna et le suivi expérimental via MLflow.

    Elle construit une fonction objectif compatible avec Optuna, c’est-à-dire une fonction à optimiser. À chaque essai (ou "trial"), elle :

    1. génère une nouvelle combinaison d’hyperparamètres à tester à partir du dictionnaire fourni.

    2. Applique ces paramètres à un pipeline de modèle (Pipeline).

    3. Effectue une validation croisée stratifiée (StratifiedKFold) pour évaluer la robustesse du modèle.

    4. Calcule des métriques personnalisées (ex. : test_business, roc_auc, etc.) définies par l’utilisateur.

    5. Suit et enregistre automatiquement chaque essai avec MLflow (hyperparamètres, scores, signature du modèle...).

    Ainsi, cette fonction permet une évaluation systématique, traçable et reproductible des performances des modèles entraînés, en capitalisant sur la puissance combinée d’Optuna (pour l’optimisation) et MLflow (pour le suivi).
    """
    )
    return


@app.cell
def _(StratifiedKFold, cross_validate, datetime, mlflow):
    def compute_model_scores(experiment_id, model, param_grid, model_name, features, target, scoring):
        """Create folds in the data to make a cross_validation when training the model.

        --------------------
        Arguments :
            experiment_id
            model : Pipeline : pipeline of the model to train and evaluate
            param_grid : dict : the list of parameters to apply to the model
            model_name : str : the name of the model
            features : array-like of shape (n_samples, n_features) : training data
            target : array-like of shape (n_samples,) : target variable
            scoring : dict: the scoring dict
        --------------------
        """

        def objective(trial):
            # Train model with MLflow tracking
            with mlflow.start_run(experiment_id=experiment_id, nested=True):
                # Evaluate each lambda to obtain the different parameters
                params = {param: param_fn(trial) for param, param_fn in param_grid.items()}

                # Ajustement des paramètres invalides selon le modèle
                if "model__penalty" in params and params["model__penalty"] != "elasticnet":
                    params.pop("model__l1_ratio", None)

                # Set model parameters
                model.set_params(**params)

                # Define cross-validation
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                # Perform cross-validation
                scores = cross_validate(
                    model,
                    features,
                    target,
                    cv=skf,
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1,
                )

                # Get metrics
                fit_time = scores["fit_time"].mean()
                train_business = scores["train_business"].mean()
                test_business = scores["test_business"].mean()
                train_rocauc = scores["train_rocauc"].mean()
                test_rocauc = scores["test_rocauc"].mean()

                # Set tags of the model
                mlflow.set_tag("model", model_name)
                mlflow.set_tag("mlflow.runName", f"{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                mlflow.set_tag("search_method", "Optuna")
                mlflow.set_tag("trial_number", trial.number)
                mlflow.set_tag("dataset_version", "v1.0")

                # Log the best params found
                mlflow.log_params(params)

                # Log metrics
                mlflow.log_metric("fit_time", fit_time)
                mlflow.log_metric("train_business", train_business)
                mlflow.log_metric("test_business", test_business)
                mlflow.log_metric("train_ROCAUC", train_rocauc)
                mlflow.log_metric("test_ROCAUC", test_rocauc)

            return test_business

        return objective
    return (compute_model_scores,)


@app.cell(hide_code=True)
def _(mo):
    run_button = mo.ui.run_button(
        kind="warn", full_width=True, label="Click to find optimals hyperparameters. ⚠️ Caution ! Can take 6 hours"
    )
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def _(
    ARTIFACT_LOCATION,
    OPTUNA_DIR,
    StratifiedKFold,
    SuccessiveHalvingPruner,
    TPESampler,
    X_train,
    X_val,
    champion_callback,
    compute_model_scores,
    cross_validate,
    get_or_create_experiment,
    infer_signature,
    mlflow,
    mo,
    models_dict,
    optuna,
    run_button,
    scoring,
    y_train,
):
    mo.stop(not run_button.value, mo.md("⚠️ **Long running process.**<br> Click 👆 to run this cell").callout(kind="warn"))
    # Boucle sur tous les modèles du dictionnaire
    for _name, _infos in models_dict.items():
        # Crée un nom d’étude unique par modèle
        _study_name = f"credit_scoring_{_name.replace(' ', '_')}"

        # Chemin complet pour la base de données Optuna par modèle
        _db_path = OPTUNA_DIR / f"{_study_name}.db"
        _storage_name = f"sqlite:///{_db_path}"

        # Crée (ou récupère) un experiment MLflow nommé "credit_scoring"
        _experiment_id = get_or_create_experiment("credit_scoring", artifact_location=ARTIFACT_LOCATION)

        # Lance une exécution MLflow pour ce modèle, dans l'expérience définie plus haut
        # nested=True : cette exécution est imbriquée (utile pour structurer les runs avec Optuna)
        with mlflow.start_run(experiment_id=_experiment_id, run_name=_name, nested=True):
            # Enregistre la version du dataset utilisé
            mlflow.set_tag("dataset_version", "v1.0")

            # Crée une fonction objectif pour Optuna, adaptée au modèle et ses hyperparamètres
            _objective = compute_model_scores(
                experiment_id=_experiment_id,
                model=_infos["model"],  # le pipeline (avec preprocessing + estimator)
                param_grid=_infos["params"],  # le dictionnaire des hyperparamètres à explorer
                model_name=_name,  # nom du modèle (pour tagging et suivi)
                features=X_train,
                target=y_train,  # données d'entraînement
                scoring=scoring,  # dictionnaire des métriques de validation croisée
            )

            # Initialise une étude Optuna pour maximiser la métrique cible (business score)
            _study = optuna.create_study(
                direction="maximize",
                pruner=SuccessiveHalvingPruner(),
                study_name=_study_name,
                sampler=TPESampler(seed=42),
                storage=_storage_name,
                load_if_exists=True,
            )

            # Nombre d'essais selon le modèle
            _n_trials = 1 if _name == "DummyClassifier" else 20

            # Lancement de l'optimisation avec la fonction objective et un callback
            _study.optimize(_objective, n_trials=_n_trials, callbacks=[champion_callback], show_progress_bar=True)

            # Récupère les meilleurs paramètres trouvés par Optuna
            _best_params = _study.best_params

            # Loggue les meilleurs paramètres et la meilleure valeur de la métrique business
            mlflow.log_params(_best_params)
            mlflow.log_metric("best_business", _study.best_value)

            # Réévaluation complète avec cross-validation pour toutes les métriques
            _best_model = _infos["model"].set_params(**_best_params)
            _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            _scores = cross_validate(
                _best_model, X_train, y_train, cv=_skf, scoring=scoring, return_train_score=True, n_jobs=-1
            )

            # Log des métriques
            mlflow.log_metric("fit_time", _scores["fit_time"].mean())
            mlflow.log_metric("train_business", _scores["train_business"].mean())
            mlflow.log_metric("test_business", _scores["test_business"].mean())
            mlflow.log_metric("train_ROCAUC", _scores["train_rocauc"].mean())
            mlflow.log_metric("test_ROCAUC", _scores["test_rocauc"].mean())
            mlflow.log_metric("train_precision", _scores["train_precision"].mean())
            mlflow.log_metric("test_precision", _scores["test_precision"].mean())
            mlflow.log_metric("train_recall", _scores["train_recall"].mean())
            mlflow.log_metric("test_recall", _scores["test_recall"].mean())
            mlflow.log_metric("train_f1", _scores["train_f1_score"].mean())
            mlflow.log_metric("test_f1", _scores["test_f1_score"].mean())

            _best_model.fit(X_train, y_train)

            # Déduit la signature d'entrée/sortie du modèle (utile pour la mise en production)
            _signature = infer_signature(
                X_val,
                _best_model.predict(X_val),
                params=_best_params,
            )

            # Loggue le modèle entraîné dans MLflow avec sa signature et un exemple d’entrée
            mlflow.sklearn.log_model(
                _best_model,
                _name,
                signature=_signature,
                registered_model_name=_name,
                input_example=X_val.head(1),
            )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Evaluation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    | Indicateur         | Description                                                                 | Intérêt                                                                                       |
    |--------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
    | **test_business**  | Score métier calculé sur l'ensemble de test.                               | Principal objectif à maximiser pour répondre aux besoins métiers spécifiques.                |
    | **train vs test gap** | Écart entre les performances sur les ensembles d'entraînement et de test. | Un faible écart indique une bonne généralisation du modèle aux nouvelles données.            |
    | **test_ROCAUC**    | Aire sous la courbe ROC pour l'ensemble de test.                           | Valeur > 0.75 généralement recherchée pour une bonne discrimination entre les classes.       |
    | **fit_time**       | Temps nécessaire pour entraîner le modèle.                                 | Doit être adapté aux contraintes opérationnelles et de déploiement.                          |
    | **test_f1**        | Score F1, moyenne harmonique de la précision et du rappel, sur le test.    | Indique une bonne balance entre précision et rappel, crucial pour les jeux déséquilibrés.    |
    """
    )
    return


@app.cell
def _(mlflow, pd):
    # Connexion au client MLflow
    client = mlflow.MlflowClient()

    # Récupère tous les modèles enregistrés dans le registre MLflow
    _registered_models = mlflow.search_registered_models()

    # Récupère les infos (nom, version, métriques, etc.)
    _model_records = []

    for _model in _registered_models:
        for _version in _model.latest_versions:
            try:
                _run_id = _version.run_id
                _run = client.get_run(_run_id)
                _model_records.append({
                    "model_name": _model.name,
                    "version": _version.version,
                    "run_id": _run_id,
                    "test_business": _run.data.metrics.get("test_business"),
                    "train_business": _run.data.metrics.get("train_business"),
                    "test_ROCAUC": _run.data.metrics.get("test_ROCAUC"),
                    "train_ROCAUC": _run.data.metrics.get("train_ROCAUC"),
                    "test_f1": _run.data.metrics.get("test_f1"),
                    "train_f1": _run.data.metrics.get("train_f1"),
                    "fit_time": _run.data.metrics.get("fit_time"),
                })
            except Exception as e:
                print(f"Erreur lors du traitement de la version {_version.version} du modèle {_model.name}: {e}")

    # Convertir en DataFrame
    df_models = pd.DataFrame(_model_records)

    # Nettoyage : tri par test_business décroissant
    df_models = df_models.sort_values(by="test_business", ascending=False)

    # Aperçu des modèles
    df_models
    return (df_models,)


@app.cell
def _(df_models, plt, sns):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_models, x="train_business", y="test_business", hue="model_name", s=100)
    plt.plot(
        [df_models.train_business.min(), df_models.train_business.max()],
        [df_models.train_business.min(), df_models.train_business.max()],
        "--",
        color="gray",
    )
    plt.title("Business Score: Train vs Test")
    plt.xlabel("Train Business")
    plt.ylabel("Test Business")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_models, plt, sns):
    plt.figure(figsize=(10, 5))
    _ax = sns.barplot(data=df_models, x="model_name", y="test_business", hue="version")

    # Ajoute valeurs
    for _container in _ax.containers:
        _ax.bar_label(_container, fmt="%.3f")

    plt.title("Test Business Score par modèle enregistré")
    plt.ylabel("Test Business Score")
    plt.xlabel("Modèle")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_models, plt, sns):
    plt.figure(figsize=(10, 5))
    _ax = sns.barplot(data=df_models, x="model_name", y="test_ROCAUC", hue="version")

    # Ajoute valeurs
    for _container in _ax.containers:
        _ax.bar_label(_container, fmt="%.3f")

    plt.title("Test ROCAUC par modèle enregistré")
    plt.ylabel("Test ROCAUC")
    plt.xlabel("Modèle")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_models, plt, sns):
    plt.figure(figsize=(10, 5))
    _ax = sns.barplot(data=df_models, x="model_name", y="fit_time", hue="version")

    # Ajoute valeurs
    for _container in _ax.containers:
        _ax.bar_label(_container, fmt="%.3f")

    plt.title("fit time par modèle enregistré (s)")
    plt.ylabel("fit_time")
    plt.xlabel("Modèle")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_models, plt, sns):
    plt.figure(figsize=(10, 5))
    _ax = sns.barplot(data=df_models, x="model_name", y="test_f1", hue="version")

    # Ajoute valeurs
    for _container in _ax.containers:
        _ax.bar_label(_container, fmt="%.3f")

    plt.title("Test F1 par modèle enregistré")
    plt.ylabel("Test F1")
    plt.xlabel("Modèle")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Model Selectionné
        type: success

    **LGBMClassifier sans SMOTE**
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    | Modèle                      | Test Business | Train vs Test Gap | Test ROCAUC | Fit Time (s) | Test F1 | Analyse synthétique |
    |----------------------------|---------------|-------------------|-------------|--------------|---------|----------------------|
    | **LGBMClassifier sans SMOTE** | **0.689**      | Faible            | **0.749**    | 83.26        | 0.270   | ✅ Meilleur score métier, meilleure ROC AUC, bon compromis général |
    | LogisticRegression         | 0.645         | Faible-moyen      | 0.742       | 🔴 349.86     | **0.284** | Bon F1, mais temps d'entraînement très élevé |
    | LGBMClassifier             | 0.638         | Moyen             | 0.713       | ✅ 47.23      | 0.254   | Score global plus faible, discrimination correcte |
    | DummyClassifier            | 0.502         | Faible (modèle naïf) | 0.499       | 🟢 5.36       | 0.129   | Modèle de base, performances très limitées |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Le modèle "LGBMClassifier sans SMOTE" est le plus performant selon :

    - Le score métier (test) le plus élevé (0.689)

    - Le meilleur ROCAUC (0.749)

    - Un temps d'entraînement raisonnable

    - Une bonne généralisation (gap faible entre train/test)

    - Un score F1 correct, bien qu’un peu en dessous de "LogisticRegression"
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Optuna hyperparamètres""")
    return


@app.cell
def _(OPTUNA_DIR, optuna):
    # Chemin vers la base de données Optuna
    _study_name = "credit_scoring_LGBMClassifier_sans_SMOTE"
    _db_path = OPTUNA_DIR / f"{_study_name}"
    _storage = f"sqlite:///{_db_path}.db"

    # Charger l'étude
    _study = optuna.load_study(study_name=_study_name, storage=_storage)

    # Récupérer les meilleurs hyperparamètres
    best_params = _study.best_trial.params

    print("Meilleurs hyperparamètres :")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    return


@app.cell
def _(mo):
    mo.image(
        "notebooks/plots/hyperparamter_importance.png",
        width=700,
        height=500,
        rounded=True,
        caption="Hyperparamters importance LightGBM_sans_smote",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Optimisation du seuil de décision""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Maintenant que j'ai déterminer le modèle le plus performant avec ses hyperpramètres associés.

    Voyons si je peux optimiser optimiser le seuil de décision qui fournira le meilleur score métier
    """
    )
    return


@app.cell
def _(
    StratifiedKFold,
    cross_val_predict,
    normalized_business_score,
    np,
    plt,
    precision_recall_curve,
):
    def optimize_threshold_post_training(model, features, target, thresholds=np.arange(0.0, 1.0, 0.01)):
        # Define cross-validation splitter
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Get out-of-fold probabilities (only the class 1 proba)
        oof_proba = cross_val_predict(
            model,
            features,
            target,
            cv=skf,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]  # Take the probability of the positive class

        scores = []
        for t in thresholds:
            y_pred = (oof_proba >= t).astype(int)
            score = normalized_business_score(target, y_pred)
            scores.append(score)

        scores = np.array(scores)
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]

        # Score à seuil 0.5 pour vérification
        score_05 = scores[50]

        precision, recall, threshold = precision_recall_curve(target, oof_proba)

        print(f"Score business à seuil 0.5 (référence) : {score_05:.3f}")
        print(f"Seuil optimal trouvé : {best_threshold:.2f} avec score {best_score:.3f}")

        # Affichage
        plt.figure(figsize=(10, 6))
        plt.plot(threshold, precision[:-1], label="precision")
        plt.plot(threshold, recall[:-1], label="recall")
        plt.plot(thresholds, scores, label="Business score")
        plt.axvline(x=0.5, color="red", linestyle="--", label="Seuil 0.5")
        plt.axvline(x=best_threshold, color="green", linestyle="--", label="Seuil optimal")
        plt.title("Optimisation du seuil post-entrainement")
        plt.xlabel("Seuil de classification")
        plt.ylabel("Business score")
        plt.legend()
        plt.grid(True)
        plt.show()

        return best_threshold, best_score
    return (optimize_threshold_post_training,)


@app.cell
def _(X_train, mlflow, optimize_threshold_post_training, y_train):
    # Configuration MLflow
    _model_name = "LGBMClassifier sans SMOTE"
    _model_version = "latest"
    _target_col = "TARGET"

    # Charger le modèle depuis le Model Registry
    _model_uri = f"models:/{_model_name}/{_model_version}"
    _model = mlflow.sklearn.load_model(_model_uri)

    # Optimisation du seuil via KFold
    best_threshold, best_score = optimize_threshold_post_training(_model, X_train, y_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Conclusion sur le seuil de classification

        L'analyse du graphique révèle que le seuil de classification par défaut de 0,5 produit un score business de optimum.

    L'utilisation d'Optuna pour l'optimisation des hyperparamètres a implicitement ajusté le seuil de classification, rendant les modifications supplémentaires du seuil peu bénéfiques pour l'amélioration du score business. 

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Analyse de l'importance des caractéristiques""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - SHAP (SHapley Additive exPlanations):

    C' est une méthode d'interprétation des modèles de machine learning qui utilise la théorie des valeurs de Shapley pour expliquer l'impact de chaque feature sur les prédictions.

    Elle permet d'analyser l'importance des features à la fois globalement (importance moyenne) et localement (explications pour une prédiction spécifique).
    """
    )
    return


@app.cell
def _(X_val, mlflow, shap):
    # Configuration MLflow
    _model_name = "LGBMClassifier sans SMOTE"
    _model_version = "latest"
    _target_col = "TARGET"

    # Charger le modèle depuis le Model Registry
    _model_uri = f"models:/{_model_name}/{_model_version}"
    best_model = mlflow.sklearn.load_model(_model_uri)

    # Récupère uniqument le model lgbm du pipeline
    _final_lgb_model = best_model.named_steps["model"]

    # Calcul des valeurs SHAP
    _explainer = shap.Explainer(_final_lgb_model)
    shap_values = _explainer(X_val)
    return best_model, shap_values


@app.cell
def _(plt, shap):
    def predict_class_with_shap(client_idx: int, X_test, model, optimal_threshold: float = 0.5):
        """
        Predict the class for a given client by applying an optimized threshold and display SHAP explanations.

        Parameters:
        - client_idx: Index of the client in the dataset.
        - X_test: features data for prediction
        - model: Model encapsulated in a pipeline (including preprocessing steps and the final model).
        - optimal_threshold: Optimized threshold to use for classification, based on predicted probabilities by the model.

        Returns:
        - Tuple predicted probability, and class after applying the optimal threshold.
        """
        # Display client number
        print("Client number:", X_test.iloc[client_idx, 0])

        # Extract the final model (LightGBM) from the pipeline
        lgbm_model = model.named_steps["model"]

        # Predict the probability for the client via the model
        y_pred_proba = lgbm_model.predict_proba(X_test)[:, 1]  # Probability of non-repayment

        # Apply the optimized threshold
        y_pred_custom_threshold = (y_pred_proba[client_idx] > optimal_threshold).astype(int)

        # Display predictions
        print(f"There is {y_pred_proba[client_idx]:.1%} risk that the client will have payment difficulties.")
        print(f"Class after applying the optimized threshold ({optimal_threshold}): {y_pred_custom_threshold}")

        # Calcul des valeurs SHAP
        _explainer = shap.Explainer(lgbm_model)
        shap_values = _explainer(X_test)

        # Display SHAP waterfall plot
        plt.figure()
        shap.plots.waterfall(shap_values[client_idx], max_display=15)
        plt.close()

        plt.figure()
        shap.plots.bar(shap_values[0], max_display=15)
        plt.close()

        return y_pred_proba[client_idx], y_pred_custom_threshold
    return (predict_class_with_shap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Fetaures importances locales""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- client pour qui le prêt est refusé""")
    return


@app.cell
def _(X_val, best_model, predict_class_with_shap):
    predict_class_with_shap(0, X_val, best_model, optimal_threshold=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Client pour qui le prêt est accordé""")
    return


@app.cell
def _(X_val, best_model, predict_class_with_shap):
    predict_class_with_shap(42, X_val, best_model, optimal_threshold=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Feature importance globales""")
    return


@app.cell
def _(np, pd, shap_values):
    _cohorts = {"": shap_values}
    _cohort_labels = list(_cohorts.keys())
    _cohort_explanations = list(_cohorts.values())

    # Calculer la moyenne absolue des valeurs SHAP si en 2D
    for _idx, _explanation in enumerate(_cohort_explanations):
        if _explanation.values.ndim == 2:
            _cohort_explanations[_idx] = _explanation.abs.mean(axis=0)

    # Supposer que tous les cohorts ont les mêmes features
    _features = _cohort_explanations[0].data
    _feature_names = _cohort_explanations[0].feature_names

    # Extraire les valeurs SHAP dans un tableau numpy
    _values = np.array([_exp.values for _exp in _cohort_explanations])

    # Calculer l'importance globale (somme des SHAP absolues)
    _importances = _values.sum(axis=0)

    # Construction du DataFrame des importances
    feature_importance = pd.DataFrame({"features": _feature_names, "importance": _importances}).sort_values(
        by="importance", ascending=False
    )

    # Affichage des 15 features les plus importantes
    feature_importance.head(15)
    return (feature_importance,)


@app.cell
def _(plt, shap, shap_values):
    plt.figure()
    shap.plots.bar(shap_values, max_display=15)
    plt.close()
    return


@app.cell
def _(plt, shap, shap_values):
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=15)
    plt.close()
    return


@app.cell
def _(pd, shap, shap_values):
    shap_vals_array = shap_values.values  # shape: (n_samples, n_features)

    # Extraire le DataFrame des features (il garde les noms)
    X_features = pd.DataFrame(shap_values.data, columns=shap_values.feature_names)

    # Dependence Plot for a specific feature
    _feature_name = "EXT_SOURCE_3"

    shap.dependence_plot(
        _feature_name,
        shap_vals_array,
        X_features,
        interaction_index=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Data drift""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Informations importantes
        type: info
    > Comme vous avez pu le constater le cycle de vie du modèle n’est pas complet, j’ai oublié dans la démarche MLOps la dernière étape de suivi de la performance du modèle en production. C’est un peu normal car le modèle n’est pas encore en production !


    > En prévision, je souhaiterais que vous testiez l’utilisation de la librairie evidently pour détecter dans le futur du Data Drift en production. Pour cela vous prendrez comme hypothèse que le dataset “application_train” représente les datas pour la modélisation et le dataset “application_test” représente les datas de nouveaux clients une fois le modèle en production.

    ///
    """
    )
    return


@app.cell
def _(
    DataDriftPreset,
    Report,
    app_test_encoded,
    app_train_encoded,
    feature_importance,
):
    _features = feature_importance["features"].head(15).to_list()

    # Créer le rapport Evidently (v0.3.x / v0.4.x)
    _report = Report(
        metrics=[DataDriftPreset()],
    )

    # Exécuter le rapport (référence en 1er, courant en 2ème)
    # _report.run(reference_data=_app_train_df, current_data=_app_test_df)

    _report.run(reference_data=app_train_encoded[_features], current_data=app_test_encoded[_features])

    # Visualiser dans Notebook (optionnel)
    _report.show(mode="inline")

    # Sauvegarder le Dashboard HTML
    _report.save_html("data/evidently/data_drift_dashboard.html")
    print("Dashboard saved: data_drift_dashboard.html")
    return


@app.cell
def _(mo):
    with open("data/evidently/data_drift_dashboard.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    mo.iframe(html_content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Encapsulation du model dans une API""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    | À faire                                          | Où ?                                     |
    | ------------------------------------------------ | ---------------------------------------- |
    | Encodage des colonnes                            | Dans l’API                               |
    | Nettoyage des noms de colonnes                   | Dans l’API                               |
    | Alignement des colonnes avec le modèle           | Dans l’API                               |
    | Sauvegarder la liste des colonnes d’entraînement | Depuis le notebook, à charger dans l’API |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Sauvegarde du model et du pipeline mais dans mon cas, pas d'imputation ou de standardisation des valeurs numériques.""")
    return


@app.cell
def _(best_model, pickle):
    _file_path = "model/Best_LGBM_Model.pkl"
    with open(_file_path, "wb") as _file:
        pickle.dump(best_model, _file)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Sauvegarde de l'encodage des variables catégorielles binaires et multiples qui n'étaient pas dans le pipeline.
    - Alignement des colonnes avec le modèle.


    ///admonition | Voir Section **Encodage des variables catégorielles**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Alignement des colonnes avec le modèle.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Enregistrement d'un échantillon de données de test au format csv""")
    return


@app.cell
def _(app_test_df_filtered):
    app_test_df_filtered.sort_values(by="SK_ID_CURR").head(1000).to_csv("model/customers_data.csv", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Test du code pour l'API""")
    return


@app.cell
def _(Path, logging, pd, pickle):
    # Logging config
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # --- Load model artifacts ---
    MODELS_DIR = Path("model")

    MODEL_PATH = MODELS_DIR / "Best_LGBM_Model.pkl"
    ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
    FEATURES_PATH = MODELS_DIR / "model_features.pkl"
    CUSTOMERS_PATH = MODELS_DIR / "customers_data.csv"

    with open(MODEL_PATH, "rb") as _f:
        model = pickle.load(_f)

    with open(ENCODERS_PATH, "rb") as _f:
        encoders = pickle.load(_f)

    with open(FEATURES_PATH, "rb") as _f:
        model_features = pickle.load(_f)

    customers_df = pd.read_csv(CUSTOMERS_PATH, index_col="SK_ID_CURR")

    print("Model, encoders and customer data loaded successfully")
    print(customers_df.index[:10])
    return customers_df, encoders, logger, model, model_features


@app.cell
def _(customers_df, encoders, logger, model_features, pd):
    def preprocess_client(client_id: int):
        if client_id not in customers_df.index:
            raise ValueError(f"Client ID {client_id} not found")

        client_data = customers_df.loc[[client_id]].copy()

        # Appliquer les encoders
        for col, enc in encoders.items():
            if col in client_data.columns:
                if hasattr(enc, "transform"):
                    try:
                        transformed = enc.transform(client_data[[col]])
                        if transformed.ndim == 1:  # LabelEncoder
                            client_data[col] = transformed
                        else:  # OneHotEncoder
                            ohe_df = pd.DataFrame(
                                transformed,
                                columns=[f"{col}_{c}" for c in enc.categories_[0]],
                                index=client_data.index,
                            )
                            client_data.drop(columns=[col], inplace=True)
                            client_data = pd.concat([client_data, ohe_df], axis=1)
                    except Exception as e:
                        logger.error(f"Encoding failed for column {col}: {e}")
                        raise

        # Colonnes manquantes
        missing_cols = [c for c in model_features if c not in client_data.columns]
        for c in missing_cols:
            client_data[c] = 0

        client_data = client_data[model_features]

        return client_data
    return (preprocess_client,)


@app.cell
def _(customers_df, preprocess_client):
    test_id = customers_df.index[0]  # premier client
    X = preprocess_client(test_id)
    print("Shape of input:", X.shape)
    print("First row:\n", X.head())
    return X, test_id


@app.cell
def _(X, model, test_id):
    proba = model.predict_proba(X)[0, 1]
    prediction = int(proba > 0.5)

    print(f"Client: {test_id}")
    print(f"Prediction (0=no default, 1=default): {prediction}")
    print(f"Probability of default: {proba:.4f}")
    return


@app.cell
def _(pd):
    _df = pd.read_csv("model/customers_data.csv")
    print(_df.head())
    print(_df["SK_ID_CURR"].head())
    print(100002 in _df["SK_ID_CURR"].values)
    return


if __name__ == "__main__":
    app.run()
