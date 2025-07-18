import marimo

__generated_with = "0.14.11"
app = marimo.App(app_title="P7 modelisation")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Implémenter un modèle de scoring""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Objectif du projet""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Définition du scope:

    1. **Enterprise** : Prêt à dépenser
    2. **Activité** : Proposition de crédit à la consommation (Personne ayant peu ou pas du tout d’historique de prêt)


    **Mission** :

    1. Proposer un système de notation pour accorder ou non un crédit à un client.

    **Objectif** :

    1. **Calculer la probabilité qu’un client rembourse**.
    2. **Classifier la demande en crédit accordé ou refusé.**
    3. **Classifier la demande de crédit en type accordée ou refusée (binaire)**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Démarche complète du projet de scoring de crédit

    | Étape | Objectif | Outil / Méthode | Détail de l'action |
    |-------|----------|------------------|---------------------|
    | 1️⃣ | Préparation des données | Kaggle Kernel: `HomeCredit_columns_description.csv` | Utilisation du kernel pour sélectionner, nettoyer et préparer les données |
    | 2️⃣ | Modélisation + tracking | **MLFlow** + **Ngrok** | Lancement de MLflow localement + tunnel Ngrok pour traquer les expérimentations |
    | 3️⃣ | Interface utilisateur | **Streamlit** | Création d'une interface simple pour soumettre les caractéristiques d'un client |
    | 4️⃣ | API backend | **FastAPI** | Mise en place d’une API REST pour exposer le modèle de scoring |
    | 5️⃣ | Versioning | **Git / GitHub** | Versionnage du code localement (Git) et à distance (GitHub) |
    | 6️⃣ | Déploiement API | **Render** (FastAPI) | Hébergement de l’API backend sur Render |
    | 7️⃣ | Déploiement UI | **Render** (Streamlit) | Hébergement de l’interface front-end Streamlit |
    | 8️⃣ | Keep-Alive API (anti-sommeil) | **cron-job.org** | Création d'un cronjob toutes les 10 minutes pour envoyer une requête GET à l'API et éviter qu'elle ne "dorme" en plan gratuit |
    | ✅ | CI/CD (intégration et déploiement continus) | **GitHub Actions**  | Pour automatiser les déploiements  |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Préparation de l'environnement d'éxpérimentation MLFlow""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Initialisation de l'environement **MLFlow** pour le tracking pendant l'entraînement
    - Configurer **MLFlow** pour suivre les expérimentations.
    - Initialiser le **serveur MLFlow** et **connecter le notebook au serveur**.

    **MLFlow** est une plateforme open-source conçue pour **gérer le cycle de vie des projets de machine learning**. Elle offre plusieurs fonctionnalités clés :

    1. **Tracking des expérimentations** : Permet de suivre les paramètres, métriques, artefacts (comme les fichiers modèles) et résultats des expérimentations.

    2. **Model Registry** : Centralise **le stockage** des modèles avec versioning pour faciliter leur gestion et déploiement.

    3. **Projects** : Standardise l'emballage du code ML pour le rendre facilement reproductible et partageable.

    4. **Models** : Fournit un format standard pour packager les modèles afin de simplifier leur déploiement.

    **MLFlow** est particulièrement utile dans une démarche **MLOps**, car il automatise le suivi, le stockage et le déploiement des modèles tout en assurant leur traçabilité.

    -------------------------------------------------------------------------------

    **MLOps** signifie **Machine Learning Operations**. C'est une pratique qui combine les principes de DevOps (Development and Operations) avec le développement et la gestion des modèles de machine learning. Voici ce que MLOps implique :

    - **ML** : Machine Learning
    - **Ops** : Operations

    **MLOps vise à** :

    1. Automatiser le cycle de vie des modèles ML, de l'expérimentation à la production.

    2. Assurer la reproductibilité des expérimentations.

    3. Gérer le déploiement, la surveillance, et la maintenance des modèles en production.

    4. Faciliter la collaboration entre les équipes de développement, de données, et d'opérations.

    4. Optimiser les processus pour réduire le temps de mise en production des modèles.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Lancement de MLFlow""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Installation des dépendances nécéssaires pour établir la connexion avec MLFlow""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Démarrage d'un serveur MLflow accessible via Ngrok""")
    return


@app.cell
def _():
    # Installation de la librairie relative à Ngrok
    from pyngrok import ngrok
    import mlflow
    from mlflow.tracking import MlflowClient
    import os  # Import du module os
    import time

    # 1. Configuration ngrok
    ngrok.set_auth_token("token")

    # 2. Démarrage MLflow avant ngrok
    os.makedirs("./mlruns", exist_ok=True)

    # Démarrage MLflow avec l'option --serve-artifacts pour activer le stockage
    os.system(
        "mlflow server --backend-store-uri file://./mlruns "
        "--default-artifact-root ./mlruns "
        "--host 0.0.0.0 --port 5000 "
        "--serve-artifacts &"
    )

    # Attente de 5 secondes que le serveur démarre
    time.sleep(5)

    # 3. Créer le tunnel ngrok après le démarrage de MLflow en spécifiant le domaine réservé
    public_url = ngrok.connect(addr="5000", proto="http", domain="credit.ngrok.app", bind_tls=True)
    print(f"URL MLflow publique : {public_url.public_url}")

    # 4. Vérifier la connexion manuellement
    print("Vérifiez que MLflow est accessible via :", public_url.public_url)
    return MlflowClient, mlflow, ngrok, os, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##### 1.3 **Configuration et Lancement du Serveur MLflow avec Tunnel Ngrok**""")
    return


@app.cell
def _(MlflowClient, mlflow, ngrok, os, time):
    ngrok.set_auth_token('')
    os.makedirs('./mlruns', exist_ok=True)
    os.system('mlflow server --backend-store-uri file://./mlruns --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 --serve-artifacts &')
    time.sleep(5)
    public_url_1 = ngrok.connect(addr='5000', proto='http', domain='credit.ngrok.app', bind_tls=True)
    print(f'\nInterface MLflow disponible à : {public_url_1.public_url}')
    mlflow.set_tracking_uri(public_url_1.public_url)
    client = MlflowClient()
    return (public_url_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. **Chargement et Analyse des données pour établir " l'Outil de Scoring Crédit"**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.1 **Chargement & Lecture des données**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Tout d'abord, nous listons tous les fichiers de données disponibles.
    - Il y a un total de 10 fichiers _détail des fichiers cléfs_ :

    1. fichier principal pour **l'entraînement** : contient la variable cible **(TARGET)**.

    1. fichier principal pour **le test** : ne contient pas la variable cible.

    1. fichier d'exemple de **soumission** : utilisé pour soumettre les prédictions.

    6. fichiers **supplémentaires** : fournissent des **informations complémentaires** sur chaque prêt.
    """
    )
    return


@app.cell
def _(os_1):
    file_paths = ['./HomeCredit_columns_description.csv', './POS_CASH_balance.csv', './application_test.csv', './application_train.csv', './bureau.csv', './bureau_balance.csv', './credit_card_balance.csv', './installments_payments.csv', './previous_application.csv', './sample_submission.csv']
    file_status = {file_path: os_1.path.exists(file_path) for file_path in file_paths}
    print(file_status)
    return


@app.cell
def _():
    # numpy et pandas pour la manipulation des données
    import numpy as np
    import pandas as pd

    # sklearn preprocessing pour gérer les variables catégorielles
    from sklearn.preprocessing import LabelEncoder

    # Gestion du système de fichiers

    # Suppression des avertissements
    import warnings
    warnings.filterwarnings('ignore')

    # matplotlib et seaborn pour les graphiques
    import matplotlib.pyplot as plt
    import seaborn as sns
    return LabelEncoder, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.2 **Analyse générale des données**""")
    return


@app.cell
def _(pd):
    # Chargement du fichier explicatifs des données mis à disposition pour l'entraînement et le test
    columns_description = pd.read_csv("HomeCredit_columns_description.csv", encoding='latin1')

    # Afficher la forme des données
    print('Forme des données descriptives du projet : ', columns_description.shape)

    # Afficher les premières lignes sous forme de tableau
    columns_description.head()
    return


@app.cell
def _(pd):
    columns_description_1 = pd.read_csv('HomeCredit_columns_description.csv', encoding='latin1')
    print('Forme des données descriptives du projet : ', columns_description_1.shape)
    descriptions = {'SK_ID_CURR': 'ID du prêt dans notre échantillon.', 'TARGET': 'Variable cible (1 - client en difficulté de paiement : il/elle a eu un retard de paiement de plus de X jours sur au moins une des Y premières échéances du prêt dans notre échantillon, 0 - tous les autres cas)', 'NAME_CONTRACT_TYPE': 'Identification si le prêt est en espèces ou revolving.', 'CODE_GENDER': 'Genre du client.', 'FLAG_OWN_CAR': 'Indique si le client possède une voiture.'}
    for column, description in descriptions.items():
        print(f'{column}: {description}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.3 **Analyse des données _d'entraînement_ : _Les données pour l'apprentissage du modèle prédictif_**""")
    return


@app.cell
def _(pd):
    # Charger les données d'entraînement depuis le fichier CSV
    app_train = pd.read_csv(r"application_train.csv")

    # Afficher la forme des données
    print('Forme des données d\'entraînement : ', app_train.shape)

    # Afficher les #5 premières lignes sous forme de tableau
    app_train.head()
    return (app_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Les données d'entraînement contiennent 307 511 observations (chaque observation représente un prêt distinct)
    -  122 caractéristiques (variables), y compris la variable cible TARGET, qui est l'étiquette à prédire. (121 variables explicatives = X | 1 variable à prédire (target) = y
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.4 **Analyse des données de _test_: _Les données qui seront utilisées pour le test_**""")
    return


@app.cell
def _(pd):
    # Charger les données de test depuis le fichier CSV
    app_test = pd.read_csv(r"application_test.csv")
    # Afficher la forme des données
    print('Forme des données de test : ', app_test.shape)

    # Afficher les premières lignes sous forme de tableau
    app_test.head()
    return (app_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Les données de test contiennent 487.44 observations (chaque observation représente un prêt distinct)
    L'ensemble de test est considérablement plus petit et ne contient pas de colonne TARGET ce qui est normale
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. **EDA** - **Exploratory Data Analysis | Analyse Exploratoire des données**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - **Processus ouvert** : Calcul de statistiques et création de figures pour identifier tendances, anomalies, motifs ou relations dans les données.
    - **Objectif** : Comprendre ce que les données peuvent nous révéler.
    - **Démarche** : Commence par une vue d'ensemble, puis se concentre sur des zones spécifiques d'intérêt.
    - **Utilité** : Les découvertes peuvent être intéressantes en elles-mêmes ou guider les choix de modélisation, comme la sélection des caractéristiques.
    --------------------------------------------------------------------------------
    **Objectif de la prédiction** :
    1. La variable cible **(TARGET) = 0** indique si le prêt a été **remboursé à temps (0)**
    2. La variable cible **(TARGET) = 1** indique si le client a **eu des difficultés de paiement (1)**.
    --------------------------------------------------------------------------------

    - Une courbe **ROC** est une représentation graphique des performances d'un modèle de classification binaire (true/Fasle) pour tous les seuils de classification.

    - **ROC** signifie **"Receiver Operating Characteristic"** (caractéristique de fonctionnement du récepteur ou caractéristique de performance).

    - C'est une représentation graphique des performances d'un modèle de classification binaire pour tous les seuils de classification.

    - Ci-dessous une vidéo explicative du fonctionnement de la courbe de  ROC qui nous permet d'évaluaer un modèle de classifaction binaire grâce à la métrique Area Under the Curve **AUC** située entre **0 et 1** plus elle est proche de **1** plus le modèle **de classification binaire est performant**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 3.2 **Analyse initiale : Examen de la répartition du nombre de prêts dans chaque catégorie (Target = 0 Sans difficulté de paiement | Target = 1 Difficulté de paiement)**""")
    return


@app.cell
def _(app_train):
    # On affiche les colonnes des données d'entraînement qui doivent contenir la Target
    # Target = y, la variable à prédire
    app_train.columns.values
    return


@app.cell
def _(app_test):
    # On affiche les colonnes des données de test
    app_test.columns.values
    return


@app.cell
def _(app_train):
    # Calculer la distribution de la colonne TARGET
    target_counts = app_train['TARGET'].value_counts()

    # Calculer le ratio de chaque catégorie
    target_ratios = target_counts / target_counts.sum()

    # Afficher le nombre de prêts pour chaque catégorie et le ratio
    print(f"0 = nombre de {target_counts[0]} ({target_ratios[0]:.2%})")
    print(f"1 = nombre de {target_counts[1]} ({target_ratios[1]:.2%})")
    return


@app.cell
def _(app_train):
    # Tracer l'histogramme de la distribution de la colonne TARGET
    # La méthode astype(int) est utilisée pour s'assurer que les valeurs sont des entiers
    # plot.hist() crée un histogramme des valeurs de la colonne TARGET
    app_train['TARGET'].astype(int).plot.hist();
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Problème de classes déséquilibrées : Il y a beaucoup plus de **prêts remboursés(0)** à temps que de **prêts non remboursés(1)**.
    - Implications pour le modèle : Pour les modèles de machine learning plus sophistiqués, nous pourrions pondérer les classes avec **SMOTE** (_**Synthetic Minority Oversampling Technique**_) qui est une méthode utilisée pour résoudre le problème de déséquilibre des classes dans les ensembles de données. En fonction de leur représentation dans les données pour refléter cet déséquilibre.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 3.3 **Examen des Valeurs Manquantes**""")
    return


@app.cell
def _(pd):
    # Fonction pour calculer les valeurs manquantes par colonne
    def missing_values_table(df):
        # Total des valeurs manquantes
        mis_val = df.isnull().sum()

        # Pourcentage des valeurs manquantes
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # Créer un tableau avec les résultats
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Renommer les colonnes
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Valeurs Manquantes', 1: '% des Valeurs Totales'})

        # Trier le tableau par pourcentage de valeurs manquantes en ordre décroissant
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% des Valeurs Totales', ascending=False).round(1)

        # Afficher des informations résumées
        print("Votre dataframe sélectionné a " + str(df.shape[1]) + " colonnes.\n"
              "Il y a " + str(mis_val_table_ren_columns.shape[0]) +
              " colonnes qui ont des valeurs manquantes.")

        # Retourner le dataframe avec les informations sur les valeurs manquantes
        return mis_val_table_ren_columns
    return (missing_values_table,)


@app.cell
def _(app_train, missing_values_table):
    # Statistiques des valeurs manquantes
    missing_values = missing_values_table(app_train)
    missing_values.head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - **Imputation** : Lors de la construction de nos modèles de machine learning, nous devrons remplir **ces valeurs manquantes** (connu sous le nom d'imputation).

    - **Modèles avancés** : Dans des travaux ultérieurs, nous utiliserons des modèles comme **XGBoost** qui peuvent gérer les **valeurs manquantes** sans besoin d'imputation.

    - **Suppression de colonnes** : Une autre option serait de **supprimer les colonnes avec un pourcentage élevé de valeurs manquantes**, bien qu'il soit impossible de savoir à l'avance si ces colonnes seront utiles pour notre modèle.

    - **Conservation des colonnes** : Pour l'instant, nous conserverons toutes les colonnes.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 3.4 **La typologie des colonnes**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Types de Colonnes : Examinons le nombre de colonnes de chaque type de données. int64 et float64 sont **des variables numériques** (qui peuvent être discrètes ou continues). Les colonnes de type object contiennent **des chaînes de caractères et sont des caractéristiques catégorielles**.""")
    return


@app.cell
def _(app_train):
    # Nombre de colonne par typologie
    app_train.dtypes.value_counts()
    return


@app.cell
def _(app_train, pd):
    # Examinons maintenant le nombre d'entrées uniques dans chaque colonne de type object (colonnes catégorielles).
    # Nombre de classes uniques dans chaque colonne de type object
    app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - La plupart des variables catégorielles ont un nombre relativement faible d'entrées uniques.

      => Nous opterons une méthode d'encodage de ces variable via One Hot Coder de la librairie Sickit Learn pour encoder les variables catégorielles.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 3.5 **Encodage des variables catégorielles**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Encodage des Variables Catégorielles
    Problème : Les modèles de machine learning ne peuvent pas traiter directement les variables catégorielles.

    - Solutions :

    1. **Encodage par étiquette** _(Label Encoding)_ : Assignation d'un entier à chaque catégorie unique. Utile pour les variables avec **2 catégories**.

    2. **Encodage à chaud** _(One-Hot Encoding)_ : Création d'une colonne pour chaque catégorie unique. Préconisé pour **plus de 2 catégories** pour éviter les biais d'ordre arbitraire.

    - A). **Approche** : Utilisation de l'encodage par étiquette pour les variables avec 2 catégories et de l'encodage à chaud pour celles avec plus de 2 catégories.

    - B). **Outils** : LabelEncoder de Scikit-Learn pour l'encodage par étiquette et get_dummies de Pandas pour l'encodage à chaud.

    - C). **Réduction de dimension** : Envisagée pour gérer l'explosion des dimensions due à l'encodage à chaud, mais non utilisée dans ce notebook.
    """
    )
    return


@app.cell
def _(LabelEncoder, app_test, app_train):
    le = LabelEncoder()
    le_count = 0
    for col in app_train:
        if app_train[col].dtype == 'object':
            if len(list(app_train[col].unique())) <= 2:
                le.fit(app_train[col])
                app_train[col] = le.transform(app_train[col])
                app_test[col] = le.transform(app_test[col])
                le_count = le_count + 1
    print(f'{le_count} colonnes ont été encodées par étiquette.')
    return


@app.cell
def _(app_test, app_train, pd):
    app_train_1 = pd.get_dummies(app_train)
    app_test_1 = pd.get_dummies(app_test)
    print("Forme des caractéristiques d'entraînement : ", app_train_1.shape)
    print('Forme des caractéristiques de test : ', app_test_1.shape)
    print('pd.get_dummies() convertit chaque variable catégorielle en plusieurs colonnes binaires (0 ou 1), ce qui est nécessaire pour que les algorithmes de machine learning puissent traiter ces données.')
    return app_test_1, app_train_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    `Alignement des Données d'Entraînement et de Test`:

    - **Objectif** : Assurer que les caractéristiques (colonnes) sont identiques dans les données **d'entraînement** et de **test**.

    1. **Problème** : **L'encodage à chaud a créé plus de colonnes dans les données d'entraînement** car certaines variables catégorielles ont des catégories non représentées dans les données de test.

    2. **Solution** : **Aligner les dataframes** pour **supprimer les colonnes présentes dans les données d'entraînement mais absentes dans les données de test.**

    3. **Extraction de la cible** : Extraire la colonne cible **(TARGET)** des données d'entraînement **avant l'alignement**.

    4. **Alignement** : Utiliser **align** avec axis=1 pour **aligner les dataframes basés sur les colonnes, et non sur les lignes**.
    """
    )
    return


@app.cell
def _(app_test_1, app_train_1):
    train_labels = app_train_1['TARGET']
    app_train_2, app_test_2 = app_train_1.align(app_test_1, join='inner', axis=1)
    app_train_2['TARGET'] = train_labels
    print("Forme des caractéristiques d'entraînement : ", app_train_2.shape)
    print('Forme des caractéristiques de test : ', app_test_2.shape)
    return app_test_2, app_train_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. **Alignement des données** : Les datasets **d'entraînement et de test** ont maintenant **les mêmes caractéristiques**, **nécessaire pour le machine learning.**


    2. **Le nombre de colonnes dans app_test (239)** est égal à celui de **app_train (240)**, car la colonne _TARGET_ variable cible est exclut


    3. **Augmentation des caractéristiques** : Le nombre de caractéristiques a considérablement augmenté à cause de **l'encodage à chaud**. _pd.get_dummies_

    4. **Réduction de dimension** : Nous réduirons probablement la dimensionnalité (en supprimant les caractéristiques non pertinentes) pour réduire la taille des datasets.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 3.6 **Retour à l'analyse exploratoire après le processus d'encodage**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - L'objectif du retour à l'Analyse Exploratoire des données est de:

    1. **Anomalies** : Rechercher **des anomalies dans les données**, qui peuvent être dues à **des erreurs de saisie**, des **problèmes d'équipement de mesure ou des mesures extrêmes mais valides**.

    2. **Détection quantitative** : Utiliser la méthode describe pour examiner les statistiques d'une colonne.
    """
    )
    return


@app.cell
def _(app_train_2):
    print("**DAYS_BIRTH** : Les valeurs sont négatives car **elles sont enregistrées par rapport à la date de l'application de prêt actuelle**. Pour les convertir en années, **on multiplie par -1 et diviser par le nombre de jours dans une année.")
    (app_train_2['DAYS_BIRTH'] / -365).describe()
    return


@app.cell
def _(app_train_2):
    print("Cela ne semble pas correct ! La valeur maximale (en plus d'être positive) est d'environ 1000 ans !")
    app_train_2['DAYS_EMPLOYED'].describe()
    return


@app.cell
def _(app_train_2, plt):
    app_train_2['DAYS_EMPLOYED'].plot.hist(title="Histogramme de la durée d'emploi en jours")
    plt.xlabel("Durée d'emploi en jours")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Nous examinons donc les clients anormaux pour voir s'ils ont tendance à avoir des taux de défaut plus élevés ou plus bas que les autres clients.""")
    return


@app.cell
def _(app_train_2):
    anom = app_train_2[app_train_2['DAYS_EMPLOYED'] == 365243]
    non_anom = app_train_2[app_train_2['DAYS_EMPLOYED'] != 365243]
    print('Les non-anomalies font défaut sur %0.2f%% des prêts' % (100 * non_anom['TARGET'].mean()))
    print('Les anomalies font défaut sur %0.2f%% des prêts' % (100 * anom['TARGET'].mean()))
    print("Il y a %d jours d'emploi anormaux" % len(anom))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Anomalies détéctée : **Les clients avec une durée d'emploi anormale ont un taux de défaut plus bas.**

    - Gestion des anomalies :

    1. **Imputation** : Remplacer les valeurs anormales par des valeurs manquantes (np.nan) et les imputer.

    2. **Indicateur** : Créer une colonne booléenne pour indiquer si la valeur était anormale.
    """
    )
    return


@app.cell
def _(app_train_2, np, plt):
    app_train_2['DAYS_EMPLOYED_ANOM'] = app_train_2['DAYS_EMPLOYED'] == 365243
    app_train_2['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    app_train_2['DAYS_EMPLOYED'].plot.hist(title="Histogramme de la durée d'emploi en jours")
    plt.xlabel("Durée d'emploi en jours")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Constat :
    - **Distribution corrigée :**
    1. La distribution de la durée d'emploi est maintenant conforme à ce que l'on attend, après avoir remplacé les valeurs anormales par NaN.

    2. **Nouvelle colonne **: Une colonne supplémentaire a été créée pour indiquer que certaines valeurs étaient initialement des anomalies.

    3. **Imputation des valeurs manquantes** : Les valeurs NaN seront probablement remplacées par la médiane de la colonne.

    4. **Autres colonnes avec "DAYS"** : Ces colonnes semblent conformes aux attentes, sans outliers évidents.

    5. **Données de test** : Toute **modification apportée aux données d'entraînement** doit également être appliquée aux données de test, y compris la création de la nouvelle colonne et le remplacement des anomalies par NaN.
    """
    )
    return


@app.cell
def _(app_test_2, np):
    app_test_2['DAYS_EMPLOYED_ANOM'] = app_test_2['DAYS_EMPLOYED'] == 365243
    app_test_2['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    print('Il y a %d anomalies dans les données de test sur un total de %d entrées' % (app_test_2['DAYS_EMPLOYED_ANOM'].sum(), len(app_test_2)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - **Corrélations**

    1. **Objectif** : **Identifier les relations entre les caractéristiques et la cible (TARGET) en calculant le coefficient de corrélation de Pearson**.

    2. **Méthode** : Utiliser la méthode .corr sur le DataFrame pour obtenir les corrélations entre chaque variable et la cible.

    3. Interprétation des valeurs absolues :

    - 0.00–0.19 : **Très faible**

    - 0.20–0.39 : **Faible**

    - 0.40–0.59 : **Modérée**

    - 0.60–0.79 : **Forte**

    - 0.80–1.00 : **Très forte**

    **Limitation** : Le coefficient **de corrélation ne représente pas toujours parfaitement la pertinence d'une caractéristique,** mais il donne une première indication des relations possibles dans les données.
    """
    )
    return


@app.cell
def _(app_train_2):
    correlations = app_train_2.corr()['TARGET'].sort_values()
    print('Corrélations positives les plus élevées :\n', correlations.tail(15))
    print('\nCorrélations négatives les plus élevées :\n', correlations.head(15))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    `Les variables **EXT_SOURCE_1, EXT_SOURCE_2, et EXT_SOURCE_3** représentent des scores normalisés provenant de sources de données externes.`

    1. Ces scores sont utilisés pour évaluer la solvabilité ou le risque d'un client à partir d'informations externes, comme des données de bureau de crédit ou d'autres bases de données financières.

    2. Ces scores sont souvent agrégés ou calculés à partir de plusieurs facteurs externes et fournissent une indication globale du risque.
    --------------------------------------------------------------------------------

    - **Corrélation significative** : La variable **DAYS_BIRTH** montre la corrélation **positive la plus forte** (hors TARGET, car une variable est toujours corrélée à 1 avec elle-même).

    - **Interprétation** : **DAYS_BIRTH** représente l'âge du client en jours négatifs. Une corrélation positive signifie que les clients plus âgés sont moins susceptibles de faire défaut sur leur prêt (TARGET == 0).

    - **Clarification** : En prenant la valeur absolue de **DAYS_BIRTH**, la corrélation devient négative, ce qui reflète mieux la relation inverse entre l'âge et le risque de défaut.
    """
    )
    return


@app.cell
def _(app_train_2):
    app_train_2['DAYS_BIRTH'] = abs(app_train_2['DAYS_BIRTH'])
    app_train_2['DAYS_BIRTH'].corr(app_train_2['TARGET'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - **Relation Négative** : Il existe une relation linéaire négative entre l'âge du client et la cible (TARGET), indiquant que les clients plus âgés ont tendance à rembourser leurs prêts à temps plus souvent.

    1. **Analyse de la Variable** : Commençons par examiner cette variable.

    2. **Histogramme de l'Âge** : Nous allons tracer un histogramme de l'âge, en utilisant l'axe des x en années pour rendre le graphique plus compréhensible.
    """
    )
    return


@app.cell
def _(app_train_2, plt):
    print("La distribution de l'âge ne révèle pas grand-chose d'autre que l'absence de valeurs aberrantes, mais un graphique KDE coloré par la cible montrera l'effet de l'âge sur le remboursement des prêts.")
    plt.style.use('fivethirtyeight')
    plt.hist(app_train_2['DAYS_BIRTH'] / 365, edgecolor='k', bins=25)
    plt.title('Âge du Client')
    plt.xlabel('Âge (années)')
    plt.ylabel('Nombre')
    return


@app.cell
def _(app_train_2, plt, sns):
    print("Distribution d'Âge : La courbe pour target == 1 (prêts non remboursés) est plus concentrée chez les jeunes.")
    print("Corrélation Faible : Bien que la corrélation soit faible (-0.07), l'âge est probablement utile pour les modèles de machine learning.")
    print("Analyse par Tranche d'Âge :Création de Bins : Diviser l'âge en tranches de 5 ans.Calcul de la Moyenne : Calculer le taux moyen de défaut de paiement par tranche d'âge.")
    plt.figure(figsize=(10, 8))
    sns.kdeplot(app_train_2.loc[app_train_2['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label='target == 0')
    sns.kdeplot(app_train_2.loc[app_train_2['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label='target == 1')
    plt.xlabel('Âge (années)')
    plt.ylabel('Densité')
    plt.title('Distribution des Âges')
    return


@app.cell
def _(app_train_2, np, pd):
    age_data = app_train_2[['TARGET', 'DAYS_BIRTH']]
    age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365
    age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins=np.linspace(20, 70, num=11))
    age_data.head(10)
    return (age_data,)


@app.cell
def _(age_data):
    # Grouper par tranche d'âge et calculer les moyennes
    age_groups = age_data.groupby('YEARS_BINNED').mean()
    age_groups
    return (age_groups,)


@app.cell
def _(age_groups, plt):
    # Définir la taille de la figure
    plt.figure(figsize=(8, 8))

    # Tracer les tranches d'âge et la moyenne de la cible sous forme de diagramme en barres
    plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

    # Étiqueter le graphique
    plt.xticks(rotation=75); plt.xlabel('Groupe d\'Âge (années)'); plt.ylabel('Taux de Défaut de Paiement (%)')
    plt.title('Taux de Défaut de Paiement par Groupe d\'Âge');
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. **Tendance Claire** : **Les jeunes demandeurs sont plus susceptibles de ne pas rembourser leur prêt**. Le taux de **défaut de paiement est supérieur à 10% pour les trois groupes d'âge les plus jeunes et inférieur à 5% pour le groupe d'âge le plus âgé**.

    2. **Implications pour la Banque** : **Les clients plus jeunes pourraient bénéficier de conseils ou de plans financier** pour les aider à rembourser à temps, sans pour autant discriminer contre eux.

    3. **Sources Externes** : Les variables EXT_SOURCE_1, EXT_SOURCE_2, et EXT_SOURCE_3 montrent les corrélations négatives les plus fortes avec la cible (TARGET). **Elles représentent un score normalisé provenant de sources de données externes, possiblement une évaluation de crédit cumulative.**

    4. **Analyse des Variables** : Nous éxaminom ces variables pour comprendre leur relation avec la cible et entre elles.
    """
    )
    return


@app.cell
def _(app_train_2):
    ext_data = app_train_2[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    ext_data_corrs = ext_data.corr()
    ext_data_corrs
    return ext_data, ext_data_corrs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. **Corrélations Négatives** : Les trois variables **EXT_SOURCE** sont **négativement corrélées avec la cible (TARGET)**, ce qui indique qu'une augmentation de leur valeur est associée à une probabilité plus élevée de remboursement du prêt.

    2. Corrélation **Positive** avec l'Âge : **DAYS_BIRTH est positivement corrélé avec EXT_SOURCE_1,** suggérant que l'âge pourrait être un facteur pris en compte dans ce score.

    3. **Visualisation** : Examiner la distribution de ces variables, colorées selon la valeur de la cible, pour observer leur effet sur le remboursement.
    """
    )
    return


@app.cell
def _(ext_data_corrs, plt, sns):
    # Définir la taille de la figure
    plt.figure(figsize=(8, 6))

    # Carte de chaleur des corrélations
    sns.heatmap(ext_data_corrs, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=True, vmax=0.6)

    # Ajouter un titre au graphique
    plt.title('Matrice de corrélation')
    return


@app.cell
def _(app_train_2, plt, sns):
    plt.figure(figsize=(10, 12))
    for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
        plt.subplot(3, 1, i + 1)
        sns.kdeplot(app_train_2.loc[app_train_2['TARGET'] == 0, source], label='target == 0')
        sns.kdeplot(app_train_2.loc[app_train_2['TARGET'] == 1, source], label='target == 1')
        plt.title('Distribution de %s selon la valeur de TARGET' % source)
        plt.xlabel('%s' % source)
        plt.ylabel('Densité')
    plt.tight_layout(h_pad=2.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. **EXT_SOURCE_3** : Cette variable 1. montre la plus grande différence entre les valeurs de la cible1. , **indiquant une relation avec la probabilité de remboursement d'un prêt**, bien que cette relation soit **faible**.

    2. **Utilité pour le Machine Learning** : Malgré **des corrélations faibles**, ces variables **restent utiles** pour **prédire si un demandeur remboursera son prêt à temps** (soit TARGET >1).

    3. **Pairs Plot** : Ce graphique permet d'explorer les relations entre **plusieurs variables (EXT_SOURCE et DAYS_BIRTH)** ainsi que leurs distributions. Il combine des scatterplots, des histogrammes et des courbes de densité 2D pour une analyse visuelle approfondie.
    """
    )
    return


@app.cell
def _(age_data, ext_data, np, plt, sns):
    # Copier les données pour le tracé
    # On retire la colonne DAYS_BIRTH et on copie les données dans un nouveau DataFrame
    plot_data = ext_data.drop(columns=['DAYS_BIRTH']).copy()

    # Ajouter l'âge du client en années
    plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']

    # Supprimer les valeurs manquantes et limiter à 100000 premières lignes
    plot_data = plot_data.dropna().loc[:100000, :]

    # Fonction pour calculer le coefficient de corrélation entre deux colonnes
    def corr_func(x, y, **kwargs):
        r = np.corrcoef(x, y)[0][1]  # Calcul du coefficient de corrélation
        ax = plt.gca()  # Obtenir l'axe actuel
        ax.annotate("r = {:.2f}".format(r),  # Annoter le graphique avec la valeur de r
                    xy=(.2, .8), xycoords=ax.transAxes,
                    size=20)

    # Créer un objet PairGrid pour visualiser les relations entre les variables
    grid = sns.PairGrid(data=plot_data, height=3, diag_sharey=False,
                        hue='TARGET',
                        vars=[x for x in list(plot_data.columns) if x != 'TARGET'])

    # Tracé supérieur : nuage de points (scatter plot)
    grid.map_upper(plt.scatter, alpha=0.2)

    # Diagonale : histogramme (distribution des variables)
    grid.map_diag(sns.kdeplot)

    # Tracé inférieur : courbes de densité 2D
    grid.map_lower(sns.kdeplot, cmap=plt.cm.OrRd_r)

    # Ajouter un titre au graphique global
    plt.suptitle('Ext Source and Age Features Pairs Plot', size=32, y=1.05)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. **Couleurs** :
    - **Rouge** pour les prêts **non remboursés ! **
    - **Bleu** pour les *prêts **remboursés**.

    2. **Relations** : On observe **des relations** entre **les variables EXT_SOURCE et DAYS_BIRTH (ou YEARS_BIRTH).**

    3. **Corrélation Positive** : **Une relation linéaire positive modérée **entre EXT_SOURCE_1** et l'âge, **suggérant que l'âge du client** pourrait être un facteur dans ce score**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. **Réalisation du prétraitement des données avant modélisation**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### 4.1 **Prétraitement et préparation des données avant l'entraînement**""")
    return


@app.cell
def _(app_test_2, app_train_2, pd):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.impute import SimpleImputer
    target = app_train_2['TARGET'] if 'TARGET' in app_train_2.columns else None
    train_ids = app_train_2['SK_ID_CURR']
    test_ids = app_test_2['SK_ID_CURR']
    features_train = app_train_2.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    features_test = app_test_2.drop(columns=['SK_ID_CURR'], errors='ignore')
    imputer = SimpleImputer(strategy='median')
    imputer.fit(features_train)
    train_imputed = pd.DataFrame(imputer.transform(features_train), columns=features_train.columns)
    test_imputed = pd.DataFrame(imputer.transform(features_test), columns=features_test.columns)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = pd.DataFrame(scaler.fit_transform(train_imputed), columns=train_imputed.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_imputed), columns=test_imputed.columns)
    train_scaled['SK_ID_CURR'] = train_ids.values
    test_scaled['SK_ID_CURR'] = test_ids.values
    if target is not None:
        train_scaled['TARGET'] = target.values
    print('Forme entraînement :', train_scaled.shape)
    print('Forme test :', test_scaled.shape)
    print('Colonnes entraînement :', train_scaled.columns.tolist())
    print('Colonnes de test :', test_scaled.columns.tolist())
    return (SimpleImputer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. **Modélisation _**sans feature engineering**_ + enregistrement des experiences dans MLFLow**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### 5.1 **Définition des métriques d'évaluation des performances du modèle & Caclul du _score métier_**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##### **Les métriques de performance d'évaluation des modèles**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. **auc_mean** : Moyenne des scores AUC Area Under the Curves ur les validations croisées.

    2. **auc_std** : Écart-type des scores AUC (stabilité du modèle).

    3. **auc_global** : Score AUC global sur toutes les prédictions.

    4. **business_score_mean** : Moyenne du score métier normalisé (coût évité) sur les folds.

    5. **usiness_score_global** : Score métier normalisé sur toutes les prédictions.

    6. **accuracy_mean** : Moyenne de la précision globale (toutes classes confondues).

    7. **accuracy_global** : Précision globale du modèle (sur tout l'ensemble).

    8. **precision_mean** : Moyenne de la précision positive (sur les folds).

    9. **precision_global** : Précision positive globale (proportion de bons positifs parmi les positifs prédits).

    10. **recall_mean** : Moyenne du rappel (sur les folds).

    11. **recall_global** : Rappel global (proportion de vrais positifs détectés).

    12. **f1_mean** : Moyenne du F1-score (équilibre précision/rappel).

    13. **f1_global** : F1-score global sur l’ensemble des prédictions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##### **Le score métier** avec exemple d'interprétation""")
    return


@app.cell
def _(np):
    # Importation de la bibliothèque NumPy pour le traitement numérique des tableaux

    # Définition de la fonction de score métier normalisé
    def calculate_normalized_business_score(y_true, y_proba, threshold=0.5, cost_fn=10, cost_fp=1):
        """
        Calcule un score métier pondéré et normalisé entre 0 et 1.

        Le score reflète le coût des erreurs (FN, FP) en fonction d’un seuil de décision.
        Il est ensuite converti en score entre 0 (pire) et 1 (meilleur) pour pouvoir être comparé
        à d'autres scores de modèles (comme l’AUC).

        Paramètres :
        - y_true : array-like, vraies classes (0 = bon client, 1 = défaut)
        - y_proba : array-like, probabilités prédites pour la classe 1 (risque)
        - threshold : float, seuil pour transformer les probabilités en classes
        - cost_fn : float, coût d’un faux négatif
        - cost_fp : float, coût d’un faux positif

        Retour :
        - float : Score métier normalisé entre 0 (pire) et 1 (meilleur)
        """

        # Étape 1 : transformation des probabilités en classes binaires selon le seuil
        # Si la probabilité prédite est supérieure ou égale au seuil, on classe en 1 (risque), sinon en 0 (bon client)
        y_pred = (y_proba >= threshold).astype(int)

        # Étape 2 : calcul du nombre de faux positifs (FP)
        # FP = cas où on prédit 1 (risque) alors que la vraie classe est 0 (client fiable)
        fp = np.sum((y_pred == 1) & (y_true == 0))

        # Calcul du nombre de faux négatifs (FN)
        # FN = cas où on prédit 0 (bon client) alors que la vraie classe est 1 (risque réel)
        fn = np.sum((y_pred == 0) & (y_true == 1))

        # Étape 3 : calcul du coût total basé sur les FP et FN pondérés
        # On multiplie le nombre de FP par leur coût, idem pour les FN, puis on additionne
        total_cost = fp * cost_fp + fn * cost_fn

        # Étape 4 : calcul du pire coût possible (pire scénario)
        # Nombre maximal de FP : si on classe tous les bons clients en risque
        worst_fp = np.sum(y_true == 0)

        # Nombre maximal de FN : si on classe tous les clients risqués en bon client
        worst_fn = np.sum(y_true == 1)

        # Coût maximal = ce qu'on aurait si toutes les prédictions étaient mauvaises
        worst_cost = worst_fp * cost_fp + worst_fn * cost_fn

        # Étape 5 : calcul du score normalisé entre 0 et 1
        # On inverse la proportion du coût total sur le coût maximal pour que 1 = parfait, 0 = pire
        score_normalized = 1 - (total_cost / worst_cost)

        # Retourne le score final normalisé, prêt à être utilisé pour comparer des modèles ou optimiser un seuil
        return score_normalized
    return (calculate_normalized_business_score,)


@app.cell
def _(calculate_normalized_business_score, np):
    # Définition d'une fonction pour trouver le seuil optimal qui maximise le score métier
    def find_optimal_threshold_business_score(y_true, y_proba, cost_fn=10, cost_fp=1):
        """
        Recherche le seuil optimal qui maximise le score métier normalisé.

        Paramètres :
        - y_true : array-like, les vraies étiquettes (0 ou 1)
        - y_proba : array-like, probabilités prédites pour la classe 1 (risque)
        - cost_fn : float, coût d’un faux négatif
        - cost_fp : float, coût d’un faux positif

        Retour :
        - best_threshold : seuil optimal (float)
        - best_score : score métier normalisé correspondant à ce seuil
        """

        # Création d'une liste de 100 seuils possibles entre 0.01 et 0.99 (grille de recherche)
        thresholds = np.linspace(0.01, 0.99, 100)

        # Initialisation du meilleur score à une valeur très basse (on cherche à maximiser)
        best_score = -1

        # Initialisation du meilleur seuil (par défaut à 0.5 au cas où aucun seuil ne serait meilleur)
        best_threshold = 0.5

        # Boucle sur tous les seuils possibles pour trouver celui qui maximise le score métier
        for threshold in thresholds:
            # Calcul du score métier normalisé pour le seuil courant
            score = calculate_normalized_business_score(y_true, y_proba, threshold, cost_fn, cost_fp)

            # Si le score est meilleur que le précédent, on le sauvegarde ainsi que le seuil associé
            if score > best_score:
                best_score = score              # Mise à jour du meilleur score
                best_threshold = threshold      # Mise à jour du meilleur seuil

        # Retourne le meilleur seuil trouvé et le score associé
        return best_threshold, best_score
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. **Définition du score métier** : c'est  une mesure qui quantifie **les coûts financiers** des erreurs de classification d'un modèle (**faux positifs et faux négatifs**), en tenant compte de leur impact asymétrique sur l'entreprise. Il permet d'optimiser les décisions en alignant les performances du modèle avec les objectifs économiques.
    ------------------------------------------------------------------------------
    -   A) **Faux Négatif** (FN) – **Client risqué** classé comme **fiable**
    Le modèle prédit que le client est "bon", mais en réalité il est "à risque".
    -  Crédit accordé à tort → Perte en capital directe pour l'entreprise.
    -  Très coûteux, car le client peut ne pas rembourser son emprunt.
    -  Coût attribué (exemple) : cost_fn = 10
    ------------------------------------------------------------------------------
    -  B) **Faux Positif** (FP) – Client fiable classé comme **risqué**
    Le modèle refuse un crédit à un client qui aurait été solvable.
    -  Manque à gagner → Opportunité commerciale perdue.
    -  Moins grave qu’un FN mais reste une erreur à éviter.
    -  Coût attribué (exemple) : cost_fp = 1
    ------------------------------------------------------------------------------
     Pondération des erreurs
    On considère que 1 FN = 10 FP en termes de coût.

    Cela permet de protéger le capital de l’entreprise tout en gardant une ouverture commerciale maîtrisée.

     **Exemple: une situation du score métier (normalisé)**

    1. 50 **faux positifs (FP)**
     Clients fiables refusés à tort : 50 × 1 = 50

    2. 20 **faux négatifs (FN)**
    Clients risqués acceptés par erreur : 20 × 10 = 200

    3. **Coût total des erreurs**
     total_cost = 50 + 200 = 250

    4. **Coût maximal possible (si toutes les prédictions étaient fausses)**
     On suppose **200 vrais bons clients** et **100 vrais clients à risque** :
    **worst_cost** = 200 × 1 + 100 × 10 = 200 + 1000 = 1200

    **Score métier normalisé**
     score = 1 - (250 / 1200) = 0.7917

    - Le score métier de **0.79** indique que le modèle évite **79 % des pertes financières maximales possibles**, ce qui représente un bon niveau de performance.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 5.2 **Entraînement du modèle de Régression Logistique sur les données de train et de test. Stockage centralisé du modèle assuré.**
    ⚠️ **Entraînement sur les données de base sans le feature Engineering**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Modèle de Régression Logistique : Utilisation de LogisticRegression de Scikit-Learn avec un paramètre de régularisation C réduit pour limiter le surapprentissage.

    - Syntaxe Scikit-Learn : Création du modèle, entraînement avec .fit, et prédiction des probabilités avec .predict_proba.

    - Maintenant que le modèle a été entraîné, nous pouvons l'utiliser pour faire des prédictions.
    -  Nous voulons prédire les probabilités de non-remboursement d'un prêt, donc nous utilisons la méthode predict_proba du modèle.
    -   Cela retourne un tableau de m x 2 où m est le nombre d'observations.
    -   La première colonne est la probabilité que la cible soit 0 et la deuxième colonne est la probabilité que la cible soit 1 (donc pour une seule ligne, les deux colonnes doivent s'additionner à 1).
    -    Nous voulons la probabilité que le prêt ne soit pas remboursé, donc nous sélectionnerons la deuxième colonne.


    Ici nous utilisons la **régression logistique** pour prédire **la probabilité qu'un client rembourse un crédit**.

    **Définition du modèle de régression logistique**
    La **régression logistique** est un modèle **statistique** utilisé pour résoudre des problèmes de classification ***binaire** (ou multiclasse avec des extensions). Elle prédit la probabilité qu'une observation appartienne à une classe donnée.

    La méthode **predict_proba** est utilisée pour obtenir les **probabilités** :

    **La première colonne correspond à :**
    - P(y=0∣X) (probabilité de non-remboursement).
    **La deuxième colonne correspond à :**
    - P(y=1∣X) (probabilité de remboursement).

    Cette approche est  adaptée au problème de scoring crédit, car elle permet d'évaluer le risque de remboursement et non remboursement associé à chaque client.


    --------------------------------------------------------------------------------
    - **Baseline Naïve** : Une prédiction naïve consisterait à attribuer une probabilité de 0.5 à toutes les observations du jeu de test, ce qui donnerait un score AUC ROC de 0.5 (équivalent à une prédiction aléatoire).

    - **Modèle Baseline** : Au lieu d'une approche naïve, on utilise la régression logistique comme baseline, un modèle légèrement plus sophistiqué.

    - **Prétraitement des Données** : Les données sont prétraitées en remplissant les valeurs manquantes (imputation) et en normalisant les caractéristiques (feature scaling) avant d'entraîner le modèle.

    - **Dans le Machine Learning** :
    Dans le machine learning, l'AUC fait référence à **l'Area Under the Receiver Operating Characteristic Curve (AUC-ROC)**. C'est une mesure utilisée pour évaluer la performance des modèles de classification binaire. Elle représente la capacité d'un modèle à distinguer correctement les classes positives et négatives.

    - **Interprétation** : L'AUC-ROC varie entre 0 et 1. Une valeur de 1 indique un modèle parfait, tandis qu'une valeur de 0,5 signifie que le modèle ne fait que des prédictions aléatoires. Une valeur supérieure à 0,5 indique que le modèle a une certaine capacité à prédire, mais une valeur supérieure à 0,8 est généralement considérée comme bonn.
    """
    )
    return


@app.cell
def _(SimpleImputer, app_test_2, app_train_2, pd):
    X_train = app_train_2.drop(columns=['SK_ID_CURR', 'TARGET'])
    y_train = app_train_2['TARGET']
    X_test = app_test_2.drop(columns=['SK_ID_CURR'])
    test_ids_1 = app_test_2['SK_ID_CURR']
    imputer_1 = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer_1.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer_1.transform(X_test), columns=X_test.columns)
    return X_test, X_train


@app.cell
def _(X_test, X_train):
    print("NaN dans X_train après imputation :", X_train.isna().sum().sum())  # Doit retourner 0
    print("NaN dans X_test après imputation :", X_test.isna().sum().sum())    # Doit retourner 0
    return


@app.cell
def _(
    MlflowClient_1,
    SimpleImputer_1,
    app_test_2,
    app_train_2,
    mlflow_1,
    np_1,
    pd_1,
    public_url_1,
):
    import os
    import time
    from pyngrok import ngrok
    import mlflow

    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.impute import SimpleImputer
    import joblib
    from mlflow.models.signature import infer_signature

    def calculate_normalized_business_score_1(y_true, y_proba, threshold=0.5, cost_fn=10, cost_fp=1):
        """
        Calcule un score métier pondéré et normalisé entre 0 et 1.
        Le score reflète le coût des erreurs (FN, FP) en fonction d’un seuil de décision.
        Il est ensuite converti en score entre 0 (pire) et 1 (meilleur).
        """
        y_pred = (y_proba >= threshold).astype(int)
        fp = np_1.sum((y_pred == 1) & (y_true == 0))
        fn = np_1.sum((y_pred == 0) & (y_true == 1))
        total_cost = fp * cost_fp + fn * cost_fn
        worst_fp = np_1.sum(y_true == 0)
        worst_fn = np_1.sum(y_true == 1)
        worst_cost = worst_fp * cost_fp + worst_fn * cost_fn
        score_normalized = 1 - total_cost / worst_cost
        return score_normalized
    client_1 = MlflowClient_1()
    experiment_name = 'Credit_Scoring_Tool_baseline'
    try:
        experiment = client_1.get_experiment_by_name(experiment_name)
        if experiment and experiment.lifecycle_stage == 'deleted':
            client_1.restore_experiment(experiment.experiment_id)
            print(f"Expérience '{experiment_name}' restaurée")
        elif not experiment:
            client_1.create_experiment(experiment_name)
            print(f"Nouvelle expérience '{experiment_name}' créée")
        mlflow_1.set_experiment(experiment_name)
    except Exception as e:
        print(f'Erreur configuration MLflow : {str(e)}')
        raise
    with mlflow_1.start_run(run_name='Logistic_Regression_Baseline') as run:
        mlflow_1.set_tags({'Business_Score': 'Normalisé: 1 - (FP*1 + FN*10) / Cout_maximal', 'Data_Version': '1.0', 'Model_Type': 'LogisticRegression', 'Feature_Engineering': 'Non appliqué'})
        mlflow_1.log_params({'C': 0.0001, 'solver': 'lbfgs', 'max_iter': 1000})
        X_train_1 = app_train_2.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
        y_train_1 = app_train_2['TARGET']
        X_test_1 = app_test_2.drop(columns=['SK_ID_CURR'], errors='ignore')
        test_ids_2 = app_test_2['SK_ID_CURR']
        assert 'TARGET' in app_train_2.columns, ' Vérifie la présence de la cible'
        assert 'SK_ID_CURR' in app_test_2.columns, "Vérifie la présence de l'ID client"
        imputer_2 = SimpleImputer_1(strategy='median')
        X_train_1 = pd_1.DataFrame(imputer_2.fit_transform(X_train_1), columns=X_train_1.columns)
        X_test_1 = pd_1.DataFrame(imputer_2.transform(X_test_1), columns=X_test_1.columns)
        assert X_train_1.isna().sum().sum() == 0
        assert X_test_1.isna().sum().sum() == 0
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        metrics = {'auc': [], 'business_score': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        y_true_all, y_proba_all = ([], [])
        for train_idx, val_idx in kf.split(X_train_1):
            X_fold_train, X_fold_val = (X_train_1.iloc[train_idx], X_train_1.iloc[val_idx])
            y_fold_train, y_fold_val = (y_train_1.iloc[train_idx], y_train_1.iloc[val_idx])
            model = LogisticRegression(C=0.0001, solver='lbfgs', max_iter=1000, random_state=42)
            model.fit(X_fold_train, y_fold_train)
            y_proba = model.predict_proba(X_fold_val)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            y_true_all.extend(y_fold_val)
            y_proba_all.extend(y_proba)
            metrics['auc'].append(roc_auc_score(y_fold_val, y_proba))
            metrics['business_score'].append(calculate_normalized_business_score_1(y_fold_val, y_proba))
            metrics['accuracy'].append(accuracy_score(y_fold_val, y_pred))
            metrics['precision'].append(precision_score(y_fold_val, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_fold_val, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_fold_val, y_pred, zero_division=0))
        y_true_all = np_1.array(y_true_all)
        y_proba_all = np_1.array(y_proba_all)
        y_pred_global = (y_proba_all >= 0.5).astype(int)
        auc_global = roc_auc_score(y_true_all, y_proba_all)
        business_global = calculate_normalized_business_score_1(y_true_all, y_proba_all)
        accuracy_global = accuracy_score(y_true_all, y_pred_global)
        precision_global = precision_score(y_true_all, y_pred_global, zero_division=0)
        recall_global = recall_score(y_true_all, y_pred_global, zero_division=0)
        f1_global = f1_score(y_true_all, y_pred_global, zero_division=0)
        print('\nRÉSULTATS VALIDATION CROISÉE')
        print(f"AUC moyen : {np_1.mean(metrics['auc']):.3f} ± {np_1.std(metrics['auc']):.3f}")
        print(f'AUC global : {auc_global:.3f}')
        print(f'Score métier global : {business_global:.3f}')
        print(f"Score métier moyen : {np_1.mean(metrics['business_score']):.3f}")
        print(f"Accuracy moyen : {np_1.mean(metrics['accuracy']):.3f}")
        print(f'Accuracy global : {accuracy_global:.3f}')
        print(f"Précision moyenne : {np_1.mean(metrics['precision']):.3f}")
        print(f'Précision globale : {precision_global:.3f}')
        print(f"Rappel moyen : {np_1.mean(metrics['recall']):.3f}")
        print(f'Rappel global : {recall_global:.3f}')
        print(f"F1-score moyen : {np_1.mean(metrics['f1']):.3f}")
        print(f'F1-score global : {f1_global:.3f}')
        mlflow_1.log_metrics({'auc_mean': float(np_1.mean(metrics['auc'])), 'auc_std': float(np_1.std(metrics['auc'])), 'auc_global': float(auc_global), 'business_score_global': float(business_global), 'accuracy_mean': float(np_1.mean(metrics['accuracy'])), 'accuracy_global': float(accuracy_global), 'precision_mean': float(np_1.mean(metrics['precision'])), 'precision_global': float(precision_global), 'recall_mean': float(np_1.mean(metrics['recall'])), 'recall_global': float(recall_global), 'f1_mean': float(np_1.mean(metrics['f1'])), 'f1_global': float(f1_global)})
        final_model = LogisticRegression(C=0.0001, solver='lbfgs', max_iter=1000, random_state=42)
        final_model.fit(X_train_1, y_train_1)
        joblib.dump(final_model, 'logistic_regression_baseline.pkl')
        mlflow_1.log_artifact('logistic_regression_baseline.pkl')
        signature = infer_signature(X_train_1, final_model.predict(X_train_1))
        mlflow_1.sklearn.log_model(sk_model=final_model, artifact_path='model', signature=signature, registered_model_name='Logistic_Regression_Baseline')
        test_proba = final_model.predict_proba(X_test_1)[:, 1]
        submit = pd_1.DataFrame({'SK_ID_CURR': test_ids_2, 'TARGET': test_proba})
        submit.to_csv('Logistic_Regression_baseline.csv', index=False)
        print('\n APERÇU DES PRÉDICTIONS ')
        print(submit.head())
    print('\n Entraînement terminé avec succès!')
    print(f' Accès MLflow : {public_url_1.public_url}')
    return LogisticRegression, SimpleImputer, mlflow, ngrok, np, os, pd, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. **AUC > 0.6** → Le modèle a un pouvoir prédictif qui est modéré.

    2. **Accuracy élevée mais rappel nul** → Le modèle prédit uniquement la classe majoritaire (déséquilibre des classes).

    3. **Score métier** = 0.532 → Le modèle évite un peu plus de 50% des pertes possibles.

    4. **F1-score** = 0.000 → Aucun défaut correctement prédit (pour l’instant).


    - Les résultats sont normaux pour un problème de prédiction de défaut de crédit:

    _Déséquilibre des classes : nous prédisons la probabilité "que le prêt ne sera pas remboursé". Dans les données de crédit, **la classe positive (défaut) est  très minoritaire,  moins de 10% des cas.**_

    5. Nous utiliserons SMOTE à l'avenir  pour gérer le déséquilibre des classes
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 5.3 **Entraînement du modèle de _Forêt Aléatoire_ sur les données de train et de test. Stockage centralisé du modèle assuré.**
     **Entraînement sur les données de base sans le feature Engineering**
    """
    )
    return


@app.cell
def _(
    KFold_1,
    MlflowClient_2,
    accuracy_score_1,
    app_test_2,
    app_train_2,
    f1_score_1,
    joblib_1,
    mlflow_2,
    np_2,
    pd_2,
    precision_score_1,
    public_url_1,
    recall_score_1,
    roc_auc_score_1,
):
    import os
    import time
    from pyngrok import ngrok
    import mlflow

    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import joblib
    from mlflow.models.signature import infer_signature

    def calculate_normalized_business_score_2(y_true, y_proba, threshold=0.5, cost_fn=10, cost_fp=1):
        """
        Calcule un score métier entre 0 (pire) et 1 (meilleur), pondéré par le coût des erreurs de classification.
        """
        y_pred = (y_proba >= threshold).astype(int)
        fp = np_2.sum((y_pred == 1) & (y_true == 0))
        fn = np_2.sum((y_pred == 0) & (y_true == 1))
        total_cost = fp * cost_fp + fn * cost_fn
        worst_fp = np_2.sum(y_true == 0)
        worst_fn = np_2.sum(y_true == 1)
        worst_cost = worst_fp * cost_fp + worst_fn * cost_fn
        return 1 - total_cost / worst_cost
    client_2 = MlflowClient_2()
    experiment_name_1 = 'Credit_Scoring_Tool_baseline'
    try:
        experiment_1 = client_2.get_experiment_by_name(experiment_name_1)
        if experiment_1 and experiment_1.lifecycle_stage == 'deleted':
            client_2.restore_experiment(experiment_1.experiment_id)
        elif not experiment_1:
            client_2.create_experiment(experiment_name_1)
        mlflow_2.set_experiment(experiment_name_1)
    except Exception as e:
        print(f'Erreur configuration MLflow : {str(e)}')
        raise
    with mlflow_2.start_run(run_name='Random_Forest_Baseline'):
        mlflow_2.set_tags({'Business_Score_Definition': 'Normalisé: 1 - (FP*1 + FN*10) / worst_cost', 'Feature_Engineering': 'Non appliqué - Données brutes', 'Model': 'RandomForestClassifier', 'Metrics': 'Accuracy, AUC, Precision, Recall, F1, Business Score'})
        mlflow_2.log_params({'n_estimators': 100, 'random_state': 42, 'n_jobs': -1})
        X_train_2 = app_train_2.drop(columns=['SK_ID_CURR', 'TARGET'])
        y_train_2 = app_train_2['TARGET']
        X_test_2 = app_test_2.drop(columns=['SK_ID_CURR'])
        test_ids_3 = app_test_2['SK_ID_CURR']
        kf_1 = KFold_1(n_splits=5, shuffle=True, random_state=42)
        metrics_1 = {'auc': [], 'business_score': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        y_true_all_1, y_proba_all_1 = ([], [])
        for train_idx_1, val_idx_1 in kf_1.split(X_train_2):
            X_fold_train_1, X_fold_val_1 = (X_train_2.iloc[train_idx_1], X_train_2.iloc[val_idx_1])
            y_fold_train_1, y_fold_val_1 = (y_train_2.iloc[train_idx_1], y_train_2.iloc[val_idx_1])
            model_1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model_1.fit(X_fold_train_1, y_fold_train_1)
            y_proba_1 = model_1.predict_proba(X_fold_val_1)[:, 1]
            y_pred_1 = (y_proba_1 >= 0.5).astype(int)
            y_true_all_1.extend(y_fold_val_1)
            y_proba_all_1.extend(y_proba_1)
            metrics_1['auc'].append(roc_auc_score_1(y_fold_val_1, y_proba_1))
            metrics_1['business_score'].append(calculate_normalized_business_score_2(y_fold_val_1, y_proba_1))
            metrics_1['accuracy'].append(accuracy_score_1(y_fold_val_1, y_pred_1))
            metrics_1['precision'].append(precision_score_1(y_fold_val_1, y_pred_1, zero_division=0))
            metrics_1['recall'].append(recall_score_1(y_fold_val_1, y_pred_1, zero_division=0))
            metrics_1['f1'].append(f1_score_1(y_fold_val_1, y_pred_1, zero_division=0))
        y_true_all_1 = np_2.array(y_true_all_1)
        y_proba_all_1 = np_2.array(y_proba_all_1)
        y_pred_global_1 = (y_proba_all_1 >= 0.5).astype(int)
        auc_global_1 = roc_auc_score_1(y_true_all_1, y_proba_all_1)
        business_global_1 = calculate_normalized_business_score_2(y_true_all_1, y_proba_all_1)
        accuracy_global_1 = accuracy_score_1(y_true_all_1, y_pred_global_1)
        precision_global_1 = precision_score_1(y_true_all_1, y_pred_global_1, zero_division=0)
        recall_global_1 = recall_score_1(y_true_all_1, y_pred_global_1, zero_division=0)
        f1_global_1 = f1_score_1(y_true_all_1, y_pred_global_1, zero_division=0)
        print('\nRÉSULTATS VALIDATION CROISÉE')
        print(f"AUC moyen : {np_2.mean(metrics_1['auc']):.3f} ± {np_2.std(metrics_1['auc']):.3f}")
        print(f'AUC global : {auc_global_1:.3f}')
        print(f"Score métier moyen : {np_2.mean(metrics_1['business_score']):.3f}")
        print(f'Score métier global : {business_global_1:.3f}')
        print(f"Accuracy moyen : {np_2.mean(metrics_1['accuracy']):.3f}")
        print(f'Accuracy global : {accuracy_global_1:.3f}')
        print(f"Précision moyenne : {np_2.mean(metrics_1['precision']):.3f}")
        print(f'Précision globale : {precision_global_1:.3f}')
        print(f"Rappel moyen : {np_2.mean(metrics_1['recall']):.3f}")
        print(f'Rappel global : {recall_global_1:.3f}')
        print(f"F1-score moyen : {np_2.mean(metrics_1['f1']):.3f}")
        print(f'F1-score global : {f1_global_1:.3f}')
        mlflow_2.log_metrics({'auc_mean': float(np_2.mean(metrics_1['auc'])), 'auc_std': float(np_2.std(metrics_1['auc'])), 'auc_global': float(auc_global_1), 'business_score_mean': float(np_2.mean(metrics_1['business_score'])), 'business_score_global': float(business_global_1), 'accuracy_mean': float(np_2.mean(metrics_1['accuracy'])), 'accuracy_global': float(accuracy_global_1), 'precision_mean': float(np_2.mean(metrics_1['precision'])), 'precision_global': float(precision_global_1), 'recall_mean': float(np_2.mean(metrics_1['recall'])), 'recall_global': float(recall_global_1), 'f1_mean': float(np_2.mean(metrics_1['f1'])), 'f1_global': float(f1_global_1)})
        final_model_1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        final_model_1.fit(X_train_2, y_train_2)
        joblib_1.dump(final_model_1, 'Random_Forest_baseline.pkl')
        mlflow_2.log_artifact('Random_Forest_baseline.pkl')
        test_proba_1 = final_model_1.predict_proba(X_test_2)[:, 1]
        submit_1 = pd_2.DataFrame({'SK_ID_CURR': test_ids_3, 'TARGET': test_proba_1})
        submit_1.to_csv('Random_Forest_baseline.csv', index=False)
    print('\n Entraînement terminé avec succès!')
    print(f' Accès MLflow : {public_url_1.public_url}')
    return mlflow, ngrok, np, os, pd, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - **AUC** : 0.712 → Bon pouvoir de discrimination.

    1.  **Score métier** : 0.533 → Le modèle évite ~53% des pertes maximales possibles.
    2.  **Accuracy** : 0.919 → Trompeuse à cause du déséquilibre des classes.

    3. **Precision** : 0.667 → Prédictions positives assez fiables.

    4. **Recall** : 0.001 → Très faible, le modèle détecte peu les défauts.

    5. **F1-score** : 0.002 → Déséquilibre sévère : à améliorer.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 5.4 **Entraînement du modèle de _XG Boost_ sur les données de train et de test. Stockage centralisé du modèle assuré.**
     **Entraînement sur les données de base sans le feature Engineering**
    """
    )
    return


app._unparsable_cell(
    r"""
    !pip install xgboost
    """,
    name="_"
)


@app.cell
def _(
    KFold_2,
    MlflowClient_3,
    SimpleImputer_2,
    accuracy_score_2,
    app_test_2,
    app_train_2,
    f1_score_2,
    infer_signature_2,
    joblib_2,
    mlflow_3,
    np_3,
    pd_3,
    precision_score_2,
    public_url_1,
    recall_score_2,
    roc_auc_score_2,
):
    import os
    import time
    from pyngrok import ngrok
    import mlflow

    import pandas as pd
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.impute import SimpleImputer
    import joblib
    from mlflow.models.signature import infer_signature

    def calculate_normalized_business_score_3(y_true, y_proba, threshold=0.5, cost_fn=10, cost_fp=1):
        """
        Calcule un score métier entre 0 (pire) et 1 (meilleur), en fonction du coût des erreurs (FP/FN).
        """
        y_pred = (y_proba >= threshold).astype(int)
        fp = np_3.sum((y_pred == 1) & (y_true == 0))
        fn = np_3.sum((y_pred == 0) & (y_true == 1))
        total_cost = fp * cost_fp + fn * cost_fn
        worst_fp = np_3.sum(y_true == 0)
        worst_fn = np_3.sum(y_true == 1)
        worst_cost = worst_fp * cost_fp + worst_fn * cost_fn
        return 1 - total_cost / worst_cost
    client_3 = MlflowClient_3()
    experiment_name_2 = 'Credit_Scoring_Tool_baseline'
    try:
        experiment_2 = client_3.get_experiment_by_name(experiment_name_2)
        if experiment_2 and experiment_2.lifecycle_stage == 'deleted':
            client_3.restore_experiment(experiment_2.experiment_id)
        elif not experiment_2:
            client_3.create_experiment(experiment_name_2)
        mlflow_3.set_experiment(experiment_name_2)
    except Exception as e:
        print(f'Erreur configuration MLflow : {str(e)}')
        raise
    with mlflow_3.start_run(run_name='XGBoost_Baseline'):
        mlflow_3.set_tags({'Business_Score': 'Normalisé: 1 - (FP*1 + FN*10) / coût_max', 'Data_Version': '1.0', 'Model_Type': 'XGBoost', 'Feature_Engineering': 'Non appliqué'})
        mlflow_3.log_params({'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42})
        X_train_3 = app_train_2.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
        y_train_3 = app_train_2['TARGET']
        X_test_3 = app_test_2.drop(columns=['SK_ID_CURR'], errors='ignore')
        test_ids_4 = app_test_2['SK_ID_CURR']
        imputer_3 = SimpleImputer_2(strategy='median')
        X_train_3 = pd_3.DataFrame(imputer_3.fit_transform(X_train_3), columns=X_train_3.columns)
        X_test_3 = pd_3.DataFrame(imputer_3.transform(X_test_3), columns=X_test_3.columns)
        kf_2 = KFold_2(n_splits=5, shuffle=True, random_state=42)
        metrics_2 = {'auc': [], 'business_score': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        y_true_all_2, y_proba_all_2 = ([], [])
        for train_idx_2, val_idx_2 in kf_2.split(X_train_3):
            X_fold_train_2, X_fold_val_2 = (X_train_3.iloc[train_idx_2], X_train_3.iloc[val_idx_2])
            y_fold_train_2, y_fold_val_2 = (y_train_3.iloc[train_idx_2], y_train_3.iloc[val_idx_2])
            model_2 = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')
            model_2.fit(X_fold_train_2, y_fold_train_2)
            y_proba_2 = model_2.predict_proba(X_fold_val_2)[:, 1]
            y_pred_2 = (y_proba_2 >= 0.5).astype(int)
            y_true_all_2.extend(y_fold_val_2)
            y_proba_all_2.extend(y_proba_2)
            metrics_2['auc'].append(roc_auc_score_2(y_fold_val_2, y_proba_2))
            metrics_2['business_score'].append(calculate_normalized_business_score_3(y_fold_val_2, y_proba_2))
            metrics_2['accuracy'].append(accuracy_score_2(y_fold_val_2, y_pred_2))
            metrics_2['precision'].append(precision_score_2(y_fold_val_2, y_pred_2, zero_division=0))
            metrics_2['recall'].append(recall_score_2(y_fold_val_2, y_pred_2, zero_division=0))
            metrics_2['f1'].append(f1_score_2(y_fold_val_2, y_pred_2, zero_division=0))
        y_true_all_2 = np_3.array(y_true_all_2)
        y_proba_all_2 = np_3.array(y_proba_all_2)
        y_pred_global_2 = (y_proba_all_2 >= 0.5).astype(int)
        auc_global_2 = roc_auc_score_2(y_true_all_2, y_proba_all_2)
        business_global_2 = calculate_normalized_business_score_3(y_true_all_2, y_proba_all_2)
        accuracy_global_2 = accuracy_score_2(y_true_all_2, y_pred_global_2)
        precision_global_2 = precision_score_2(y_true_all_2, y_pred_global_2, zero_division=0)
        recall_global_2 = recall_score_2(y_true_all_2, y_pred_global_2, zero_division=0)
        f1_global_2 = f1_score_2(y_true_all_2, y_pred_global_2, zero_division=0)
        print('\n=== RÉSULTATS VALIDATION CROISÉE ===')
        print(f"AUC moyen : {np_3.mean(metrics_2['auc']):.3f} ± {np_3.std(metrics_2['auc']):.3f}")
        print(f'AUC global : {auc_global_2:.3f}')
        print(f"Score métier moyen : {np_3.mean(metrics_2['business_score']):.3f}")
        print(f'Score métier global : {business_global_2:.3f}')
        print(f"Accuracy moyen : {np_3.mean(metrics_2['accuracy']):.3f}")
        print(f'Accuracy global : {accuracy_global_2:.3f}')
        print(f"Précision moyenne : {np_3.mean(metrics_2['precision']):.3f}")
        print(f'Précision globale : {precision_global_2:.3f}')
        print(f"Rappel moyen : {np_3.mean(metrics_2['recall']):.3f}")
        print(f'Rappel global : {recall_global_2:.3f}')
        print(f"F1-score moyen : {np_3.mean(metrics_2['f1']):.3f}")
        print(f'F1-score global : {f1_global_2:.3f}')
        mlflow_3.log_metrics({'auc_mean': float(np_3.mean(metrics_2['auc'])), 'auc_std': float(np_3.std(metrics_2['auc'])), 'auc_global': float(auc_global_2), 'business_score_mean': float(np_3.mean(metrics_2['business_score'])), 'business_score_global': float(business_global_2), 'accuracy_mean': float(np_3.mean(metrics_2['accuracy'])), 'accuracy_global': float(accuracy_global_2), 'precision_mean': float(np_3.mean(metrics_2['precision'])), 'precision_global': float(precision_global_2), 'recall_mean': float(np_3.mean(metrics_2['recall'])), 'recall_global': float(recall_global_2), 'f1_mean': float(np_3.mean(metrics_2['f1'])), 'f1_global': float(f1_global_2)})
        final_model_2 = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')
        final_model_2.fit(X_train_3, y_train_3)
        joblib_2.dump(final_model_2, 'XGBoost_baseline.pkl')
        mlflow_3.log_artifact('XGBoost_baseline.pkl')
        test_proba_2 = final_model_2.predict_proba(X_test_3)[:, 1]
        submit_2 = pd_3.DataFrame({'SK_ID_CURR': test_ids_4, 'TARGET': test_proba_2})
        submit_2.to_csv('XGBoost_baseline.csv', index=False)
        signature_1 = infer_signature_2(X_train_3, final_model_2.predict_proba(X_train_3)[:, 1])
        mlflow_3.xgboost.log_model(xgb_model=final_model_2, artifact_path='model', signature=signature_1, registered_model_name='XGBoost_Baseline')
    print('\n Entraînement terminé avec succès!')
    print(f' Accès MLflow : {public_url_1.public_url}')
    return SimpleImputer, client_3, mlflow, ngrok, np, os, pd, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    -  **AUC** : 0.751 → Bon pouvoir de discrimination du modèle.

    -  **Score métier global** : 0.53 → Score métier le plus élevé (0.536) → il minimise le coût financier.

    -  **Accuracy** : 0.920 → Bonne précision globale (dominée par la classe majoritaire).

    - **Précision** : 0.627 → Les prédictions positives sont assez fiables.

    -  **Rappel** : 0.008 → Le modèle détecte très peu de défauts.

    -  **F1-score** : 0.016 → Faible équilibre entre précision et rappel.

    ----------------------------------------------------------------------------
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### 5.5 **RESULTAT DES MODELISATIONS SANS LE FEATURE ENGINEERING**""")
    return


@app.cell
def _(pd_3, plt):
    data = {'Modèle': ['Régression Logistique', 'Random Forest', 'XGBoost'], 'AUC moyen': [0.634, 0.712, 0.751], 'AUC global': [0.634, 0.712, 0.751], 'Accuracy': [0.919, 0.92, 0.92], 'Précision': [0.0, 0.667, 0.627], 'Rappel': [0.0, 0.001, 0.008], 'F1-score': [0.0, 0.002, 0.016], 'Score métier': [0.532, 0.533, 0.536]}
    df = pd_3.DataFrame(data).set_index('Modèle')
    colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1']
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    df.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=8, padding=3)
    ax.set_title('Comparaison des performances des modèles (sans feature engineering)', fontsize=14)
    ax.set_ylabel('Valeur normalisée', fontsize=12)
    ax.set_xlabel('Modèles', fontsize=12)
    ax.legend(title='Métriques', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ** XGBoost est le meilleur modèle globalement :**

    - Meilleur AUC (0.751) → il classe le mieux les clients risqués.

    - Score métier le plus élevé (0.536) → il minimise le coût financier.

    - F1-score le plus élevé (0.016) → meilleure détection (modeste) de la classe minoritaire.

     Bien que les scores absolus restent faibles (déséquilibre des classes), le modèle XGBoost surpasse RF et la Régression Logistique sur l’ensemble des critères.
    """
    )
    return


@app.cell
def _(pd_3, plt):
    definitions = {'Métrique': ['AUC moyen', 'AUC global', 'Accuracy', 'Précision', 'Rappel', 'F1-score', 'Score métier'], 'Définition': ['MoyenneAUC sur K folds (validation croisée) (capacité à classer correctement).', 'AUC calculé sur l’ensemble des prédictions du modèle (performance globale).', 'Proportion d’exemples bien classés toutes classes confondues.', 'Part des prédictions positives qui sont réellement positives (qualité du modèle #class 1).', 'Part des vrais positifs détectés parmi tous les cas positifs (sensibilité).', 'Moyenne harmonique entre précision et rappel (équilibre détection/fiabilité).', 'Score métier normalisé entre 0 (coût max) et 1 (aucun coût) basé sur FP/FN pondérés.']}
    df_def = pd_3.DataFrame(definitions)
    fig_1, ax_1 = plt.subplots(figsize=(13, 4))
    fig_1.patch.set_facecolor('white')
    ax_1.set_facecolor('white')
    ax_1.axis('off')
    table = ax_1.table(cellText=df_def.values, colLabels=df_def.columns, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.7)
    plt.title('📘 Définitions des métriques utilisées', fontsize=13, weight='bold', pad=12)
    plt.show()
    return


@app.cell
def _(os_4):
    from PIL import Image, ImageDraw, ImageFont
    icons = {'régression_logistique': 'RL', 'random_forest': 'RF', 'xgboost': 'XGB'}
    img_size = (150, 150)
    font = ImageFont.load_default()
    output_dir = '/mnt/data/icons'
    os_4.makedirs(output_dir, exist_ok=True)
    for name, label in icons.items():
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        w, h = draw.textbbox((0, 0), label, font=font)[2:]
        position = ((img_size[0] - w) // 2, (img_size[1] - h) // 2)
        draw.text(position, label, fill='black', font=font)
        img.save(f'{output_dir}/{name}.png')
    os_4.listdir(output_dir)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 6. Feature Engineering création des caractéristiques **polynomiales** et **basées sur le domaine**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 6.1 Définition et importance des **caractéristiques polynomiales**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. **Importance de l'Ingénierie des Caractéristiques** : Les compétitions de machine learning sont souvent gagnées par ceux qui peuvent créer les caractéristiques les plus pertinentes à partir des données, car l'ingénierie des caractéristiques offre un meilleur retour sur investissement que la construction de modèles ou le réglage des hyperparamètres.

    2. **Construction et Sélection des Caractéristiques** : L'ingénierie des caractéristiques inclut **la construction de nouvelles caractéristiques** à partir des données existantes et la sélection des caractéristiques les plus importantes pour réduire la dimensionnalité.


    3. Nous utiliserons deux méthodes simples : **les caractéristiques polynomiales**, qui créent des puissances et des termes d'interaction entre les variables
    --------------------------------------------------------------------------------

    - **Définition et Utilité des Features Polynomiales**
    Les caractéristiques polynomiales permettent de capturer des relations non linéaires dans les données en créant de nouvelles variables à partir des variables existantes. Cela inclut:

    - Élévation à une puissance (x², x³, etc.) qui permet de mieux représenter les effets exponentiels

    - Termes d'interaction (x₁×x₂) qui capturent les effets combinés de plusieurs variables

    - Cette technique est particulièrement utile pour les modèles linéaires (comme la régression logistique) qui ne peuvent pas naturellement capturer les relations complexes. En créant ces nouvelles features, nous "préparons" les données pour que même un modèle simple puisse identifier des patterns complexes.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 6.1.2  Création des caractéristiques **polynomiales**""")
    return


@app.cell
def _(SimpleImputer_2, app_test_2, app_train_2):
    from sklearn.preprocessing import PolynomialFeatures
    poly_features = app_train_2[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
    poly_features_test = app_test_2[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    poly_target = poly_features['TARGET']
    poly_features = poly_features.drop(columns=['TARGET'])
    imputer_4 = SimpleImputer_2(strategy='median')
    poly_features = imputer_4.fit_transform(poly_features)
    poly_features_test = imputer_4.transform(poly_features_test)
    poly_transformer = PolynomialFeatures(degree=3)
    return poly_features, poly_features_test, poly_target, poly_transformer


@app.cell
def _(poly_features, poly_features_test, poly_transformer):
    poly_transformer.fit(poly_features)
    poly_features_1 = poly_transformer.transform(poly_features)
    poly_features_test_1 = poly_transformer.transform(poly_features_test)
    print('Forme des caractéristiques polynomiales : ', poly_features_1.shape)
    return poly_features_1, poly_features_test_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Création et Nommage des Caractéristiques Polynomiales""")
    return


@app.cell
def _(poly_transformer):
    # Obtenir les noms des 15 premières caractéristiques polynomiales créées
    # On utilise la méthode get_feature_names_out avec les noms des caractéristiques d'entrée
    poly_transformer.get_feature_names_out(input_features=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Il y a 35 caractéristiques avec des puissances individuelles jusqu'au degré 3 et des termes d'interaction. Maintenant, nous pouvons voir si l'une de ces nouvelles caractéristiques est corrélée avec la cible""")
    return


@app.cell
def _(pd_3, poly_features_1, poly_target, poly_transformer):
    poly_features_2 = pd_3.DataFrame(poly_features_1, columns=poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))
    poly_features_2['TARGET'] = poly_target
    poly_corrs = poly_features_2.corr()['TARGET'].sort_values()
    print(poly_corrs.head(10))
    print(poly_corrs.tail(5))
    return (poly_features_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Corrélations Plus Fortes : Plusieurs des nouvelles variables polynomiales montrent des corrélations plus fortes avec la cible que les caractéristiques originales.

    - Évaluation des Modèles : Nous ajouterons ces caractéristiques aux données d'entraînement et de test pour évaluer les modèles avec et sans ces nouvelles variables.

    - Approche Empirique : En machine learning, la seule façon de vérifier l'efficacité d'une méthode est souvent de l'essayer.
    """
    )
    return


@app.cell
def _(
    app_test_2,
    app_train_2,
    pd_3,
    poly_features_2,
    poly_features_test_1,
    poly_transformer,
):
    poly_features_test_2 = pd_3.DataFrame(poly_features_test_1, columns=poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))
    poly_features_2['SK_ID_CURR'] = app_train_2['SK_ID_CURR']
    app_train_poly = app_train_2.merge(poly_features_2, on='SK_ID_CURR', how='left')
    poly_features_test_2['SK_ID_CURR'] = app_test_2['SK_ID_CURR']
    app_test_poly = app_test_2.merge(poly_features_test_2, on='SK_ID_CURR', how='left')
    app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis=1)
    print("Forme des données d'entraînement avec les caractéristiques polynomiales : ", app_train_poly.shape)
    print('Forme des données de test avec les caractéristiques polynomiales :  ', app_test_poly.shape)
    return app_test_poly, app_train_poly


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Caractéristiques Polynomiales Créées**

    | Feature                          | Définition                                                                                 | Utilité                                                                                  |
    |----------------------------------|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
    | 1                                | Terme constant                                                                             | Représente l'ordonnée à l'origine dans les modèles                                      |
    | EXT_SOURCE_1                     | Score externe provenant d'une source normalisée 1                                          | Indicateur de fiabilité financière externe                                              |
    | EXT_SOURCE_2                     | Score externe provenant d'une source normalisée 2                                          | Deuxième évaluation financière externe                                                  |
    | EXT_SOURCE_3                     | Score externe provenant d'une source normalisée 3                                          | Troisième évaluation financière externe                                                 |
    | DAYS_BIRTH                       | Âge du client en jours (valeur négative)                                                  | Indicateur démographique important                                                     |
    | EXT_SOURCE_1²                    | Carré de EXT_SOURCE_1                                                                     | Accentue l'effet des scores extrêmes                                                   |
    | EXT_SOURCE_1 × EXT_SOURCE_2      | Interaction entre sources 1 et 2                                                          | Capture la concordance/discordance entre évaluations                                    |
    | EXT_SOURCE_1 × EXT_SOURCE_3      | Interaction entre sources 1 et 3                                                          | Effet combiné de différentes évaluations                                               |
    | EXT_SOURCE_1 × DAYS_BIRTH        | Interaction entre source 1 et âge                                                         | Variation de la fiabilité financière selon l'âge                                       |
    | EXT_SOURCE_2²                    | Carré de EXT_SOURCE_2                                                                     | Amplifie l'importance des valeurs extrêmes                                             |
    | EXT_SOURCE_2 × EXT_SOURCE_3      | Interaction entre sources 2 et 3                                                          | Concordance entre différentes évaluations                                              |
    | EXT_SOURCE_2 × DAYS_BIRTH        | Interaction entre source 2 et âge                                                         | Impact combiné de l'âge et du score financier                                          |
    | EXT_SOURCE_3²                    | Carré de EXT_SOURCE_3                                                                     | Amplifie les valeurs élevées/basses                                                    |
    | EXT_SOURCE_3 × DAYS_BIRTH        | Interaction entre source 3 et âge                                                         | Relation entre l'âge et cette évaluation                                               |
    | DAYS_BIRTH²                      | Carré de l'âge                                                                            | Capture les effets non linéaires de l'âge                                              |

    ---

    **Utilité des Caractéristiques Polynomiales**

    1. **Effets Non Linéaires** :
       - Les termes au carré (ex: `EXT_SOURCE_1²`, `DAYS_BIRTH²`) permettent de capturer des relations non linéaires qui ne peuvent pas être modélisées par des termes simples.

    2. **Effets Combinés** :
       - Les interactions (ex: `EXT_SOURCE_1 × DAYS_BIRTH`) permettent d'explorer la relation combinée entre deux variables, ce qui peut révéler des patterns complexes.

    3. **Amélioration des Modèles Simples** :
       - Ces caractéristiques enrichissent les données, permettant à des modèles simples comme la régression logistique ou le Random Forest de mieux capturer la complexité des données.

    4. **Préparation pour les Modèles Avancés** :
       - Ces caractéristiques sont également utiles pour améliorer les performances des modèles avancés comme XGBoost ou LightGBM.

    ---

    **Remarque**
    Ces caractéristiques polynomiales ont été générées à partir des variables `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` et `DAYS_BIRTH` avec un degré polynomial de 3. Elles enrichissent considérablement le jeu de données pour capturer des relations complexes.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 6.1.3  Définition et importance des **caractéristiques basées sur le domaine**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Définition** :
    La création **de caractéristiques basées sur le domaine consiste à transformer les données brutes** en variables pertinentes et interprétables, en s'appuyant sur des connaissances spécifiques du domaine d'application.

    **Importance** :
    Une bonne création de caractéristiques **améliore la performance des modèles** de machine learning en leur fournissant des informations plus riches et adaptées au problème à résoudre.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 6.1.4  Création des caractéristiques **basée sur la connaissance du domaine**""")
    return


@app.cell
def _(app_test_2, app_train_2):
    app_train_domain = app_train_2.copy()
    app_test_domain = app_test_2.copy()
    app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
    app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
    app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
    app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']
    return app_test_domain, app_train_domain


@app.cell
def _(app_test_domain):
    # Le DataFrame app_test_domain est défini avant d'exécuter ce code
    # Calculer les nouvelles caractéristiques basées sur la connaissance du domaine
    app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
    app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
    app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
    app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    --------**Caractéristiques Financières Créées**-------------

    | Feature                     | Définition                                                                                  | Utilité                                                                                  |
    |-----------------------------|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
    | CREDIT_INCOME_PERCENT       | Ratio entre le montant du crédit accordé et le revenu total du client                       | Évalue la capacité d'endettement (plus le ratio est élevé, plus le risque de défaut augmente) |
    | ANNUITY_INCOME_PERCENT      | Ratio entre l'annuité mensuelle du prêt et le revenu mensuel du client                      | Mesure l'effort de remboursement mensuel (% du revenu consacré au prêt)                 |
    | CREDIT_TERM                 | Durée totale du prêt en mois                                                               | Impacte le coût total du crédit et la solvabilité à long terme                          |
    | DAYS_EMPLOYED_PERCENT       | Ratio entre l'ancienneté professionnelle (en jours) et l'âge du client (en jours)           | Indique la stabilité professionnelle (ratio élevé = emploi stable et continu)           |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Visualisation des Nouvelles Variables
    Pour explorer visuellement les nouvelles variables basées sur la connaissance du domaine, nous allons créer des graphiques KDE (Kernel Density Estimation) colorés selon la valeur de la cible (TARGET). Voici comment nous pourrions procéder :
    """
    )
    return


@app.cell
def _(app_train_domain, plt, sns):
    plt.figure(figsize=(10, 12))
    variables = ['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']
    for i_1, var in enumerate(variables):
        plt.subplot(4, 1, i_1 + 1)
        sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, var], label='target == 0', color='blue')
        sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, var], label='target == 1', color='red')
        plt.title(f'Distribution of {var} by Target Value')
        plt.xlabel(var)
        plt.ylabel('Density')
        plt.legend()
    plt.tight_layout(h_pad=2.5)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. **CREDIT_INCOME_PERCENT** : Le pourcentage du montant du crédit par rapport au revenu du client. Cette caractéristique peut indiquer la capacité de remboursement du client en fonction de son revenu.

    2. **ANNUITY_INCOME_PERCENT** : Le pourcentage de la rente du prêt par rapport au revenu du client. Cela montre combien de son revenu le client doit consacrer au remboursement mensuel du prêt.

    3. **CREDIT_TERM** : La durée du prêt en mois. Cette information est cruciale car une durée plus longue peut signifier des paiements mensuels plus faibles mais un coût total du prêt plus élevé.

    4. **DAYS_EMPLOYED_PERCENT** : Le pourcentage des jours d'emploi par rapport à l'âge du client. Cela peut refléter la stabilité de l'emploi du client, un facteur important pour la capacité de remboursement.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""7. Modélisation **avec feature engineering** + enregistrement des experiences dans MLFLow""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 7.1 Entraînement sur les caractéristiques **polynomiales**
    - Regresession Logistique
    - Forêt aléatoire
    - XgBoost
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##### 7.1.1 Entraînement du modèle de **Régression Logistique** sur les données de train et de test. Stockage centralisé du modèle assuré.
      **Entraînement sur les données de base **avec le feature Engineering _caractéristiques polynomiales_**
    """
    )
    return


@app.cell
def _(app_train_poly):
    # On affiche les colonnes disponibles dans le dataframe d'entraînement avec les catactéistiques polynomiales
    print("Colonnes disponibles dans app_train_poly :", app_train_poly.columns.tolist())
    return


@app.cell
def _(app_train_poly):
    print("Nombre de NaN dans le dataset d'entraînement :")
    print(app_train_poly.isna().sum().sort_values(ascending=False))
    return


@app.cell
def _(
    KFold_3,
    LogisticRegression,
    SimpleImputer_3,
    accuracy_score_3,
    app_test_2,
    app_test_poly,
    app_train_poly,
    client_3,
    f1_score_3,
    infer_signature_3,
    joblib_3,
    mlflow_4,
    np_4,
    pd_3,
    poly_target,
    precision_score_3,
    public_url_1,
    recall_score_3,
    roc_auc_score_3,
):
    import os
    import time
    from pyngrok import ngrok
    import mlflow

    import numpy as np
    from mlflow.models.signature import infer_signature
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.impute import SimpleImputer
    import joblib

    def calculate_normalized_business_score_4(y_true, y_proba, threshold=0.5, cost_fn=10, cost_fp=1):
        """Score entre 0 (pire) et 1 (parfait), basé sur les FP/FN pondérés."""
        y_pred = (y_proba >= threshold).astype(int)
        fp = np_4.sum((y_pred == 1) & (y_true == 0))
        fn = np_4.sum((y_pred == 0) & (y_true == 1))
        total_cost = fp * cost_fp + fn * cost_fn
        worst_fp = np_4.sum(y_true == 0)
        worst_fn = np_4.sum(y_true == 1)
        worst_cost = worst_fp * cost_fp + worst_fn * cost_fn
        return 1 - total_cost / worst_cost
    experiment_name_3 = 'Credit_Scoring_Tool_FeatureEng'
    try:
        experiment_3 = client_3.get_experiment_by_name(experiment_name_3) or client_3.create_experiment(experiment_name_3)
        mlflow_4.set_experiment(experiment_name_3)
    except Exception as e:
        print(f'Erreur configuration expérience: {str(e)}')
        raise
    with mlflow_4.start_run(run_name='Logistic_Regression_Polynomial_Features'):
        mlflow_4.set_tags({'Business_Score_Definition': 'Score normalisé basé sur FP/FN pondérés', 'Feature_Engineering': 'Polynomial Features', 'Model_Type': 'LogisticRegression'})
        if 'TARGET' in app_train_poly.columns:
            X = app_train_poly.drop('TARGET', axis=1)
            y = app_train_poly['TARGET']
        else:
            print('Utilisation de poly_target comme variable cible')
            X = app_train_poly
            y = poly_target
        imputer_5 = SimpleImputer_3(strategy='median')
        X = pd_3.DataFrame(imputer_5.fit_transform(X), columns=X.columns)
        test = pd_3.DataFrame(imputer_5.transform(pd_3.DataFrame(app_test_poly, columns=X.columns)), columns=X.columns)
        kf_3 = KFold_3(n_splits=5, shuffle=True, random_state=42)
        metrics_3 = {'auc': [], 'business_score': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        y_true_all_3, y_proba_all_3 = ([], [])
        for train_idx_3, val_idx_3 in kf_3.split(X):
            X_train_fold, X_val_fold = (X.iloc[train_idx_3], X.iloc[val_idx_3])
            y_train_fold, y_val_fold = (y.iloc[train_idx_3], y.iloc[val_idx_3])
            model_3 = LogisticRegression(C=0.0001, solver='lbfgs', max_iter=1000, class_weight='balanced')
            model_3.fit(X_train_fold, y_train_fold)
            y_proba_3 = model_3.predict_proba(X_val_fold)[:, 1]
            y_pred_3 = (y_proba_3 >= 0.5).astype(int)
            y_true_all_3.extend(y_val_fold)
            y_proba_all_3.extend(y_proba_3)
            metrics_3['auc'].append(roc_auc_score_3(y_val_fold, y_proba_3))
            metrics_3['business_score'].append(calculate_normalized_business_score_4(y_val_fold, y_proba_3))
            metrics_3['accuracy'].append(accuracy_score_3(y_val_fold, y_pred_3))
            metrics_3['precision'].append(precision_score_3(y_val_fold, y_pred_3, zero_division=0))
            metrics_3['recall'].append(recall_score_3(y_val_fold, y_pred_3, zero_division=0))
            metrics_3['f1'].append(f1_score_3(y_val_fold, y_pred_3, zero_division=0))
        y_true_all_3 = np_4.array(y_true_all_3)
        y_proba_all_3 = np_4.array(y_proba_all_3)
        y_pred_global_3 = (y_proba_all_3 >= 0.5).astype(int)
        auc_global_3 = roc_auc_score_3(y_true_all_3, y_proba_all_3)
        business_global_3 = calculate_normalized_business_score_4(y_true_all_3, y_proba_all_3)
        accuracy_global_3 = accuracy_score_3(y_true_all_3, y_pred_global_3)
        precision_global_3 = precision_score_3(y_true_all_3, y_pred_global_3, zero_division=0)
        recall_global_3 = recall_score_3(y_true_all_3, y_pred_global_3, zero_division=0)
        f1_global_3 = f1_score_3(y_true_all_3, y_pred_global_3, zero_division=0)
        score_metier_moyen = np_4.mean(metrics_3['business_score'])
        print('\n=== RÉSULTATS VALIDATION CROISÉE ===')
        print(f"AUC moyen : {np_4.mean(metrics_3['auc']):.3f} ± {np_4.std(metrics_3['auc']):.3f}")
        print(f'AUC global : {auc_global_3:.3f}')
        print(f'Score métier moyen : {score_metier_moyen:.3f}')
        print(f'Score métier global : {business_global_3:.3f}')
        print(f'Accuracy global : {accuracy_global_3:.3f}')
        print(f'Precision global : {precision_global_3:.3f}')
        print(f'Recall global : {recall_global_3:.3f}')
        print(f'F1-score global : {f1_global_3:.3f}')
        mlflow_4.log_metrics({'auc_mean': float(np_4.mean(metrics_3['auc'])), 'auc_std': float(np_4.std(metrics_3['auc'])), 'auc_global': float(auc_global_3), 'business_score_mean': float(score_metier_moyen), 'business_score_global': float(business_global_3), 'accuracy_global': float(accuracy_global_3), 'precision_global': float(precision_global_3), 'recall_global': float(recall_global_3), 'f1_global': float(f1_global_3)})
        final_model_3 = LogisticRegression(C=0.0001, solver='lbfgs', max_iter=1000, class_weight='balanced')
        final_model_3.fit(X, y)
        joblib_3.dump(final_model_3, 'Logistic_Regression_Polynomial_FeatureEng.pkl')
        mlflow_4.log_artifact('Logistic_Regression_Polynomial_FeatureEng.pkl')
        test_proba_3 = final_model_3.predict_proba(test)[:, 1]
        submit_3 = pd_3.DataFrame({'SK_ID_CURR': app_test_2['SK_ID_CURR'], 'TARGET': test_proba_3})
        submit_3.to_csv('Logistic_Regression_Polynomial_FeatureEng.csv', index=False)
        signature_2 = infer_signature_3(X, final_model_3.predict(X))
        mlflow_4.sklearn.log_model(sk_model=final_model_3, artifact_path='model', signature=signature_2, registered_model_name='LogisticRegression_Polynomiale_FeatureEng')
    print('\n Entraînement terminé avec succès!')
    print(f' Accès MLflow : {public_url_1.public_url}')
    return SimpleImputer, mlflow, ngrok, np, os, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 7.1.2 Entraînement du modèle de **Forêt Aléatoire** sur les données de train et de test. Stockage centralisé du modèle assuré.
     **Entraînement sur les données de base **avec le feature Engineering _caractéristiques polynomiales_**
    """
    )
    return


@app.cell
def _(
    KFold_4,
    RandomForestClassifier_1,
    SimpleImputer_3,
    accuracy_score_4,
    app_test_2,
    app_test_poly,
    app_train_poly,
    client_3,
    f1_score_4,
    infer_signature_4,
    joblib_4,
    mlflow_5,
    np_4,
    pd_3,
    poly_target,
    precision_score_4,
    public_url_1,
    recall_score_4,
    roc_auc_score_4,
):
    from pyngrok import ngrok
    import mlflow

    from mlflow.models.signature import infer_signature
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import joblib

    def calculate_normalized_business_score_5(y_true, y_proba, threshold=0.5, cost_fn=10, cost_fp=1):
        """Calcule un score normalisé basé sur les faux positifs et faux négatifs."""
        y_pred = (y_proba >= threshold).astype(int)
        fp = np_4.sum((y_pred == 1) & (y_true == 0))
        fn = np_4.sum((y_pred == 0) & (y_true == 1))
        total_cost = fp * cost_fp + fn * cost_fn
        worst_fp = np_4.sum(y_true == 0)
        worst_fn = np_4.sum(y_true == 1)
        worst_cost = worst_fp * cost_fp + worst_fn * cost_fn
        return 1 - total_cost / worst_cost
    experiment_name_4 = 'Credit_Scoring_Tool_FeatureEng'
    try:
        experiment_4 = client_3.get_experiment_by_name(experiment_name_4) or client_3.create_experiment(experiment_name_4)
        mlflow_5.set_experiment(experiment_name_4)
    except Exception as e:
        print(f'Erreur configuration expérience: {str(e)}')
        raise
    with mlflow_5.start_run(run_name='Random_Forest_Polynomial_Features'):
        mlflow_5.set_tags({'Business_Score_Definition': 'Score normalisé basé sur FP/FN pondérés', 'Feature_Engineering': 'Polynomial Features', 'Model_Type': 'RandomForest'})
        if 'TARGET' in app_train_poly.columns:
            X_1 = app_train_poly.drop('TARGET', axis=1, errors='ignore')
            y_1 = app_train_poly['TARGET']
        else:
            print('Utilisation de poly_target comme variable cible')
            X_1 = app_train_poly
            y_1 = poly_target
        imputer_6 = SimpleImputer_3(strategy='median')
        X_1 = pd_3.DataFrame(imputer_6.fit_transform(X_1), columns=X_1.columns)
        test_1 = pd_3.DataFrame(imputer_6.transform(pd_3.DataFrame(app_test_poly, columns=X_1.columns)), columns=X_1.columns)
        kf_4 = KFold_4(n_splits=5, shuffle=True, random_state=42)
        metrics_4 = {'auc': [], 'business_score': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        y_true_all_4, y_proba_all_4 = ([], [])
        for train_idx_4, val_idx_4 in kf_4.split(X_1):
            X_train_fold_1, X_val_fold_1 = (X_1.iloc[train_idx_4], X_1.iloc[val_idx_4])
            y_train_fold_1, y_val_fold_1 = (y_1.iloc[train_idx_4], y_1.iloc[val_idx_4])
            model_4 = RandomForestClassifier_1(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model_4.fit(X_train_fold_1, y_train_fold_1)
            y_proba_4 = model_4.predict_proba(X_val_fold_1)[:, 1]
            y_pred_4 = (y_proba_4 >= 0.5).astype(int)
            y_true_all_4.extend(y_val_fold_1)
            y_proba_all_4.extend(y_proba_4)
            metrics_4['auc'].append(roc_auc_score_4(y_val_fold_1, y_proba_4))
            metrics_4['business_score'].append(calculate_normalized_business_score_5(y_val_fold_1, y_proba_4))
            metrics_4['accuracy'].append(accuracy_score_4(y_val_fold_1, y_pred_4))
            metrics_4['precision'].append(precision_score_4(y_val_fold_1, y_pred_4, zero_division=0))
            metrics_4['recall'].append(recall_score_4(y_val_fold_1, y_pred_4, zero_division=0))
            metrics_4['f1'].append(f1_score_4(y_val_fold_1, y_pred_4, zero_division=0))
        y_true_all_4 = np_4.array(y_true_all_4)
        y_proba_all_4 = np_4.array(y_proba_all_4)
        y_pred_global_4 = (y_proba_all_4 >= 0.5).astype(int)
        auc_global_4 = roc_auc_score_4(y_true_all_4, y_proba_all_4)
        business_global_4 = calculate_normalized_business_score_5(y_true_all_4, y_proba_all_4)
        accuracy_global_4 = accuracy_score_4(y_true_all_4, y_pred_global_4)
        precision_global_4 = precision_score_4(y_true_all_4, y_pred_global_4, zero_division=0)
        recall_global_4 = recall_score_4(y_true_all_4, y_pred_global_4, zero_division=0)
        f1_global_4 = f1_score_4(y_true_all_4, y_pred_global_4, zero_division=0)
        print('\n RÉSULTATS VALIDATION CROISÉE')
        print(f"AUC moyen : {np_4.mean(metrics_4['auc']):.3f} ± {np_4.std(metrics_4['auc']):.3f}")
        print(f'AUC global : {auc_global_4:.3f}')
        print(f"Score métier moyen : {np_4.mean(metrics_4['business_score']):.3f}")
        print(f'Score métier global : {business_global_4:.3f}')
        print(f'Accuracy global : {accuracy_global_4:.3f}')
        print(f'Precision global : {precision_global_4:.3f}')
        print(f'Recall global : {recall_global_4:.3f}')
        print(f'F1-score global : {f1_global_4:.3f}')
        mlflow_5.log_metrics({'auc_mean': float(np_4.mean(metrics_4['auc'])), 'auc_std': float(np_4.std(metrics_4['auc'])), 'auc_global': float(auc_global_4), 'business_score_mean': float(np_4.mean(metrics_4['business_score'])), 'business_score_global': float(business_global_4), 'accuracy_global': float(accuracy_global_4), 'precision_global': float(precision_global_4), 'recall_global': float(recall_global_4), 'f1_global': float(f1_global_4)})
        final_model_4 = RandomForestClassifier_1(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        final_model_4.fit(X_1, y_1)
        joblib_4.dump(final_model_4, 'Random_Forest_Polynomial_FeatureEng.pkl')
        mlflow_5.log_artifact('Random_Forest_Polynomial_FeatureEng.pkl')
        test_proba_4 = final_model_4.predict_proba(test_1)[:, 1]
        submit_4 = pd_3.DataFrame({'SK_ID_CURR': app_test_2['SK_ID_CURR'], 'TARGET': test_proba_4})
        submit_4.to_csv('Random_Forest_Polynomial_FeatureEng.csv', index=False)
        signature_3 = infer_signature_4(X_1, final_model_4.predict(X_1))
        mlflow_5.sklearn.log_model(sk_model=final_model_4, artifact_path='model', signature=signature_3, registered_model_name='RandomForest_Polynomiale_FeatureEng')
    print('\n Entraînement terminé avec succès!')
    print(f' Accès MLflow : {public_url_1.public_url}')
    return mlflow, ngrok


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 7.1.3 Entraînement du modèle de **XG Boost** sur les données de train et de test. Stockage centralisé du modèle assuré.
     **Entraînement sur les données de base **avec le feature Engineering _caractéristiques polynomiales_**
    """
    )
    return


@app.cell
def _(
    KFold_5,
    MlflowClient_6,
    SimpleImputer_3,
    XGBClassifier_1,
    accuracy_score_5,
    app_test_2,
    app_test_poly,
    app_train_poly,
    f1_score_5,
    infer_signature_5,
    joblib_4,
    mlflow_5,
    np_4,
    pd_3,
    poly_target,
    precision_score_5,
    public_url_1,
    recall_score_5,
    roc_auc_score_5,
):
    import os
    import time

    from mlflow.models.signature import infer_signature
    from xgboost import XGBClassifier
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    def calculate_normalized_business_score_6(y_true, y_proba, threshold=0.5, cost_fn=10, cost_fp=1):
        """Calcule un score métier normalisé (0 = pire, 1 = parfait)"""
        y_pred = (y_proba >= threshold).astype(int)
        fp = np_4.sum((y_pred == 1) & (y_true == 0))
        fn = np_4.sum((y_pred == 0) & (y_true == 1))
        total_cost = fp * cost_fp + fn * cost_fn
        worst_cost = np_4.sum(y_true == 0) * cost_fp + np_4.sum(y_true == 1) * cost_fn
        return 1 - total_cost / worst_cost
    mlflow_5.set_tracking_uri(public_url_1.public_url)
    client_4 = MlflowClient_6()
    experiment_name_5 = 'Credit_Scoring_Tool_FeatureEng'
    try:
        experiment_5 = client_4.get_experiment_by_name(experiment_name_5) or client_4.create_experiment(experiment_name_5)
        mlflow_5.set_experiment(experiment_name_5)
    except Exception as e:
        print(f'Erreur configuration expérience: {str(e)}')
        raise
    with mlflow_5.start_run(run_name='XGBoost_Polynomial_Features'):
        mlflow_5.set_tags({'Business_Score_Definition': 'Score normalisé (1 - coût/maximum)', 'Feature_Engineering': 'Polynomial Features', 'Model_Type': 'XGBoost'})
        mlflow_5.log_params({'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8})
        if 'TARGET' in app_train_poly.columns:
            X_2 = app_train_poly.drop('TARGET', axis=1, errors='ignore')
            y_2 = app_train_poly['TARGET']
        else:
            print('Utilisation de poly_target comme variable cible')
            X_2 = app_train_poly
            y_2 = poly_target
        imputer_7 = SimpleImputer_3(strategy='median')
        X_2 = pd_3.DataFrame(imputer_7.fit_transform(X_2), columns=X_2.columns)
        test_2 = pd_3.DataFrame(imputer_7.transform(pd_3.DataFrame(app_test_poly, columns=X_2.columns)), columns=X_2.columns)
        kf_5 = KFold_5(n_splits=5, shuffle=True, random_state=42)
        metrics_5 = {'auc': [], 'business_score': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        y_true_all_5, y_proba_all_5 = ([], [])
        for train_idx_5, val_idx_5 in kf_5.split(X_2):
            X_train_fold_2, X_val_fold_2 = (X_2.iloc[train_idx_5], X_2.iloc[val_idx_5])
            y_train_fold_2, y_val_fold_2 = (y_2.iloc[train_idx_5], y_2.iloc[val_idx_5])
            model_5 = XGBClassifier_1(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')
            model_5.fit(X_train_fold_2, y_train_fold_2)
            y_proba_5 = model_5.predict_proba(X_val_fold_2)[:, 1]
            y_pred_5 = (y_proba_5 >= 0.5).astype(int)
            y_true_all_5.extend(y_val_fold_2)
            y_proba_all_5.extend(y_proba_5)
            metrics_5['auc'].append(roc_auc_score_5(y_val_fold_2, y_proba_5))
            metrics_5['business_score'].append(calculate_normalized_business_score_6(y_val_fold_2, y_proba_5))
            metrics_5['accuracy'].append(accuracy_score_5(y_val_fold_2, y_pred_5))
            metrics_5['precision'].append(precision_score_5(y_val_fold_2, y_pred_5, zero_division=0))
            metrics_5['recall'].append(recall_score_5(y_val_fold_2, y_pred_5, zero_division=0))
            metrics_5['f1'].append(f1_score_5(y_val_fold_2, y_pred_5, zero_division=0))
        y_true_all_5 = np_4.array(y_true_all_5)
        y_proba_all_5 = np_4.array(y_proba_all_5)
        y_pred_global_5 = (y_proba_all_5 >= 0.5).astype(int)
        auc_global_5 = roc_auc_score_5(y_true_all_5, y_proba_all_5)
        business_global_5 = calculate_normalized_business_score_6(y_true_all_5, y_proba_all_5)
        accuracy_global_5 = accuracy_score_5(y_true_all_5, y_pred_global_5)
        precision_global_5 = precision_score_5(y_true_all_5, y_pred_global_5, zero_division=0)
        recall_global_5 = recall_score_5(y_true_all_5, y_pred_global_5, zero_division=0)
        f1_global_5 = f1_score_5(y_true_all_5, y_pred_global_5, zero_division=0)
        print('\nRÉSULTATS VALIDATION CROISÉE')
        print(f"AUC moyen : {np_4.mean(metrics_5['auc']):.3f} ± {np_4.std(metrics_5['auc']):.3f}")
        print(f'AUC global : {auc_global_5:.3f}')
        print(f"Score métier moyen : {np_4.mean(metrics_5['business_score']):.3f}")
        print(f'Score métier global : {business_global_5:.3f}')
        print(f'Accuracy global : {accuracy_global_5:.3f}')
        print(f'Precision global : {precision_global_5:.3f}')
        print(f'Recall global : {recall_global_5:.3f}')
        print(f'F1-score global : {f1_global_5:.3f}')
        mlflow_5.log_metrics({'auc_mean': float(np_4.mean(metrics_5['auc'])), 'auc_std': float(np_4.std(metrics_5['auc'])), 'auc_global': float(auc_global_5), 'business_score_mean': float(np_4.mean(metrics_5['business_score'])), 'business_score_global': float(business_global_5), 'accuracy_global': float(accuracy_global_5), 'precision_global': float(precision_global_5), 'recall_global': float(recall_global_5), 'f1_global': float(f1_global_5)})
        final_model_5 = XGBClassifier_1(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')
        final_model_5.fit(X_2, y_2)
        joblib_4.dump(final_model_5, 'XGBoost_Polynomial_FeatureEng.pkl')
        mlflow_5.log_artifact('XGBoost_Polynomial_FeatureEng.pkl')
        test_proba_5 = final_model_5.predict_proba(test_2)[:, 1]
        submit_5 = pd_3.DataFrame({'SK_ID_CURR': app_test_2['SK_ID_CURR'], 'TARGET': test_proba_5})
        submit_5.to_csv('XGBoost_Polynomial_FeatureEng.csv', index=False)
        signature_4 = infer_signature_5(X_2, final_model_5.predict(X_2))
        mlflow_5.xgboost.log_model(xgb_model=final_model_5, artifact_path='model', signature=signature_4, registered_model_name='XGBoost_Polynomiale_FeatureEng')
    print('\n Entraînement terminé avec succès')
    print(f' Accès MLflow : {public_url_1.public_url}')
    return os, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### 7.1.4 **RESULTAT DES MODELISATIONS AVEC LE FEATURE ENGINEERING FEATURE POLYNOMIALES**""")
    return


@app.cell
def _(pd_3, plt):
    data_1 = {'Modèle': ['Régression Logistique', 'Random Forest', 'XGBoost'], 'AUC moyen': [0.711, 0.734, 0.752], 'AUC global': [0.711, 0.734, 0.752], 'Accuracy': [0.616, 0.919, 0.919], 'Précision': [0.137, 0.561, 0.646], 'Rappel': [0.711, 0.003, 0.002], 'F1-score': [0.23, 0.006, 0.004], 'Score métier': [0.656, 0.534, 0.533]}
    df_1 = pd_3.DataFrame(data_1).set_index('Modèle')
    colors_1 = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1']
    fig_2, ax_2 = plt.subplots(figsize=(12, 6))
    fig_2.patch.set_facecolor('white')
    ax_2.set_facecolor('white')
    df_1.plot(kind='bar', ax=ax_2, color=colors_1, edgecolor='black')
    for container_1 in ax_2.containers:
        ax_2.bar_label(container_1, fmt='%.2f', label_type='edge', fontsize=8, padding=3)
    ax_2.set_title('Comparaison des performances des modèles (feature engineering_Feature Polynomiales)', fontsize=14)
    ax_2.set_ylabel('Valeur normalisée', fontsize=12)
    ax_2.set_xlabel('Modèles', fontsize=12)
    ax_2.legend(title='Métriques', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. XGBoost se démarque avec le meilleur AUC moyen (0.752), indiquant sa bonne capacité à distinguer les classes malgré un rappel (0.002) très faible.

    2. Régression Logistique présente un rappel élevé (0.711) mais au prix d'une faible précision (0.137), ce qui reflète une tendance à sur-prédire les défauts.

    3. Random Forest reste un compromis avec des scores équilibrés, mais inférieurs à ceux de XGBoost en AUC et score métier.


    - **XGBoost est le plus performant en termes de classement global (AUC).**

    - Régression Logistique est la meilleure pour détecter les défauts (rappel), mais avec beaucoup de faux positifs.

    - Le score métier favorise légèrement la régression logistique (0.656) grâce à sa sensibilité accrue aux clients à risque.
    """
    )
    return


@app.cell
def _(pd_3, plt):
    definitions_1 = {'Métrique': ['AUC moyen', 'AUC global', 'Accuracy', 'Précision', 'Rappel', 'F1-score', 'Score métier'], 'Définition': ['MoyenneAUC sur K folds (validation croisée) (capacité à classer correctement).', 'AUC calculé sur l’ensemble des prédictions du modèle (performance globale).', 'Proportion d’exemples bien classés toutes classes confondues.', 'Part des prédictions positives qui sont réellement positives (qualité du modèle #class 1).', 'Part des vrais positifs détectés parmi tous les cas positifs (sensibilité).', 'Moyenne harmonique entre précision et rappel (équilibre détection/fiabilité).', 'Score métier normalisé entre 0 (coût max) et 1 (aucun coût) basé sur FP/FN pondérés.']}
    df_def_1 = pd_3.DataFrame(definitions_1)
    fig_3, ax_3 = plt.subplots(figsize=(13, 4))
    ax_3.axis('off')
    table_1 = ax_3.table(cellText=df_def_1.values, colLabels=df_def_1.columns, cellLoc='left', loc='center')
    table_1.auto_set_font_size(False)
    table_1.set_fontsize(10)
    table_1.scale(1.2, 1.7)
    plt.title(' Définitions des métriques utilisées', fontsize=13, weight='bold', pad=12)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 8. Entraînement sur les caractéristiques basé sur le **domaine**
    - Regresession Logistique
    - Forêt aléatoire
    - XgBoost
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Entraînement sur **les caractéristiques basés sur le domaine**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 8.1 Entraînement du modèle de **Régression Logistique** sur les données de train et de test. Stockage centralisé du modèle assuré.
     **Entraînement sur les données de base **avec le feature Engineering _caractéristiques basées sur le domaine**
    """
    )
    return


@app.cell
def _(
    KFold_5,
    LogisticRegression,
    MlflowClient_7,
    SimpleImputer_4,
    accuracy_score_6,
    app_test_domain,
    app_train_domain,
    f1_score_6,
    infer_signature_5,
    joblib_5,
    mlflow_6,
    np_4,
    pd_3,
    precision_score_6,
    public_url_1,
    recall_score_6,
    roc_auc_score_6,
):
    import os
    import time
    from pyngrok import ngrok
    import mlflow

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.impute import SimpleImputer
    import joblib

    def calculate_business_score(y_true, y_proba, threshold=0.5, cost_fn=10, cost_fp=1):
        """Calcule un score métier entre 0 (pire) et 1 (meilleur), basé sur les erreurs pondérées."""
        y_proba = np_4.array(y_proba)
        y_pred = (y_proba >= threshold).astype(int)
        fp = np_4.sum((y_pred == 1) & (y_true == 0))
        fn = np_4.sum((y_pred == 0) & (y_true == 1))
        total_cost = fp * cost_fp + fn * cost_fn
        worst_cost = np_4.sum(y_true == 0) * cost_fp + np_4.sum(y_true == 1) * cost_fn
        return 1 - total_cost / worst_cost
    mlflow_6.set_tracking_uri(public_url_1.public_url)
    client_5 = MlflowClient_7()
    experiment_name_6 = 'Credit_Scoring_Tool_FeatureEng'
    try:
        print(f"\n=== Configuration de l'expérience '{experiment_name_6}' ===")
        experiment_6 = client_5.get_experiment_by_name(experiment_name_6) or client_5.create_experiment(experiment_name_6)
        mlflow_6.set_experiment(experiment_name_6)
    except Exception as e:
        print(f'Erreur configuration expérience: {str(e)}')
        raise
    with mlflow_6.start_run(run_name='Logistic_Regression_DomainFeatures'):
        print("\nDébut de l'entraînement")
        print('1. Nettoyage des données...')
        app_train_domain_1 = app_train_domain.replace([np_4.inf, -np_4.inf], np_4.nan)
        app_test_domain_1 = app_test_domain.replace([np_4.inf, -np_4.inf], np_4.nan)
        print('2. Imputation des valeurs manquantes...')
        imputer_8 = SimpleImputer_4(strategy='median')
        X_train_4 = imputer_8.fit_transform(app_train_domain_1.drop(['TARGET', 'SK_ID_CURR'], axis=1))
        X_test_4 = imputer_8.transform(app_test_domain_1.drop(['SK_ID_CURR'], axis=1, errors='ignore'))
        print('3. Alignement des features...')
        train = pd_3.DataFrame(X_train_4, columns=app_train_domain_1.drop(['TARGET', 'SK_ID_CURR'], axis=1).columns)
        test_3 = pd_3.DataFrame(X_test_4, columns=app_test_domain_1.drop(['SK_ID_CURR'], axis=1, errors='ignore').columns)
        train_labels_1 = app_train_domain_1['TARGET']
        train, test_3 = train.align(test_3, join='left', axis=1, fill_value=0)
        print('4. Configuration des hyperparamètres...')
        mlflow_6.set_tag('Feature_Engineering', 'Caractéristiques domaine')
        mlflow_6.log_params({'C': 0.0001, 'solver': 'lbfgs', 'max_iter': 1000, 'class_weight': 'balanced'})
        print('\nDébut validation croisée (5 plis)')
        kf_6 = KFold_5(n_splits=5, shuffle=True, random_state=42)
        metrics_6 = {'auc': [], 'business_score': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        y_true_all_6, y_proba_all_6 = ([], [])
        for fold_idx, (train_idx_6, val_idx_6) in enumerate(kf_6.split(train), 1):
            print(f'\n--- Traitement pli {fold_idx}/5 ---')
            X_train_fold_3, X_val_fold_3 = (train.iloc[train_idx_6], train.iloc[val_idx_6])
            y_train_fold_3, y_val_fold_3 = (train_labels_1.iloc[train_idx_6], train_labels_1.iloc[val_idx_6])
            model_6 = LogisticRegression(C=0.0001, solver='lbfgs', max_iter=1000, class_weight='balanced')
            model_6.fit(X_train_fold_3, y_train_fold_3)
            y_proba_6 = model_6.predict_proba(X_val_fold_3)[:, 1]
            y_pred_6 = (y_proba_6 >= 0.3).astype(int)
            y_true_all_6.extend(y_val_fold_3)
            y_proba_all_6.extend(y_proba_6)
            metrics_6['auc'].append(roc_auc_score_6(y_val_fold_3, y_proba_6))
            metrics_6['business_score'].append(calculate_business_score(y_val_fold_3, y_proba_6, threshold=0.3))
            metrics_6['accuracy'].append(accuracy_score_6(y_val_fold_3, y_pred_6))
            metrics_6['precision'].append(precision_score_6(y_val_fold_3, y_pred_6, zero_division=0))
            metrics_6['recall'].append(recall_score_6(y_val_fold_3, y_pred_6, zero_division=0))
            metrics_6['f1'].append(f1_score_6(y_val_fold_3, y_pred_6, zero_division=0))
        print('\nCalcul des métriques globales')
        y_true_all_6 = np_4.array(y_true_all_6)
        y_proba_all_6 = np_4.array(y_proba_all_6)
        y_pred_global_6 = (y_proba_all_6 >= 0.3).astype(int)
        auc_global_6 = roc_auc_score_6(y_true_all_6, y_proba_all_6)
        business_global_6 = calculate_business_score(y_true_all_6, y_proba_all_6, threshold=0.3)
        accuracy_global_6 = accuracy_score_6(y_true_all_6, y_pred_global_6)
        precision_global_6 = precision_score_6(y_true_all_6, y_pred_global_6, zero_division=0)
        recall_global_6 = recall_score_6(y_true_all_6, y_pred_global_6, zero_division=0)
        f1_global_6 = f1_score_6(y_true_all_6, y_pred_global_6, zero_division=0)
        print('\n' + '=' * 40)
        print('RÉSULTATS VALIDATION CROISÉE')
        print('=' * 40)
        print(f"AUC moyen               : {np_4.mean(metrics_6['auc']):.3f} ± {np_4.std(metrics_6['auc']):.3f}")
        print(f'AUC global              : {auc_global_6:.3f}')
        print(f"Score métier moyen      : {np_4.mean(metrics_6['business_score']):.3f}")
        print(f'Score métier global     : {business_global_6:.3f}')
        print(f"Accuracy moyen          : {np_4.mean(metrics_6['accuracy']):.3f}")
        print(f'Accuracy global         : {accuracy_global_6:.3f}')
        print(f"Précision moyenne       : {np_4.mean(metrics_6['precision']):.3f}")
        print(f'Précision globale       : {precision_global_6:.3f}')
        print(f"Rappel moyen            : {np_4.mean(metrics_6['recall']):.3f}")
        print(f'Rappel global           : {recall_global_6:.3f}')
        print(f"F1-score moyen          : {np_4.mean(metrics_6['f1']):.3f}")
        print(f'F1-score global         : {f1_global_6:.3f}')
        print('=' * 40 + '\n')
        mlflow_6.log_metrics({'auc_mean': np_4.mean(metrics_6['auc']), 'auc_std': np_4.std(metrics_6['auc']), 'auc_global': auc_global_6, 'business_score_mean': np_4.mean(metrics_6['business_score']), 'business_score_global': business_global_6, 'accuracy_mean': np_4.mean(metrics_6['accuracy']), 'accuracy_global': accuracy_global_6, 'precision_mean': np_4.mean(metrics_6['precision']), 'precision_global': precision_global_6, 'recall_mean': np_4.mean(metrics_6['recall']), 'recall_global': recall_global_6, 'f1_mean': np_4.mean(metrics_6['f1']), 'f1_global': f1_global_6})
        print('Entraînement final sur toutes les données')
        final_model_6 = LogisticRegression(C=0.0001, solver='lbfgs', max_iter=1000, class_weight='balanced')
        final_model_6.fit(train, train_labels_1)
        print('\nSauvegarde des artefacts')
        joblib_5.dump(final_model_6, 'Logistic_Regression_DomainFeatures.pkl')
        test_proba_6 = final_model_6.predict_proba(test_3)[:, 1]
        submit_6 = pd_3.DataFrame({'SK_ID_CURR': app_test_domain_1['SK_ID_CURR'], 'TARGET': test_proba_6})
        print('\nAperçu des prédictions :')
        print(submit_6.head())
        submit_6.to_csv('Logistic_Regression_DomainFeatures.csv', index=False)
        signature_5 = infer_signature_5(train, final_model_6.predict(train))
        mlflow_6.sklearn.log_model(sk_model=final_model_6, artifact_path='model', signature=signature_5, registered_model_name='LogisticRegression_DomainFeatures')
    print('\n Entraînement terminé avec succès!')
    print(f' Accès MLflow : {public_url_1.public_url}')
    return (
        SimpleImputer,
        app_test_domain_1,
        app_train_domain_1,
        client_5,
        mlflow,
        ngrok,
        os,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 8.2 Entraînement du modèle de **Forêt Aléatoire** sur les données de train et de test. Stockage centralisé du modèle assuré.
     **Entraînement sur les données de base** avec le feature Engineering _caractéristiques basées **sur le domaine**
    """
    )
    return


@app.cell
def _(
    KFold_5,
    RandomForestClassifier_1,
    SimpleImputer_4,
    accuracy_score_6,
    app_test_domain_1,
    app_train_domain_1,
    client_5,
    f1_score_6,
    infer_signature_5,
    joblib_5,
    mlflow_6,
    np_4,
    pd_3,
    precision_score_6,
    public_url_1,
    recall_score_6,
    roc_auc_score_6,
):
    def calculate_business_score_1(y_true, y_proba, threshold=0.5, cost_fn=10, cost_fp=1):
        """Calcule un score métier entre 0 (pire) et 1 (meilleur), basé sur les erreurs pondérées."""
        y_proba = np_4.array(y_proba)
        y_pred = (y_proba >= threshold).astype(int)
        fp = np_4.sum((y_pred == 1) & (y_true == 0))
        fn = np_4.sum((y_pred == 0) & (y_true == 1))
        total_cost = fp * cost_fp + fn * cost_fn
        worst_cost = np_4.sum(y_true == 0) * cost_fp + np_4.sum(y_true == 1) * cost_fn
        return 1 - total_cost / worst_cost
    experiment_name_7 = 'Credit_Scoring_Tool_FeatureEng'
    try:
        experiment_7 = client_5.get_experiment_by_name(experiment_name_7) or client_5.create_experiment(experiment_name_7)
        mlflow_6.set_experiment(experiment_name_7)
    except Exception as e:
        print(f'Erreur configuration expérience: {str(e)}')
        raise
    with mlflow_6.start_run(run_name='Random_Forest_DomainFeatures'):
        print('\n=== Entraînement avec validation croisée sur features basées sur le domaine ===')
        app_train_domain_2 = app_train_domain_1.replace([np_4.inf, -np_4.inf], np_4.nan)
        app_test_domain_2 = app_test_domain_1.replace([np_4.inf, -np_4.inf], np_4.nan)
        imputer_9 = SimpleImputer_4(strategy='median')
        X_train_5 = imputer_9.fit_transform(app_train_domain_2.drop(['TARGET', 'SK_ID_CURR'], axis=1))
        X_test_5 = imputer_9.transform(app_test_domain_2.drop(['SK_ID_CURR'], axis=1, errors='ignore'))
        train_1 = pd_3.DataFrame(X_train_5, columns=app_train_domain_2.drop(['TARGET', 'SK_ID_CURR'], axis=1).columns)
        test_4 = pd_3.DataFrame(X_test_5, columns=app_test_domain_2.drop(['SK_ID_CURR'], axis=1, errors='ignore').columns)
        train_labels_2 = app_train_domain_2['TARGET']
        train_1, test_4 = train_1.align(test_4, join='left', axis=1, fill_value=0)
        mlflow_6.set_tag('Feature_Engineering', 'Caractéristiques domaine: CREDIT_INCOME_PERCENT, ANNUITY_INCOME_PERCENT, etc.')
        mlflow_6.log_params({'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1})
        kf_7 = KFold_5(n_splits=5, shuffle=True, random_state=42)
        metrics_7 = {'auc': [], 'business_score': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        y_true_all_7, y_proba_all_7 = ([], [])
        for train_idx_7, val_idx_7 in kf_7.split(train_1):
            X_train_fold_4, X_val_fold_4 = (train_1.iloc[train_idx_7], train_1.iloc[val_idx_7])
            y_train_fold_4, y_val_fold_4 = (train_labels_2.iloc[train_idx_7], train_labels_2.iloc[val_idx_7])
            model_7 = RandomForestClassifier_1(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model_7.fit(X_train_fold_4, y_train_fold_4)
            y_proba_7 = model_7.predict_proba(X_val_fold_4)[:, 1]
            y_pred_7 = (y_proba_7 >= 0.3).astype(int)
            y_true_fold = np_4.array(y_val_fold_4)
            y_proba_fold = np_4.array(y_proba_7)
            business_score = calculate_business_score_1(y_true_fold, y_proba_fold, threshold=0.3)
            metrics_7['business_score'].append(business_score)
            metrics_7['auc'].append(roc_auc_score_6(y_true_fold, y_proba_fold))
            metrics_7['accuracy'].append(accuracy_score_6(y_true_fold, y_pred_7))
            metrics_7['precision'].append(precision_score_6(y_true_fold, y_pred_7, zero_division=0))
            metrics_7['recall'].append(recall_score_6(y_true_fold, y_pred_7, zero_division=0))
            metrics_7['f1'].append(f1_score_6(y_true_fold, y_pred_7, zero_division=0))
            y_true_all_7.extend(y_true_fold)
            y_proba_all_7.extend(y_proba_fold)
        y_true_all_7 = np_4.array(y_true_all_7)
        y_proba_all_7 = np_4.array(y_proba_all_7)
        y_pred_global_7 = (y_proba_all_7 >= 0.3).astype(int)
        auc_global_7 = roc_auc_score_6(y_true_all_7, y_proba_all_7)
        business_global_7 = calculate_business_score_1(y_true_all_7, y_proba_all_7, threshold=0.3)
        accuracy_global_7 = accuracy_score_6(y_true_all_7, y_pred_global_7)
        precision_global_7 = precision_score_6(y_true_all_7, y_pred_global_7, zero_division=0)
        recall_global_7 = recall_score_6(y_true_all_7, y_pred_global_7, zero_division=0)
        f1_global_7 = f1_score_6(y_true_all_7, y_pred_global_7, zero_division=0)
        print('\n=== RÉSULTATS VALIDATION CROISÉE ===')
        print(f"AUC moyen : {np_4.mean(metrics_7['auc']):.3f} ± {np_4.std(metrics_7['auc']):.3f}")
        print(f'AUC global : {auc_global_7:.3f}')
        print(f"Score métier moyen : {np_4.mean(metrics_7['business_score']):.3f}")
        print(f'Score métier global : {business_global_7:.3f}')
        print(f"Accuracy moyen : {np_4.mean(metrics_7['accuracy']):.3f}")
        print(f'Accuracy global : {accuracy_global_7:.3f}')
        print(f"Précision moyenne : {np_4.mean(metrics_7['precision']):.3f}")
        print(f'Précision globale : {precision_global_7:.3f}')
        print(f"Rappel moyen : {np_4.mean(metrics_7['recall']):.3f}")
        print(f'Rappel global : {recall_global_7:.3f}')
        print(f"F1-score moyen : {np_4.mean(metrics_7['f1']):.3f}")
        print(f'F1-score global : {f1_global_7:.3f}')
        final_model_7 = RandomForestClassifier_1(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        final_model_7.fit(train_1, train_labels_2)
        mlflow_6.log_metrics({'business_score_mean': float(np_4.mean(metrics_7['business_score'])), 'business_score_global': float(business_global_7), 'auc_mean': float(np_4.mean(metrics_7['auc'])), 'auc_global': float(auc_global_7), 'accuracy_mean': float(np_4.mean(metrics_7['accuracy'])), 'accuracy_global': float(accuracy_global_7), 'precision_mean': float(np_4.mean(metrics_7['precision'])), 'precision_global': float(precision_global_7), 'recall_mean': float(np_4.mean(metrics_7['recall'])), 'recall_global': float(recall_global_7), 'f1_mean': float(np_4.mean(metrics_7['f1'])), 'f1_global': float(f1_global_7)})
        signature_6 = infer_signature_5(train_1, final_model_7.predict(train_1))
        mlflow_6.sklearn.log_model(sk_model=final_model_7, artifact_path='model', signature=signature_6, registered_model_name='RandomForest_DomainFeatures')
        joblib_5.dump(final_model_7, 'Random_Forest_DomainFeatures.pkl')
        test_proba_7 = final_model_7.predict_proba(test_4)[:, 1]
        submit_7 = pd_3.DataFrame({'SK_ID_CURR': app_test_domain_2['SK_ID_CURR'], 'TARGET': test_proba_7})
        print('\nAperçu des prédictions :')
        print(submit_7.head())
        submit_7.to_csv('Random_Forest_DomainFeatures.csv', index=False)
    print('\n Entraînement terminé avec succès!')
    print(f' Accès MLflow : {public_url_1.public_url}')
    return app_test_domain_2, app_train_domain_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### 8.3 Entraînement du modèle de **XgBoost** sur les données de train et de test. Stockage centralisé du modèle assuré. **Entraînement sur les données de base** avec le feature Engineering _caractéristiques basées **sur le domaine**""")
    return


@app.cell
def _(
    KFold_5,
    SimpleImputer_4,
    XGBClassifier_1,
    accuracy_score_6,
    app_test_domain_2,
    app_train_domain_2,
    client_5,
    f1_score_6,
    infer_signature_5,
    joblib_5,
    mlflow_6,
    np_4,
    pd_3,
    precision_score_6,
    public_url_1,
    recall_score_6,
    roc_auc_score_6,
):
    def calculate_business_score_2(y_true, y_proba, threshold=0.5, cost_fn=10, cost_fp=1):
        """Calcule un score métier entre 0 (pire) et 1 (meilleur), basé sur les erreurs pondérées."""
        y_proba = np_4.array(y_proba)
        y_pred = (y_proba >= threshold).astype(int)
        fp = np_4.sum((y_pred == 1) & (y_true == 0))
        fn = np_4.sum((y_pred == 0) & (y_true == 1))
        total_cost = fp * cost_fp + fn * cost_fn
        worst_cost = np_4.sum(y_true == 0) * cost_fp + np_4.sum(y_true == 1) * cost_fn
        return 1 - total_cost / worst_cost
    experiment_name_8 = 'Credit_Scoring_Tool_FeatureEng'
    try:
        experiment_8 = client_5.get_experiment_by_name(experiment_name_8) or client_5.create_experiment(experiment_name_8)
        mlflow_6.set_experiment(experiment_name_8)
    except Exception as e:
        print(f'Erreur configuration expérience: {str(e)}')
        raise
    with mlflow_6.start_run(run_name='XGBoost_DomainFeatures'):
        print('\n=== Entraînement avec validation croisée sur features basées sur le domaine ===')
        app_train_domain_3 = app_train_domain_2.replace([np_4.inf, -np_4.inf], np_4.nan)
        app_test_domain_3 = app_test_domain_2.replace([np_4.inf, -np_4.inf], np_4.nan)
        imputer_10 = SimpleImputer_4(strategy='median')
        X_train_6 = imputer_10.fit_transform(app_train_domain_3.drop(['TARGET', 'SK_ID_CURR'], axis=1))
        X_test_6 = imputer_10.transform(app_test_domain_3.drop(['SK_ID_CURR'], axis=1, errors='ignore'))
        train_2 = pd_3.DataFrame(X_train_6, columns=app_train_domain_3.drop(['TARGET', 'SK_ID_CURR'], axis=1).columns)
        test_5 = pd_3.DataFrame(X_test_6, columns=app_test_domain_3.drop(['SK_ID_CURR'], axis=1, errors='ignore').columns)
        train_labels_3 = app_train_domain_3['TARGET']
        train_2, test_5 = train_2.align(test_5, join='left', axis=1, fill_value=0)
        mlflow_6.set_tag('Feature_Engineering', 'Caractéristiques domaine')
        mlflow_6.log_params({'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8})
        kf_8 = KFold_5(n_splits=5, shuffle=True, random_state=42)
        metrics_8 = {'auc': [], 'business_score': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        y_true_all_8, y_proba_all_8 = ([], [])
        for train_idx_8, val_idx_8 in kf_8.split(train_2):
            X_train_fold_5, X_val_fold_5 = (train_2.iloc[train_idx_8], train_2.iloc[val_idx_8])
            y_train_fold_5, y_val_fold_5 = (train_labels_3.iloc[train_idx_8], train_labels_3.iloc[val_idx_8])
            model_8 = XGBClassifier_1(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')
            model_8.fit(X_train_fold_5, y_train_fold_5)
            y_proba_8 = model_8.predict_proba(X_val_fold_5)[:, 1]
            y_pred_8 = (y_proba_8 >= 0.3).astype(int)
            y_true_fold_1 = np_4.array(y_val_fold_5)
            y_proba_fold_1 = np_4.array(y_proba_8)
            business_score_1 = calculate_business_score_2(y_true_fold_1, y_proba_fold_1, threshold=0.3)
            metrics_8['business_score'].append(business_score_1)
            metrics_8['auc'].append(roc_auc_score_6(y_true_fold_1, y_proba_fold_1))
            metrics_8['accuracy'].append(accuracy_score_6(y_true_fold_1, y_pred_8))
            metrics_8['precision'].append(precision_score_6(y_true_fold_1, y_pred_8, zero_division=0))
            metrics_8['recall'].append(recall_score_6(y_true_fold_1, y_pred_8, zero_division=0))
            metrics_8['f1'].append(f1_score_6(y_true_fold_1, y_pred_8, zero_division=0))
            y_true_all_8.extend(y_true_fold_1)
            y_proba_all_8.extend(y_proba_fold_1)
        y_true_all_8 = np_4.array(y_true_all_8)
        y_proba_all_8 = np_4.array(y_proba_all_8)
        y_pred_global_8 = (y_proba_all_8 >= 0.3).astype(int)
        auc_global_8 = roc_auc_score_6(y_true_all_8, y_proba_all_8)
        business_global_8 = calculate_business_score_2(y_true_all_8, y_proba_all_8, threshold=0.3)
        accuracy_global_8 = accuracy_score_6(y_true_all_8, y_pred_global_8)
        precision_global_8 = precision_score_6(y_true_all_8, y_pred_global_8, zero_division=0)
        recall_global_8 = recall_score_6(y_true_all_8, y_pred_global_8, zero_division=0)
        f1_global_8 = f1_score_6(y_true_all_8, y_pred_global_8, zero_division=0)
        print('\n=== RÉSULTATS VALIDATION CROISÉE ===')
        print(f"AUC moyen : {np_4.mean(metrics_8['auc']):.3f} ± {np_4.std(metrics_8['auc']):.3f}")
        print(f'AUC global : {auc_global_8:.3f}')
        print(f"Score métier moyen : {np_4.mean(metrics_8['business_score']):.3f}")
        print(f'Score métier global : {business_global_8:.3f}')
        print(f"Accuracy moyen : {np_4.mean(metrics_8['accuracy']):.3f}")
        print(f'Accuracy global : {accuracy_global_8:.3f}')
        print(f"Précision moyenne : {np_4.mean(metrics_8['precision']):.3f}")
        print(f'Précision globale : {precision_global_8:.3f}')
        print(f"Rappel moyen : {np_4.mean(metrics_8['recall']):.3f}")
        print(f'Rappel global : {recall_global_8:.3f}')
        print(f"F1-score moyen : {np_4.mean(metrics_8['f1']):.3f}")
        print(f'F1-score global : {f1_global_8:.3f}')
        final_model_8 = XGBClassifier_1(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')
        final_model_8.fit(train_2, train_labels_3)
        mlflow_6.log_metrics({'business_score_mean': float(np_4.mean(metrics_8['business_score'])), 'business_score_global': float(business_global_8), 'auc_mean': float(np_4.mean(metrics_8['auc'])), 'auc_global': float(auc_global_8), 'accuracy_mean': float(np_4.mean(metrics_8['accuracy'])), 'accuracy_global': float(accuracy_global_8), 'precision_mean': float(np_4.mean(metrics_8['precision'])), 'precision_global': float(precision_global_8), 'recall_mean': float(np_4.mean(metrics_8['recall'])), 'recall_global': float(recall_global_8), 'f1_mean': float(np_4.mean(metrics_8['f1'])), 'f1_global': float(f1_global_8)})
        signature_7 = infer_signature_5(train_2, final_model_8.predict(train_2))
        mlflow_6.xgboost.log_model(xgb_model=final_model_8, artifact_path='model', signature=signature_7, registered_model_name='XGBoost_DomainFeatures')
        joblib_5.dump(final_model_8, 'XGBoost_DomainFeatures.pkl')
        test_proba_8 = final_model_8.predict_proba(test_5)[:, 1]
        submit_8 = pd_3.DataFrame({'SK_ID_CURR': app_test_domain_3['SK_ID_CURR'], 'TARGET': test_proba_8})
        print('\nAperçu des prédictions :')
        print(submit_8.head())
        submit_8.to_csv('XGBoost_DomainFeatures.csv', index=False)
    print('\n Entraînement terminé avec succès!')
    print(f' Accès MLflow : {public_url_1.public_url}')
    return app_test_domain_3, app_train_domain_3


@app.cell
def _(plt):
    metrics_9 = ['AUC moyen', 'AUC global', 'Accuracy', 'Précision', 'Rappel', 'F1-score', 'Score métier']
    values = [0.756, 0.756, 0.915, 0.402, 0.097, 0.156, 0.571]
    fig_4, ax_4 = plt.subplots(figsize=(10, 6))
    bars = ax_4.bar(metrics_9, values)
    fig_4.patch.set_facecolor('white')
    ax_4.set_facecolor('white')
    for bar in bars:
        height = bar.get_height()
        ax_4.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f'{height:.3f}', ha='center', va='bottom')
    ax_4.set_ylim(0, 1.1)
    ax_4.set_ylabel('Valeur normalisée')
    ax_4.set_title('Performances du meilleure modèle XGBoost')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### 8.4 **RESULTAT DES MODELISATIONS AVEC LE FEATURE ENGINEERING FEATURE BASEE SUR LE DOMAINE**""")
    return


@app.cell
def _(pd_3, plt):
    data_2 = {'Modèle': ['Régression Logistique', 'Random Forest', 'XGBoost'], 'AUC moyen': [0.628, 0.736, 0.756], 'AUC global': [0.628, 0.736, 0.756], 'Accuracy': [0.129, 0.919, 0.915], 'Précision': [0.083, 0.561, 0.402], 'Rappel': [0.977, 0.001, 0.097], 'F1-score': [0.153, 0.002, 0.156], 'Score métier': [0.486, 0.533, 0.571]}
    df_2 = pd_3.DataFrame(data_2).set_index('Modèle')
    colors_2 = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1']
    fig_5, ax_5 = plt.subplots(figsize=(12, 6))
    fig_5.patch.set_facecolor('white')
    ax_5.set_facecolor('white')
    df_2.plot(kind='bar', ax=ax_5, color=colors_2, edgecolor='black')
    for container_2 in ax_5.containers:
        ax_5.bar_label(container_2, fmt='%.3f', label_type='edge', fontsize=8, padding=3)
    ax_5.set_title('Comparaison des performances des modèles (feature engineering: domaine)', fontsize=14)
    ax_5.set_ylabel('Valeur normalisée', fontsize=12)
    ax_5.set_xlabel('Modèles', fontsize=12)
    ax_5.legend(title='Métriques', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(pd_3, plt):
    definitions_2 = {'Métrique': ['AUC moyen', 'AUC global', 'Accuracy', 'Précision', 'Rappel', 'F1-score', 'Score métier'], 'Définition': ['MoyenneAUC sur K folds (validation croisée) (capacité à classer correctement).', 'AUC calculé sur l’ensemble des prédictions du modèle (performance globale).', 'Proportion d’exemples bien classés toutes classes confondues.', 'Part des prédictions positives qui sont réellement positives (qualité du modèle #class 1).', 'Part des vrais positifs détectés parmi tous les cas positifs (sensibilité).', 'Moyenne harmonique entre précision et rappel (équilibre détection/fiabilité).', 'Score métier normalisé entre 0 (coût max) et 1 (aucun coût) basé sur FP/FN pondérés.']}
    df_def_2 = pd_3.DataFrame(definitions_2)
    fig_6, ax_6 = plt.subplots(figsize=(13, 4))
    ax_6.axis('off')
    table_2 = ax_6.table(cellText=df_def_2.values, colLabels=df_def_2.columns, cellLoc='left', loc='center')
    table_2.auto_set_font_size(False)
    table_2.set_fontsize(10)
    table_2.scale(1.2, 1.7)
    plt.title(' Définitions des métriques utilisées', fontsize=13, weight='bold', pad=12)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Le modèle XGBoost avec features basées sur le domaine offre le meilleur compromis global :

    1. Meilleur AUC moyen (0.756)

    2. Meilleur F1-score global (0.156)

    3. Meilleur score métier global (0.571)

    4. Bonne précision (0.402)

    5. Un rappel modéré (0.097), bien qu’inférieur à la régression logistique
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""8.2.4 Modélisation avec le modèle de **XGBoost** sur les **caractéristiques basés sur le domaine** XGBoost le meilleuir **modele**""")
    return


@app.cell
def _(HTML, display):
    file_id = "1xAgn7FdKgCIYe7KEdO0y0IMdwlLcG2Ha"
    display(HTML(f'<img src="https://drive.google.com/uc?id={file_id}" alt="Image" width="600">'))
    return


@app.cell
def _(app_train_domain_3):
    app_train_domain_3.shape
    return


@app.cell
def _(app_test_domain_3):
    app_test_domain_3.shape
    return


@app.cell
def _(app_test_domain_3, app_train_domain_3):
    print('Colonnes dans app_train_domain :', app_train_domain_3.columns.tolist())
    print('Colonnes dans app_test_domain :', app_test_domain_3.columns.tolist())
    return


@app.cell
def _():
    # Vérification des versions
    import xgboost, sklearn, imblearn, mlxtend
    print(f"xgboost: {xgboost.__version__}")       # Doit afficher 1.7.6
    print(f"scikit-learn: {sklearn.__version__}")  # Doit afficher 1.3.2
    print(f"imbalanced-learn: {imblearn.__version__}")  # Doit afficher 0.11.0
    print(f"mlxtend: {mlxtend.__version__}")       # Doit afficher 0.22.0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 8.2.5 Le meilleur modéle qui sera en production avec les paramètres suivants :
    - Gestion du déséquilibre des classe
    - Recherche des meilleuir hyperparametres
    - Business score
    - Sélection de feature validation croisée
    - Metrique de performance visible
    """
    )
    return


@app.cell
def _(
    MlflowClient_8,
    SimpleImputer_4,
    XGBClassifier_1,
    accuracy_score_7,
    app_test_domain_3,
    app_train_domain_3,
    f1_score_7,
    infer_signature_5,
    joblib_5,
    mlflow_7,
    np_4,
    pd_3,
    plt_1,
    precision_score_7,
    recall_score_7,
    roc_auc_score_7,
    sns_1,
):
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    import matplotlib.pyplot as plt
    import seaborn as sns
    import mlflow
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient
    import tempfile
    import requests
    from IPython.display import display, Markdown

    def business_score_2(y_true, y_proba, threshold=0.3, cost_fn=10, cost_fp=1):
        y_pred = (y_proba >= threshold).astype(int)
        fp = np_4.sum((y_pred == 1) & (y_true == 0))
        fn = np_4.sum((y_pred == 0) & (y_true == 1))
        total_cost = fp * cost_fp + fn * cost_fn
        worst_cost = np_4.sum(y_true == 0) * cost_fp + np_4.sum(y_true == 1) * cost_fn
        return 1 - total_cost / worst_cost
    business_scorer = make_scorer(business_score_2, greater_is_better=True, needs_proba=True)
    FEATURES = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'CODE_GENDER_M', 'CNT_CHILDREN', 'DAYS_EMPLOYED', 'DAYS_EMPLOYED_PERCENT', 'NAME_INCOME_TYPE_Working', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'REG_CITY_NOT_WORK_CITY', 'FLAG_OWN_REALTY', 'OCCUPATION_TYPE_Laborers']
    for name_1, df_3 in [('app_train_domain', app_train_domain_3), ('app_test_domain', app_test_domain_3)]:
        missing = [col for col in FEATURES if col not in df_3.columns]
        if missing:
            raise ValueError(f'Colonnes manquantes dans {name_1} : {missing}')

    def preprocess(df):
        df = df[FEATURES].replace([np_4.inf, -np_4.inf], np_4.nan)
        imputer = SimpleImputer_4(strategy='median')
        return pd_3.DataFrame(imputer.fit_transform(df), columns=FEATURES)
    X_3 = preprocess(app_train_domain_3)
    y_3 = app_train_domain_3['TARGET']
    X_test_7 = preprocess(app_test_domain_3)
    pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('clf', XGBClassifier_1(use_label_encoder=False, eval_metric='logloss', random_state=42))])
    param_grid = {'clf__n_estimators': [100, 200], 'clf__max_depth': [3, 6], 'clf__learning_rate': [0.05, 0.1], 'clf__subsample': [0.8, 1.0], 'clf__colsample_bytree': [0.8, 1.0]}
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=business_scorer, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), verbose=2, n_jobs=1)
    print(' Lancement du GridSearchCV basé sur le score métier...')
    grid_search.fit(X_3, y_3)
    print(' Meilleurs hyperparamètres trouvés :')
    print(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    best_model.fit(X_3, y_3)
    y_proba_train = best_model.predict_proba(X_3)[:, 1]
    thresholds = np_4.linspace(0, 1, 100)
    scores = [business_score_2(y_3, y_proba_train, threshold=t) for t in thresholds]
    best_threshold = thresholds[np_4.argmax(scores)]
    print(f' Meilleur seuil optimisé pour le score métier : {best_threshold:.3f}')
    y_pred_9 = (y_proba_train >= best_threshold).astype(int)
    auc = roc_auc_score_7(y_3, y_proba_train)
    accuracy = accuracy_score_7(y_3, y_pred_9)
    precision = precision_score_7(y_3, y_pred_9)
    recall = recall_score_7(y_3, y_pred_9)
    f1 = f1_score_7(y_3, y_pred_9)
    score_metier = business_score_2(y_3, y_proba_train, threshold=best_threshold)
    conf_matrix = confusion_matrix(y_3, y_pred_9)
    mlflow_7.set_tracking_uri('https://credit.ngrok.app')
    client_6 = MlflowClient_8()
    experiment_name_9 = 'Credit_Scoring_Tool_FeatureEng_Production_XgBoost'
    try:
        experiment_9 = client_6.get_experiment_by_name(experiment_name_9)
        if experiment_9 and experiment_9.lifecycle_stage == 'deleted':
            client_6.restore_experiment(experiment_9.experiment_id)
            print(f"Expérience '{experiment_name_9}' restaurée")
        elif not experiment_9:
            client_6.create_experiment(experiment_name_9)
            print(f"Nouvelle expérience '{experiment_name_9}' créée")
        mlflow_7.set_experiment(experiment_name_9)
    except Exception as e:
        print(f'Erreur configuration MLflow : {str(e)}')
        raise
    with mlflow_7.start_run(run_name='XGBoost_BusinessScore'):
        mlflow_7.set_tags({'Business_Score_Definition': 'Score pondéré FNx10 / FPx1', 'Model_Type': 'XGBoost_production', 'Feature_Set': '19 explicites prod'})
        mlflow_7.log_params(grid_search.best_params_)
        mlflow_7.log_metric('AUC', auc)
        mlflow_7.log_metric('Accuracy', accuracy)
        mlflow_7.log_metric('Precision', precision)
        mlflow_7.log_metric('Recall', recall)
        mlflow_7.log_metric('F1-score', f1)
        mlflow_7.log_metric('Business_Score', score_metier)
        signature_8 = infer_signature_5(X_3, best_model.predict(X_3))
        mlflow_7.xgboost.log_model(xgb_model=best_model.named_steps['clf'], artifact_path='model', signature=signature_8, registered_model_name='XGBoost_Production_20Features')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            plt_1.figure(figsize=(5, 4))
            sns_1.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt_1.title('Matrice de confusion - Seuil optimisé')
            plt_1.xlabel('Prédiction')
            plt_1.ylabel('Réalité')
            plt_1.tight_layout()
            plt_1.savefig(temp_file.name)
            mlflow_7.log_artifact(temp_file.name, artifact_path='plots')
            plt_1.show()
    joblib_5.dump(best_model, 'Best_XGBoost_Business_Model.pkl')
    test_proba_9 = best_model.predict_proba(X_test_7)[:, 1]
    test_pred = (test_proba_9 >= best_threshold).astype(int)
    predictions = pd_3.DataFrame({'SK_ID_CURR': app_test_domain_3['SK_ID_CURR'], 'SCORE_RISQUE': test_proba_9, 'PREDICTION': test_pred})
    print('Exemple de prédictions :')
    print(predictions.head())
    plt_1.figure(figsize=(10, 5))
    plt_1.plot(thresholds, scores, label='Score métier')
    plt_1.axvline(x=best_threshold, color='red', linestyle='--', label=f'Seuil optimal = {best_threshold:.2f}')
    plt_1.xlabel('Seuil de classification')
    plt_1.ylabel('Score métier')
    plt_1.title('Optimisation du seuil de classification')
    plt_1.legend()
    plt_1.grid()
    plt_1.tight_layout()
    plt_1.show()
    print('\n=== MÉTRIQUES DE PERFORMANCE ===')
    print(f'AUC : {auc:.3f}')
    print(f'Accuracy : {accuracy:.3f}')
    print(f'Précision : {precision:.3f}')
    print(f'Recall : {recall:.3f}')
    print(f'F1-score : {f1:.3f}')
    print(f'Score métier : {score_metier:.3f}')
    from sklearn.metrics import classification_report
    report = classification_report(y_3, y_pred_9, target_names=['Non Défaut', 'Défaut'])
    print('\n=== Classification Report ===')
    print(report)
    return FEATURES, MlflowClient, best_model, display, mlflow, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#9. Pocessus de mise en production du modèle XGBoost avec le seuil optimal 0.22""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Enregistrement du modèle dans ngrook.
    Puis passage au cloud

    Streamlit Community Cloud est une plateforme qui permet aux utilisateurs de partager publiquement leurs applications Streamlit, d'explorer celles des autres et d’apprendre en communauté. Elle facilite le déploiement rapide d’applications web interactives basées sur Python.

    ----------------------------Pour accéder à Streamlit----------------------------

    - 1.  Il faut se rendre sur **https://share.streamlit.io/**
    - 1.a Il faut sidentifier
    - 1.b Il faut cliquer sur "Create app" et entrer les informations requises
    - 1.c Il faut selectionner "Deploy" a public app from GitHub
    "My code is ready on a GitHub repo, and it is totally awesome."
    - 1.d Il faut remplir le formulaire Deploy an app
    - 1.e Ensuite séléctionner "My Apps" dans la barre supérieure
    - 1.f Cliquer sur l'application elle est visible
    -----------------------------------------------------------------------

    1. Dans l'explorateur _edition de fichier_ nous avons: Un dossier **.streamlit** contenant :

    - Le meilleur modèle de prédiction XGBoost format pkl : **Best_XGBoost_Business_Model**

    - Le fichier **main.py** qui contient le script python du lancement de l'API test en local

    - Le fichier **requirements.txt** qui contient la liste des dépendences à installer pour lancer l'application.

    - Le fichier **streamlit_app.py** qui contient le front le code python qui permet de générer l'interface utilisateur de l'application web " credit scoring tool"

    - Le fichier png d'illustration du front end de l'application:
    **credi_score_demo.png**

    -----------------------------------------------------------------------
                         Pour déployer l'application:

    1. Executer le fichier **"main.py"** dans l'éditeur de text

    2. Insérer dans le terminal **uvicorn main:app --host 0.0.0.0 --port 8000 --reload** pour lancer l'API le message de test réussi =   _{"message":"Bienvenue sur l'API FastAPI - visitez /docs pour explorer"}_

    3. Pour lancer et afficher l'interface utilisateur il faut insérer dans **$ streamlit run streamlit_app.py** et éxécuter. L'interface UI s'affiche en local

    4. On pousse les fichiers dans Github pour les heberger
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Commandes Git de base**

    ---

    1. **git init**
    Cette commande crée un nouveau dépôt Git dans un répertoire.


    ---


    2. Pour créer un nouveau dépôt en spécifiant le nom du projet, utilisez la commande suivante :
    **git init** [nom du projet]


    ---


    3. **git add**
    Cette commande est utilisée pour mettre en staging les modifications de fichiers, les préparant ainsi au prochain commit :git add fichier1.txt


    ---



    4. **git commit**
    Utilisez cette commande pour créer un message de validation pour les modifications, les intégrant ainsi à l’historique de votre projet :
    git commit -m "Ajouter une nouvelle fonctionnalité"


    ---



    5. **git status**
    Cette commande permet d’obtenir des informations précieuses sur les modifications apportées à vos fichiers et sur leur état d’avancement.
    git status

    ---
    6. **git log**
    L’utilisation de base de git log vous permet d’afficher une liste chronologique de l’historique des commits :
    git log
    ---
    7. **git diff**
    Cette commande vous permet de comparer les changements entre votre répertoire de travail et le dernier commit. Par exemple, cette utilisation de git diff identifie les différences dans un fichier spécifique :
    git diff fichier1.txt
    Pour comparer les modifications entre deux commits, procédez comme suit :
    git diff commit1 commit2

    ---
    8. **git rm**
    Cette commande supprime des fichiers de votre répertoire de travail et prépare la suppression pour le prochain commit.
    git rm fichier1.txt
    ---
    9. **git mv**
    Utilisez cette commande pour renommer et déplacer des fichiers dans votre répertoire de travail. Voici la commande Git pour renommer un fichier :
    git mv fichier1.txt fichier2.txt
    ---
    10. **Pour déplacer un fichier dans un autre répertoire, entrez :**
    git mv fichier1.txt nouveau_répertoire/
    ---
    11. **git config**
    Cette commande permet de configurer divers aspects de Git, y compris les informations et les préférences de l’utilisateur. Par exemple, entrez cette commande pour définir votre adresse électronique pour les commits :
    git config --global user.email "votre.email@exemple.com"

    _L’option –global applique les configurations de manière universelle, ce qui a un impact sur votre dépôt local._
    ---
    **Commandes de branchement et de fusion de Git**

    12. **git branch**
    Utilisez cette commande pour gérer les branches dans votre dépôt Git. Voici l’utilisation basique de git branch pour lister toutes les branches existantes :
    git branch
    ---
    13. **Pour créer une branche Git nommée “feature”, utilisez :**
    git branch feature
    ---
    14. **Pour renommer une branche Git, entrez cette commande :**
    git branch -m nom-branche nom-nouvelle-branche
    ---
    15. **git checkout**
    Cette commande vous permet de passer d’une branche à l’autre et de restaurer des fichiers à partir de différents commits.
    Voici un exemple d’utilisation de git checkout pour passer à une branche existante :
    git checkout nom_branche
    Pour ignorer les modifications apportées à un fichier spécifique et revenir au dernier commit, utilisez la commande

    git checkout -- nom_fichier
    ---

    16. **git merge**
    Pour combiner une branche de fonctionnalité ou de sujet avec la branche principale de Git, utilisez cette commande. Voici un exemple d’utilisation de git merge :
    git merge nom_branche
    ---
    17. **git cherry-pick**
    Cette commande vous permet d’appliquer des commits spécifiques d’une branche à une autre sans fusionner une branche entière.
    git cherry-pick commit_hash

    ---
    18. **git rebase**
    Cette commande est utilisée pour appliquer les changements d’une branche Git à une autre en déplaçant ou en combinant les commits. Elle permet de conserver un historique des commits plus propre :
    git rebase main
    ---
    19. **git tag**
    Cette commande marque des points spécifiques dans votre historique Git, tels que v1.0 ou v2.0 :
    git tag v1.0
    ---
    20. **Commandes de dépôt à distance Git**
    **git clone**
    Cette commande crée une copie d’un dépôt distant sur votre machine locale. Une utilisation basique de git clone est de cloner un dépôt depuis GitHub :

    git clone https://github.com/username/mon-projet.git
    ---

    21. **git push**
    Cette commande envoie les commits de votre branche Git locale vers un dépôt distant, le mettant à jour avec vos dernières modifications
    Par exemple, vous souhaitez transférer des modifications du dépôt local appelé “main” vers le dépôt distant appelé “origin” :
    git push origin main
    ---
    22. **git pull**
    Cette commande récupère et intègre les modifications d’un dépôt distant dans votre branche locale actuelle. Voici un exemple d’utilisation de git pull pour récupérer les modifications de la branche master :
    git pull origin master
    ---
    23. **git fetch**
    Pour récupérer les nouveaux commits d’un dépôt distant sans les fusionner automatiquement dans votre branche actuelle, utilisez cette commande :
    git fetch origin
    ---
    24. **git remote**
    Cette commande permet de gérer les dépôts distants associés à votre dépôt local. L’utilisation de git remote affiche le dépôt distant :
    git remote

    **Pour ajouter un nouveau dépôt distant, indiquez son nom et son URL. Par exemple :**
    git remote add origin https://github.com/username/origin.git
    ---
    25. **git submodule**
    Cette commande permet de gérer des dépôts séparés intégrés dans un dépôt Git.
    Pour ajouter un sous-module à votre dépôt principal, utilisez :
    git submodule add https://github.com/username/submodule-repo.git chemin/vers/sous-module

    **Commandes Git avancées**
    ---
    26. **git reset**
    Cette commande permet d’annuler des modifications et de manipuler l’historique de commits. Voici un exemple d’utilisation de git reset pour annuler des modifications :
    git reset fichier1.txt
    ---
    27. **git stash**
    Pour stocker des modifications temporaires qui ne sont pas encore prêtes à être validées, utilisez cette commande :
    git stash
    ---
    28. Pour voir la liste des stashes :
    **git stash list**
    ---
    29. **Pour appliquer le dernier stash et le supprimer de la liste des stashes :**
    git stash pop
    ---
    30. **git bisect**
    Cette commande est principalement utilisée pour identifier les bogues ou les problèmes dans l’historique de votre projet. Pour lancer le processus de bissection, utilisez cette commande :
    ---
    31. **git bisect start**
    Git vous guidera automatiquement à travers les commits pour trouver ceux qui posent problème en utilisant ce qui suit :

    git bisect run <test-script>
    ---
    32. **git blame**
    Cette commande détermine l’auteur et la modification la plus récente de chaque ligne de fichier :
    git blame fichier1.txt
    ---
    33. **git reflog**
    Cette commande enregistre les modifications apportées aux branches Git. Elle vous permet de suivre la chronologie de votre dépôt, même lorsque des commits sont supprimés ou perdus :
    git reflog
    ---
    34. **git clean**
    Enfin, cette commande supprime les fichiers non suivis de votre répertoire de travail, ce qui permet d’obtenir un dépôt propre et organisé :
    ---
    35. **git clean [options]**
    Les [options] peuvent être personnalisées en fonction de vos besoins spécifiques, par exemple -n pour un essai à blanc, -f pour une force ou -d pour des répertoires.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 10. Analyse de l'importance des caractéristiques""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### **SHAP (SHapley Additive exPlanations)**:
    C' est une méthode **d'interprétation des modèles de machine learning** qui utilise la théorie des valeurs de **Shapley** pour expliquer **l'impact de chaque feature sur les prédictions**.

    - Elle permet d'analyser **l'importance des features** à la fois **globalement** (importance moyenne) et **localement** (explications pour une prédiction spécifique).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Variables Financières Fondamentales**


    1. **AMT_INCOME_TOTAL**
    - Définition: Revenu annuel total du client
    - Importance: Fondamentale pour évaluer la capacité de remboursement
    -------------------------------------------------------------------------------
    2. **AMT_CREDIT**

    - Définition: Montant du crédit demandé
    - Importance: Détermine le niveau d'engagement financier du client
    -------------------------------------------------------------------------------

    3. **AMT_ANNUITY**
    - Définition: Montant des paiements annuels pour le crédit
    - Importance: Indique la charge financière régulière imposée au client
    -------------------------------------------------------------------------------
                                  **Ratios Financiers Dérivés**

    4. **CREDIT_INCOME_PERCENT**
    - **Définition**: Ratio entre le montant du crédit et le revenu (AMT_CREDIT/AMT_INCOME_TOTAL)
    - **Importance**: Mesure cruciale du poids relatif de l'emprunt par rapport aux ressources
    ------------------------------------------------------------------------------
    5. **ANNUITY_INCOME_PERCENT**
    - **Définition**: Ratio entre l'annuité et le revenu (AMT_ANNUITY/AMT_INCOME_TOTAL)
    - **Importance**: Évalue la part du revenu consacrée au remboursement mensuel
    -------------------------------------------------------------------------------
    6. **CREDIT_TERM**
    - **Définition**: Durée estimée du crédit en années (AMT_CREDIT/AMT_ANNUITY)
    - **Importance**: Indicateur du temps d'engagement financier
    -------------------------------------------------------------------------------
                                 **Scores Externes**
    7. **EXT_SOURCE_1**
    - **Définition**: Score normalisé provenant d'une source de données externe
    - **Importance**: Évaluation du risque par un système externe
    -------------------------------------------------------------------------------
    8. **EXT_SOURCE_2**
    - **Définition**: Score normalisé provenant d'une seconde source externe
    - **Importance**: Souvent le prédicteur le plus puissant dans les modèles de crédit
    -------------------------------------------------------------------------------
    9. **EXT_SOURCE_3**
    - **Définition**: Score normalisé provenant d'une troisième source externe
    - **Importance**: Complète l'évaluation du risque externe
    -------------------------------------------------------------------------------
                              _ **Variables Démographiques**_

    10. **DAYS_BIRTH** (0.078239)
    -**Définition**: Âge du client en jours négatifs (jours avant la demande)
    -**Importance**: Corrélation élevée (0.078) avec le risque de défaut, facteur démographique crucial


    11. **CODE_GENDER_M** (0.054713)
    - **Définition:** Genre du client (Masculin=1)
    - Importance:** Corrélation significative (0.055) avec le risque de défaut

    12. **CNT_CHILDREN**
    - **Définition**: Nombre d'enfants à charge
    - **Importance**: Impacte directement les charges financières du ménage
    Variables Professionnelles


    13. **DAYS_EMPLOYED** (0.074958)
    - Définition: Nombre de jours d'emploi avant la demande (valeur négative)
    - Importance: Forte corrélation (0.075) avec le défaut, indicateur de stabilité professionnelle


    14. **DAYS_EMPLOYED_PERCENT**
    - Définition: Ratio entre durée d'emploi et âge (DAYS_EMPLOYED/DAYS_BIRTH)
    - Importance: Mesure la stabilité professionnelle relative à l'âge


    15. **NAME_INCOME_TYPE_Working (0.057481)**
    - **Définition**: Type de revenu = Travailleur salarié
    - **Importance**: Corrélation notable (0.057) avec le risque
    -------------------------------------------------------------------------------
                                _**Variables Géographiques**_

    16. **REGION_RATING_CLIENT_W_CITY **(0.060893)
    - **Définition**: Notation de la région incluant données de la ville
    - **Importance**: Corrélation élevée (0.061) avec le défaut, indicateur géographique important


    17. **REGION_RATING_CLIENT (0.058893)**
    - **Définition**: Note générale de la région du client
    - **Importance**: Corrélation significative (0.059) avec le risque


    18. **Variables Comportementales**
    - **REG_CITY_NOT_WORK_CITY** (0.050994)
    - **Définition**: Le client travaille-t-il dans une ville différente de sa résidence
    - **Importance**: Corrélation modérée (0.051), indicateur de mobilité et stabilité


    19. **FLAG_OWN_REALTY**
    - **Définition**: Le client possède-t-il un bien immobilier
    - **Importance**: Indicateur patrimonial important, potentielle garantie


    20. **OCCUPATION_TYPE_Laborers** (0.043019)
    - **Définition**: Profession du client = Ouvrier
    - **Importance**: Corrélation positive (0.043), indique un profil professionnel à risque plus élevé
    """
    )
    return


@app.cell
def _(FEATURES, app_train_domain_3, best_model, plt_1):
    import shap
    from matplotlib.image import imread
    shap.initjs()
    X_shap = app_train_domain_3[FEATURES].copy()
    sample = X_shap.sample(n=min(1000, len(X_shap)), random_state=42)
    explainer = shap.Explainer(best_model.named_steps['clf'])
    shap_values = explainer(sample)
    plt_1.figure(figsize=(24, 14))
    shap.summary_plot(shap_values, sample, plot_type='bar', show=False)
    fig_7 = plt_1.gcf()
    fig_7.patch.set_facecolor('white')
    ax_7 = plt_1.gca()
    ax_7.set_facecolor('white')
    ax_7.grid(False)
    ax_7.tick_params(axis='y', labelsize=12)
    plt_1.title('Impact global moyen des 20 variables', fontsize=20, pad=40)
    plt_1.tight_layout()
    plt_1.savefig('shap_global.png', facecolor='white', dpi=300, bbox_inches='tight')
    plt_1.close()
    plt_1.figure(figsize=(24, 14))
    shap.summary_plot(shap_values, sample, plot_type='dot', show=False)
    fig_7 = plt_1.gcf()
    fig_7.patch.set_facecolor('white')
    ax_7 = plt_1.gca()
    ax_7.set_facecolor('white')
    ax_7.grid(False)
    ax_7.tick_params(axis='y', labelsize=12)
    plt_1.title('Distribution des impacts SHAP', fontsize=20, pad=40)
    plt_1.tight_layout()
    plt_1.savefig('shap_distribution.png', facecolor='white', dpi=300, bbox_inches='tight')
    plt_1.close()
    indices = [0, 42, 123]
    for i_2, idx in enumerate(indices):
        shap_val = shap_values[idx]
        shap_clean = shap.Explanation(values=shap_val.values, base_values=shap_val.base_values, data=[''] * len(shap_val.values), feature_names=[str(f).split('=')[-1].strip() for f in shap_val.feature_names])
        fig_7, ax_7 = plt_1.subplots(figsize=(24, 14))
        shap.plots.waterfall(shap_clean, max_display=10, show=False)
        fig_7.patch.set_facecolor('white')
        ax_7.set_facecolor('white')
        ax_7.grid(False)
        ax_7.tick_params(axis='y', labelsize=12)
        for text in ax_7.texts:
            val = text.get_text()
            if '+' in val or '-' in val:
                text.set_fontsize(11)
                text.set_fontweight('bold')
                text.set_color('black')
        plt_1.title(f'Explication locale - observation {idx}', fontsize=18, pad=20)
        plt_1.tight_layout()
        plt_1.savefig(f'shap_local_{i_2}.png', facecolor='white', dpi=300, bbox_inches='tight')
        plt_1.close()

    def display_shap_plot(filename):
        img = imread(filename)
        plt_1.figure(figsize=(24, 14))
        fig = plt_1.gcf()
        fig.patch.set_facecolor('white')
        plt_1.imshow(img)
        plt_1.axis('off')
        plt_1.tight_layout()
        plt_1.show()
    print(' SHAP - Importance globale :')
    display_shap_plot('shap_global.png')
    print(' SHAP - Distribution des impacts :')
    display_shap_plot('shap_distribution.png')
    print(' SHAP - Explications locales :')
    for i_2 in range(len(indices)):
        display_shap_plot(f'shap_local_{i_2}.png')
    return (imread,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **SHAP - Interprétation globale**

    ---

    1. Graphique barres – Impact global moyen des 20 variables :

    - Représente l’importance moyenne des variables dans les prédictions du modèle.

    - Les variables en haut, comme EXT_SOURCE_3, ont le plus grand effet sur la prédiction, en valeur absolue.

    ---

    - Ce type de graphique est utile pour prioriser les variables à analyser ou à utiliser dans des modèles simplifiés.

    2. Graphique en points (summary plot) – Distribution des impacts SHAP :

    - Chaque point est une observation (un client).

    - La couleur représente la valeur de la variable (bleu = faible, rose = élevée).

    - On visualise à la fois l’importance globale et le sens de l’impact (positif ou négatif).

    ---
    **SHAP - Interprétation locale (explication individuelle)
    Waterfall plots – Observation 0, 42, 123 :**

    ---

    - Ces graphiques montrent comment chaque variable contribue à l'écart entre la prédiction du modèle et la valeur moyenne.

    Les barres bleues diminuent la prédiction, les roses l’augmentent.
    Exemple :
    1. Pour l’observation 0, EXT_SOURCE_3 réduit fortement la probabilité prédite.
    2. Pour l’observation 123, EXT_SOURCE_3 et EXT_SOURCE_1 augmentent fortement la prédiction.

    ---

    - Ces visualisations permettent d'expliquer pourquoi une décision a été prise par le modèle pour un individu donné.

    ---

    - Utilité métier :
    1. Ces graphes sont essentiels pour la transparence et l’acceptabilité des modèles en environnement réglementé (ex. crédit, santé).

    2. Ils permettent aussi d'identifier les leviers d'action pour améliorer une prédiction négative (ex : améliorer EXT_SOURCE_2).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Les scores externes (EXT_SOURCE_3, EXT_SOURCE_1, EXT_SOURCE_2) sont les plus influents, indiquant leur rôle clé dans la prédiction du risque.

    - Les variables socio-économiques comme NAME_INCOME_TYPE_Working et FLAG_OWN_REALTY ont également un impact significatif.

    - Les ratios financiers (CREDIT_TERM, ANNUITY_INCOME_PERCENT) ont une influence moindre.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### **LIME** (Local Interpretable Model-Agnostic Explenations):

    **LIME** **(Local Interpretable Model-Agnostic Explanations)** est une méthode d'interprétation locale qui explique les prédictions d'un modèle de machine learning en approximant son comportement complexe par un modèle linéaire simple autour d'une observation spécifique. Elle est agnostique, ce qui signifie qu'elle peut être utilisée avec tout type de modèle, et se concentre sur l'importance des features pour chaque prédiction individuelle.
    """
    )
    return


app._unparsable_cell(
    r"""
    !pip install lime
    """,
    name="_"
)


@app.cell
def _(
    FEATURES,
    SimpleImputer_4,
    app_train_domain_3,
    best_model,
    imread,
    pd_3,
    plt_1,
):
    from lime.lime_tabular import LimeTabularExplainer
    X_4 = app_train_domain_3[FEATURES].copy()
    y_4 = app_train_domain_3['TARGET']
    imputer_11 = SimpleImputer_4(strategy='mean')
    X_imputed = pd_3.DataFrame(imputer_11.fit_transform(X_4), columns=FEATURES)
    X_sample = X_imputed.sample(n=min(1000, len(X_imputed)), random_state=42)
    X_np = X_sample.values
    predict_fn = best_model.named_steps['clf'].predict_proba
    explainer_1 = LimeTabularExplainer(training_data=X_np, feature_names=FEATURES, class_names=['Non défaillant', 'Défaillant'], mode='classification', discretize_continuous=False)
    indices_1 = [0, 42, 123]
    for i_3, idx_1 in enumerate(indices_1):
        instance = X_np[idx_1]
        exp = explainer_1.explain_instance(instance, predict_fn, num_features=10)
        fig_8 = exp.as_pyplot_figure()
        fig_8.set_size_inches(12, 6)
        fig_8.patch.set_facecolor('white')
        ax_8 = fig_8.gca()
        ax_8.set_facecolor('white')
        ax_8.grid(False)
        plt_1.title(f'LIME - Explication locale - observation {idx_1}', fontsize=18, pad=20)
        plt_1.tight_layout()
        plt_1.savefig(f'lime_local_{i_3}.png', facecolor='white', dpi=300, bbox_inches='tight')
        plt_1.close()

    def display_lime_plot(filename):
        img = imread(filename)
        plt_1.figure(figsize=(14, 6))
        fig = plt_1.gcf()
        fig.patch.set_facecolor('white')
        plt_1.imshow(img)
        plt_1.axis('off')
        plt_1.tight_layout()
        plt_1.show()
    print(' LIME - Explications locales :')
    for i_3 in range(len(indices_1)):
        display_lime_plot(f'lime_local_{i_3}.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##  LIME – Interprétation locale

    LIME permet d'expliquer la prédiction d'un modèle pour un individu donné.

    Chaque graphique montre, pour une observation :
    - les variables qui contribuent **positivement** (en vert) ou **négativement** (en rouge) à la prédiction,
    - l’**amplitude de leur influence locale** sur la sortie du modèle.

    **Lecture :**
    - Une barre rouge indique que la variable **fait baisser la probabilité** du résultat cible.
    - Une barre verte indique qu’elle **la fait augmenter**.

    > Ces visualisations sont utiles pour analyser le **pourquoi d’une décision modèle** à l’échelle individuelle.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 11. Analyse du data drift avec la librairie evidently""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Définition du data drift (dérive de données) et de l'utilité d'évidently""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### **Data Drift**
    Le Data Drift signifie (dérive des données)
    La dérive des données correspond à un changement de la distribution statistique des données d'entrée (features) entre l'entraînement d'un modèle et son usage réel en production.

    Exemple entrînement du modèle avec des clients ayant en moyenne 35 ans.
    Mais en production, on reçoit surtout des clients de 60 ans.
    **Résultat**:
    Le modèle n’est plus adapté → data drift sur DAYS_BIRTH (_L'âge moyen du client initiale n'est plus d'actualité_)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### **Evidently** est une librairie open-source qui :

    1. Compare des datasets (train vs test ou prod),

    2. Génère des rapports visuels (HTML ou JSON),

    3. Détecte les dérives statistiques (Data Drift) ou conceptuelles (Target Drift)

    Est est utile dans une approche MLOps (_Machine Learning Operations_) pour pour suivre l'état du modèle après à la suite de la modèlisation**
    """
    )
    return


@app.cell
def _():
    # '%pip install evidently' command supported automatically in marimo
    return


@app.cell
def _():
    # '%pip install evidently==0.4.17' command supported automatically in marimo
    return


@app.cell
def _(np_4):
    if not hasattr(np_4, 'float'):
        np_4.float = np_4.float64
    if not hasattr(np_4, 'float_'):
        np_4.float_ = np_4.float64
    return


@app.cell
def _(HTML, app_test_domain_3, app_train_domain_3, np_4):
    if not hasattr(np_4, 'float'):
        np_4.float = np_4.float64
    if not hasattr(np_4, 'float_'):
        np_4.float_ = np_4.float64
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    FEATURES_1 = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'CODE_GENDER_M', 'CNT_CHILDREN', 'DAYS_EMPLOYED', 'DAYS_EMPLOYED_PERCENT', 'NAME_INCOME_TYPE_Working', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'REG_CITY_NOT_WORK_CITY', 'FLAG_OWN_REALTY', 'OCCUPATION_TYPE_Laborers']
    reference_data = app_train_domain_3[FEATURES_1].copy()
    current_data = app_test_domain_3[FEATURES_1].copy()
    report_1 = Report(metrics=[DataDriftPreset()])
    report_1.run(reference_data=reference_data, current_data=current_data)
    HTML(report_1.get_html())
    report_1.save_html('rapport_drift_evidently.html')
    print(' Rapport sauvegardé : rapport_drift_evidently.html')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 12. Conception des tests unitaires et execution de manière automatisé lors du build réalisé sur Github""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - L'objectif du test unitaire est de tester l'authenticité d'une portion d'un programme. Ici, il s'agit de notre modèle XGBoost que nous avons mis en production.

    - Le test unitaire peut être utilisé pour la vérification de la valeur d'un attribut ou du bon fonctionnement d'une fonction.

    - Il permet de vérifier que l'instance du modèle ou ses méthodes produisent les résultats attendus.

    - C'est également un outil pour s'assurer de l'absence de régressions de code lors des évolutions.

    - Finalement, avec Pytest (ou Unittest), l'exécution de ces tests est automatisée lors du build sur GitHub Actions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##  **Test 1** - Vérification de la présence de toutes les **features d'entrée** au total 20 features""")
    return


@app.cell
def _(joblib_6):
    import joblib

    def test_pipeline_input_features():
        model = joblib_6.load('/content/Best_XGBoost_Business_Model.pkl')
        expected_features = sorted(['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'CODE_GENDER_M', 'CNT_CHILDREN', 'DAYS_EMPLOYED', 'DAYS_EMPLOYED_PERCENT', 'NAME_INCOME_TYPE_Working', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'REG_CITY_NOT_WORK_CITY', 'FLAG_OWN_REALTY', 'OCCUPATION_TYPE_Laborers'])
        model_features = sorted(model.feature_names_in_.tolist())
        assert model_features == expected_features, 'Les features du modèle ne correspondent pas aux features attendues.'
    return (test_pipeline_input_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##  **Test 2** - Vérification que le modèle XGboost peut faire une prédiction sur un petit batch""")
    return


@app.cell
def _(joblib_6, np_4):
    def test_pipeline_predicts():
        model = joblib_6.load('/content/Best_XGBoost_Business_Model.pkl')
        dummy_input = np_4.random.rand(5, 20)
        proba = model.predict_proba(dummy_input)
        assert proba.shape == (5, 2), "La sortie du modèle n'a pas la bonne forme."
    return (test_pipeline_predicts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##  **Test 3** - Vérification que les probabilités de défaut client prédites sont bien situées entre 0 et 1""")
    return


@app.cell
def _(joblib_6, np_4):
    def test_pipeline_probability_bounds():
        model = joblib_6.load('/content/Best_XGBoost_Business_Model.pkl')
        dummy_input = np_4.random.rand(2, 20)
        proba = model.predict_proba(dummy_input)
        assert np_4.all((proba >= 0) & (proba <= 1)), 'Les probabilités ne sont pas entre 0 et 1.'
    return (test_pipeline_probability_bounds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Exécution des test au niveau local""")
    return


@app.cell
def _(
    test_pipeline_input_features,
    test_pipeline_predicts,
    test_pipeline_probability_bounds,
):
    # Test #1 = Verrifier que les 20 features unniquement sont bien séléctionnées
    test_pipeline_input_features()
    # Test #2 = S'assurer que le modèle peut prédire sans problème (sans crash sur un échantillon donnée)
    test_pipeline_predicts()
    # Test #3 garantir que l'output est correct pour un classifieur probabiliste.
    test_pipeline_probability_bounds()
    print("Les test unitaires sont éffectif et passés avec succès OK")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
