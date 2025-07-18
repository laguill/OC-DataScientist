import marimo

__generated_with = "0.13.7"
app = marimo.App(app_title="P4 Predictions emissions")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Prédictions des Emissions de CO2""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | **Rappel**

    **Pour rappel, voici les colonnes de mon jeu de données après le feature engineering.**

    ```{python}
    col_geo = [
        "CouncilDistrictCode",
        "Neighborhood",
        "Latitude",
        "Longitude",
    ]
    col_archi = [
        "AgeGroup",
        "NumberofBuildings",
        "NumberofFloors",
        "PropertyGFATotal",
        "Property_ratio",
    ]

    col_usage = [
        "PrimaryPropertyType",
    ]

    col_conso = [
        "SiteEnergyUseWN(kBtu)",
        "TotalGHGEmissions",
        "PrimaryEnergySource",
        "ENERGYSTARScore",
    ]

    col_conservees = col_geo + col_archi + col_usage + col_conso
    ```

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Je reprends le notebook précédent et l'adapte pour la nouvelle valaur cible (Emissions de CO2)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Importation

    J'importe le jeu de données et les librairies.
    """
    )
    return


@app.cell
def _():
    import pandas as pd
    import plotly.express as px

    from IPython.display import Markdown, display
    from itables.widget import ITable

    from scripts import analysis_tools as tools

    tools.set_options()
    from itables import init_notebook_mode, show

    data = tools.load_intermediate_data("feature_eng")
    return ITable, Markdown, data, display, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fonctions de coût

    Les fonctions de coût (ou métriques d'évaluation) sont essentielles pour mesurer la performance des modèles de régression. Voici un aperçu rapide des cas d'utilisation et des principes de calcul pour les métriques R², RMSE et MAE :

    -   **R²** (coefficient de détermination):

        -   **Définition** : Le R² mesure le rapport entre les erreurs quadratiques et la variance des données.

        - **Exemple** : Si un modèle fait des erreurs de 1000 euros pour prédire des prix d'appartements, mais que les prix varient de plus de 100,000 euros, le R² sera proche de 1 (bon modèle).

        - **Utilisation** : Le R² est souvent utilisé pour évaluer la performance des modèles de régression.

    -   **MAE** (erreur absolue moyenne):

        -   Calcul : La MAE prend la valeur absolue des erreurs résiduelles et en fait la moyenne.

        - Exemple : Si un modèle prédit des distances de freinage avec des erreurs de 4 mètres et 0 mètre, la MAE sera de 2 mètres.

        - Utilisation : La MAE est utile pour obtenir une moyenne représentative des erreurs de votre modèle.

    -   **RMSE** (racine carré de la moyenne des erreurs moyenne)

        -   Calcul : La MSE prend le carré des erreurs résiduelles et en fait la moyenne.

        - Exemple : Pour les mêmes erreurs (4 mètres et 0 mètre), la MSE sera de 4 mètres carrés, ce qui donne une racine carrée de 2 mètres (RMSE).

        - Utilisation : La MSE est utile pour sélectionner un modèle parmi plusieurs, en privilégiant celui qui fait moins de grandes erreurs.

    **Objectif:** Je vais chercher un modèle qui maximise le R² tout en minimisant les valeurs de MAE et RMSE.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Objectif de cette partie

    L'objectif de cette section est de développer un modèle capable de prédire avec la plus grande précision possible les émissions de CO2 des bâtiments non résidentiels de la ville de Seattle.

    Il s'agit d'un problème de régression, car nous cherchons à prédire des valeurs continues.

    Pour ce faire, nous disposons d'une base de données contenant des informations détaillées sur les bâtiments, ainsi que leurs relevés de consommation énergétique et d'émissions de CO2 pour l'année 2016. Étant donné que les données sont étiquetées, nous utiliserons un apprentissage supervisé.

    Pour ce projet nous avons deux variables cibles (targets) à prédire :

    - **TotalGHGEmissions** : Les émissions totales de gaz à effet de serre.

    - **SiteEnergyUseWN(kBtu)** : La consommation énergétique du site en kilo British Thermal Units.

    Pour éviter le data leakage, nous veillerons à ne pas inclure l'une des deux variables cibles dans la sélection des features.

    (Dans cette partie je cherche à prédire `TotalGHGEmissions`)

    Nous explorerons deux approches pour la sélection des features :

    1. Utiliser toutes les features **sauf** ENERGYSTARScore : Cette approche exclut la variable `ENERGYSTARScore` pour éviter tout biais potentiel.

    2. Utiliser toutes les features y **compris** ENERGYSTARScore : Cette approche inclut la variable `ENERGYSTARScore`, ce qui pourrait améliorer la performance du modèle si cette variable est pertinente.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Développement et simulation d'un premier modèle

    ### Importation des librairies de ML
    """
    )
    return


@app.cell
def _():
    # Timer
    from timeit import default_timer as timer

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Feature Selection
    import shap

    shap.initjs()

    from sklearn.compose import ColumnTransformer, make_column_transformer

    # Modèle dummy
    from sklearn.dummy import DummyRegressor

    # Modèles ensemblistes
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    # Modèles linéaires
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

    # Métriques / Fonctions de coût
    from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

    # Models selections
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
    from sklearn.pipeline import make_pipeline

    # Preprocessing
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Modèle de Support Vector Machin
    from sklearn.svm import SVR

    # Barre de progression
    from tqdm.notebook import tqdm

    return (
        ColumnTransformer,
        DummyRegressor,
        ElasticNet,
        GradientBoostingRegressor,
        GridSearchCV,
        Lasso,
        LinearRegression,
        OneHotEncoder,
        RandomForestRegressor,
        RandomizedSearchCV,
        Ridge,
        SVR,
        StandardScaler,
        make_pipeline,
        mean_absolute_error,
        mean_squared_error,
        np,
        plt,
        r2_score,
        root_mean_squared_error,
        shap,
        sns,
        timer,
        tqdm,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Préparation des données

    #### Définition des variables cibles potentielles
    """
    )
    return


@app.cell
def _():
    # Définition des variables cibles potentielles
    targets = ["TotalGHGEmissions", "SiteEnergyUseWN(kBtu)"]
    return (targets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Choix de la variable cible à prédire""")
    return


@app.cell
def _(targets):
    # Choix de la variable cible à prédire
    target = targets[0]  # TotalGHGEmissions
    return (target,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Selection des features

    Toutes les features sont utilisées pour entraîner le modèle sauf les targets. Un second set de features sans ENERGYSTARScore.
    """
    )
    return


@app.cell
def _(data, targets):
    # Exclusion de TotalGHGEmissions,SiteEnergyUseWN(kBtu), et de ENERGYSTARScore pour une des approches

    features = data.drop(columns=[*targets, "ENERGYSTARScore"])
    return (features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Séparer les features et la cible

    Je scinde les données en deux ensembles : un ensemble d'entraînement pour entraîner le modèle et un ensemble de test pour évaluer ses performances.
    """
    )
    return


@app.cell
def _(data, features, target, train_test_split):
    # Préparation des données pour l'apprentissage
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        data[target],
        test_size=0.2,
        random_state=42,
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(Markdown, X_test, X_train, display, y_train):
    display(Markdown("**Features shapes**"))
    print(X_train.shape)
    print(y_train.shape)

    display(Markdown("**Target shapes**"))
    print(X_test.shape)
    print(y_train.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Identification des colonnes catégorielles et numériques""")
    return


@app.cell
def _(features):
    # Identification des colonnes catégorielles et numériques
    numerical_features = features.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_features = features.select_dtypes(include=["object", "category"]).columns.tolist()
    return categorical_features, numerical_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Définition de l'encodage des features""")
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    StandardScaler,
    categorical_features,
    numerical_features,
):
    # Définition des transformations pour les colonnes catégorielles et numériques

    # preprocessor = make_column_transformer(
    #     (numerical_pipeline, numerical_features),
    #     (categorical_pipeline, categorical_features),
    # )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numerical_features),
            ("cat encode", OneHotEncoder(), categorical_features),
        ]
    )
    return (preprocessor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Evaluation des modèles

    ##### Preprocessing des features

    Appliquer les transformations aux ensembles d'entraînement et de test.
    (Standardisation des valeurs numériques)
    Entraîne les modèles.
    """
    )
    return


@app.cell
def _(X_test, X_train, preprocessor):
    # Appliquer les transformations aux ensembles d'entraînement et de test
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    return X_test_transformed, X_train_transformed


@app.cell
def _(preprocessor):
    feature_names = preprocessor.get_feature_names_out()
    return (feature_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##### Entraînement d'un premier modèle

    Je commence par entraîner un modèle avec un modèle de base : Linear Regression.
    X_train_transformed contient les caractéristiques (features) et y_train contient les valeurs cibles (targets) que tu veux prédire.

    X_test_transformed et y_test sont les données utilisées pour tester le modèle. Elles permettent de voir comment ton modèle se comporte sur des données qu'il n'a jamais vues auparavant.
    """
    )
    return


@app.cell
def _(
    LinearRegression,
    X_test_transformed,
    X_train_transformed,
    timer,
    y_train,
):
    # Création du modèle
    model = LinearRegression()

    # Entraînement du modèle et mesure du temps d'exécution
    start_time = timer()
    model.fit(X_train_transformed, y_train)
    end_time = timer()

    # Prédictions sur les données de test
    predictions = model.predict(X_test_transformed)
    return end_time, model, predictions, start_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##### Evaluation du modèle

    J'évolue la qualité du modèle sur les données d'entraînements et de tests. Pour cela j'utilise les fonctions de coûts R², RMSE et MAE.
    """
    )
    return


@app.cell
def _(
    ITable,
    X_train_transformed,
    end_time,
    mean_absolute_error,
    model,
    pd,
    predictions,
    r2_score,
    root_mean_squared_error,
    start_time,
    y_test,
    y_train,
):
    results = []
    _r2_train = model.score(X_train_transformed, y_train)
    _r2_test = r2_score(y_test, predictions)
    _rmse_test = root_mean_squared_error(y_test, predictions)
    _mae_test = mean_absolute_error(y_test, predictions)
    results.append({
        "model_name": "LinearRegression",
        "r2_train": round(_r2_train, 3),
        "r2_test": round(_r2_test, 3),
        "rmse_test": round(_rmse_test, 3),
        "mae_test": round(_mae_test, 3),
        "fit_time": end_time - start_time,
    })
    results = pd.DataFrame(results)
    # ITable(results, "Evaluation du modèle de Régression")
    results
    return (results,)


@app.cell
def _(px, results):
    _results_long = results.melt(
        id_vars="model_name",
        value_vars=["r2_train", "r2_test"],
        var_name="Dataset",
        value_name="R2",
    )
    _fig = px.bar(
        _results_long,
        x="model_name",
        y="R2",
        color="Dataset",
        barmode="group",
        labels={"model_name": "Model", "R2": "R²"},
        title="Comparison of Model Performance",
        width=600,
        height=400,
    )
    _fig.update_layout(
        legend={
            "title": None,
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        }
    )
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Les R² sont corrects mais perfectibles (0.67). Cela peut indiquer la présence d'overfitting. Mais ce résultat obtenu est seulement sur une découpe possible du trainset. Je vérifie si ce résultat est redondant avec une cross validation."""
    )
    return


@app.cell
def _(plt, predictions, y_test):
    # Visualisation
    # Points des prédictions
    plt.scatter(predictions, y_test, color="blue", label="Prédictions")

    # Droite de prédiction (y = x)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linestyle="--",
        label="Droite de prédiction",
    )

    # Ajout des labels et du titre
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title("Valeurs réelles vs Prédictions")
    plt.legend()

    # Affichage du graphique
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##### Cross validation

    La validation croisée permet d'évaluer la performance de mon modèle de manière plus fiable en utilisant différentes parties des données pour l'entraînement et le test, ce qui aide à vérifier sa capacité à généraliser sur de nouvelles données.

    En appliquant une validation croisée avec cv=5, je divise mes données en 5 parties, entraîne et teste mon modèle 5 fois, puis fais la moyenne des résultats pour obtenir une estimation robuste de sa performance.
    """
    )
    return


@app.cell
def _(X_train_transformed, model, y_train):
    from sklearn.model_selection import cross_val_score

    round(cross_val_score(model, X_train_transformed, y_train, cv=5).mean(), 3)
    return (cross_val_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Le modèle n'est pas très performant (0.645 en moyenne). Je vais en essayer d'autres pour voir s'il y en a de plus prometteurs.

    ##### Modèles de regression

    Scikit learn met à disposition plusieurs modèle adaptés pour prédire des valeurs numériques.

    - Dummy_regression: Prédit la moyenne des données.

    - Linear_regression: Modèle linéaire simple de prédiction.

    - Ridge: Régression linéaire avec régularisation L2.

    - Lasso: Régression linéaire avec régularisation L1.

    - Elastic search: Combine L1 et L2 pour régulariser.

    Sans optimiser ses modèles, je vais comparer leurs valeurs de R² pour voir lequel est le plus performant après une cross validation à 5 folds.

    Je commence par créer un dictionnaire avec comme entrée le nom du model et les hyperparamètres. Pour le moment les hyperparamètres sont par défauts.
    """
    )
    return


@app.cell
def _(DummyRegressor, ElasticNet, Lasso, LinearRegression, Ridge):
    models_param_grid = {
        "Dummy_regression": {
            "model": DummyRegressor(),
            "params": {},
        },
        "Linear_regression": {
            "model": LinearRegression(),
            "params": {},
        },
        "Ridge": {
            "model": Ridge(random_state=42),
            "params": {},
        },
        "Lasso": {
            "model": Lasso(random_state=42),
            "params": {},
        },
        "ElasticNet": {
            "model": ElasticNet(random_state=42),
            "params": {},
        },
    }
    return (models_param_grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Ensuite j'écris une boucle pour tester chaque modèle sur le training set.
    Je stocke le résultat des fonctions de coûts et métriques (R², RMSE, MAE).
    """
    )
    return


@app.cell
def _(
    ITable,
    X_test_transformed,
    X_train_transformed,
    cross_val_score,
    mean_absolute_error,
    mo,
    models_param_grid,
    pd,
    r2_score,
    root_mean_squared_error,
    timer,
    y_test,
    y_train,
):
    with mo.persistent_cache(name="models_init"):
        results_1 = []

        for _model_name, _model_params in models_param_grid.items():
            model_1 = _model_params["model"]
            cv_scores = cross_val_score(
                model_1,
                X_train_transformed,
                y_train,
                cv=5,
                scoring="r2",
                n_jobs=-1,
            )
            r2_train_cv = cv_scores.mean()
            start_time_1 = timer()
            model_1.fit(X_train_transformed, y_train)
            end_time_1 = timer()
            predictions_1 = model_1.predict(X_test_transformed)
            _r2_train = model_1.score(X_train_transformed, y_train)
            _r2_test = r2_score(y_test, predictions_1)
            _rmse_test = root_mean_squared_error(y_test, predictions_1)
            _mae_test = mean_absolute_error(y_test, predictions_1)
            results_1.append({
                "model_name": _model_name,
                "r2_train_cv": round(r2_train_cv, 3),
                "r2_train": round(_r2_train, 3),
                "r2_test": round(_r2_test, 3),
                "rmse_test": round(_rmse_test, 3),
                "mae_test": round(_mae_test, 3),
                "fit_time": end_time_1 - start_time_1,
            })
        results_1 = pd.DataFrame(results_1)
        # ITable(results_1, "Evaluation des modèles de Régression")
        results_1
    return (results_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Voyons les performances des modèles en comparant leurs valeurs de r2 des train et test sets.""")
    return


@app.cell
def _(px, results_1):
    _results_long = results_1.melt(
        id_vars="model_name",
        value_vars=["r2_train_cv", "r2_train", "r2_test"],
        var_name="Dataset",
        value_name="R2",
    )
    _fig = px.bar(
        _results_long,
        x="model_name",
        y="R2",
        color="Dataset",
        barmode="group",
        labels={"model_name": "Model", "R2": "R²"},
        title="Comparaison performance models regression",
        subtitle="sans optimisation des hyperparamètres",
        width=600,
        height=400,
    )
    _fig.update_layout(
        legend={
            "title": None,
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        }
    )
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##### Optimisation des hyperparamètres

    Les hyperparamètres sont des paramètres fixés avant l'entraînement du modèle.

    Ils sont passés en tant qu'arguments au constructeur des classes d'estimateur.

    Cela consiste à trouver la combinaison optimale d'hyperparamètres qui conduit à la meilleure performance du modèle.

    **Intérêts de l'Optimisation des Hyperparamètres :**

    - Améliore la performance du modèle sur des données non vues.

    - Aide à éviter le sur-apprentissage (overfitting) et le sous-apprentissage (underfitting).
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
            On parle d'overfitting lorsque le modèle s'est trop perfectionné sur le Trainset et a perdu tout sens de généralisation.

            $r²_{train} << r²_{test}$
        """
    ).callout("info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Scikit-learn fournit plusieurs outils pour rechercher les meilleurs hyperparamètres : **GridSearchCV** et **RandomizedSearchCV**.

    - **Recherche en grille (Grid Search)** : teste systématiquement toutes les combinaisons d'hyperparamètres spécifiées.

    - **Recherche aléatoire (Random Search) **: teste un sous-ensemble aléatoire de combinaisons d'hyperparamètres.

    Pour le moment, je vais optimiser mes modèles avec GridSearchCV.
    """
    )
    return


@app.cell
def _(DummyRegressor, ElasticNet, Lasso, LinearRegression, Ridge, np):
    models_param_grid_1 = {
        "Dummy_regression": {
            "model": DummyRegressor(),
            "params": {"strategy": ["mean"]},
        },
        "Linear_regression": {"model": LinearRegression(), "params": {}},
        "Ridge": {
            "model": Ridge(random_state=42),
            "params": {"alpha": np.linspace(0.001, 10, 30)},
        },
        "Lasso": {
            "model": Lasso(random_state=0),
            "params": {
                "alpha": np.linspace(0.001, 10, 30),
                "tol": [0.01],
                "max_iter": [10000],
            },
        },
        "ElasticNet": {
            "model": ElasticNet(random_state=0),
            "params": {
                "alpha": np.linspace(0.001, 10, 30),
                "l1_ratio": np.linspace(0.1, 0.9, 5),
            },
        },
    }
    return (models_param_grid_1,)


@app.cell
def _(
    GridSearchCV,
    ITable,
    X_test_transformed,
    X_train_transformed,
    mean_absolute_error,
    mo,
    models_param_grid_1,
    pd,
    r2_score,
    root_mean_squared_error,
    timer,
    tqdm,
    y_test,
    y_train,
):
    with mo.persistent_cache(name="models_param_grid_1"):
        results_2 = []

        for _model_name, _model_params in tqdm(models_param_grid_1.items(), desc="Training models"):
            model_2 = _model_params["model"]
            param_grid = _model_params["params"]

            # Entraînement du modèle de base
            base = model_2
            base.fit(X_train_transformed, y_train)
            base_pred = base.predict(X_train_transformed)
            r2_test_base = r2_score(y_train, base_pred)

            # GriSearch pour trouver hyperparamètres
            model_search = GridSearchCV(model_2, param_grid, cv=5, n_jobs=-1)

            # Entraînement du modèles avec les meilleurs hyperparamètres
            start_time_2 = timer()
            model_search.fit(X_train_transformed, y_train)
            end_time_2 = timer()
            predictions_2 = model_search.predict(X_test_transformed)

            # Calcul des fonctions de coûts

            _r2_test = r2_score(y_test, predictions_2)
            _rmse_test = root_mean_squared_error(y_test, predictions_2)
            _mae_test = mean_absolute_error(y_test, predictions_2)

            # Enregistrement dans une liste
            results_2.append({
                "model_name": _model_name,
                "r2_base": r2_test_base,
                "best_params": model_search.best_params_,
                "best_r2_train": model_search.best_score_,
                "fit_time": end_time_2 - start_time_2,
                "r2_test": round(_r2_test, 3),
                "rmse_test": round(_rmse_test, 3),
                "mae_test": round(_mae_test, 3),
            })

        results_2 = pd.DataFrame(results_2)
        # ITable(results_2, "Evaluation des modèles de Régression")
        results_2
    return (results_2,)


@app.cell
def _(results_2):
    results_2
    return


@app.cell
def _(px, results_2):
    _results_long = results_2.melt(
        id_vars="model_name",
        value_vars=["r2_base", "best_r2_train", "r2_test"],
        var_name="Dataset",
        value_name="R2",
    )
    _fig = px.bar(
        _results_long,
        x="model_name",
        y="R2",
        color="Dataset",
        barmode="group",
        labels={"model_name": "Model", "R2": "R²"},
        title="Comparaison performance models regression",
        subtitle="sans optimisation des hyperparamètres",
        width=600,
        height=400,
    )
    _fig.update_layout(
        legend={
            "title": None,
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        }
    )
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Le modèle Linear_regression semble être le plus performant des modèles avec un R² de 0.67. Le R² du dataset de test 0.67."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Pour cette première analyse, j'ai étudié les modèles de bases de regression mais il existe 4 types famille d'algorithme différents."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Les modèles de regression

    - **Linéaire:** Les modèles linéaires sont les plus simples et les plus interprétables. Ils cherchent à établir une relation linéaire entre les variables d'entrée et la variable cible.

    - **Linear SVR:** À l'origine, les Support Vector Machines (SVM) sont utilisées pour des problèmes de classification. Cependant, elles peuvent être adaptées pour la régression, comme dans le cas du Linear SVR. Ce modèle est utile pour capturer des relations non linéaires en ajustant une marge d'erreur autour des prédictions.

    - **Random Forrest:** Les forêts aléatoires sont un ensemble de modèles basés sur des arbres de décision. Elles combinent les prédictions de plusieurs arbres pour améliorer la précision et la robustesse du modèle. Les Random Forests sont particulièrement efficaces pour gérer les données non linéaires et les interactions entre variables.

    - **Gradient Boosting:** Le Gradient Boosting construit des modèles de manière séquentielle, chaque nouvel arbre corrigeant les erreurs des précédents. C'est une méthode puissante pour obtenir des performances élevées, mais elle peut être plus complexe à ajuster et à interpréter que les autres modèles.

    Basé sur mon code précédent j'écris des fonctions pour faciliter le tests de plusieurs modèles.
    """
    )
    return


@app.cell
def _(
    GridSearchCV,
    ITable,
    X_train_transformed,
    mean_absolute_error,
    pd,
    r2_score,
    root_mean_squared_error,
    timer,
    tqdm,
):
    def model_comparison(
        X_train,
        X_test,
        y_train,
        y_test,
        models_param_grid,
        cv=3,
        n_jobs=-1,
        random_state_cv=42,
        scoring=None,
    ):
        """Compare plusieurs modèles de régression en utilisant la validation croisée.

        Cette fonction teste et compare plusieurs modèles de régression en parallèle
        en utilisant la validation croisée. GridSearchCV est utilisée pour optimiser les hyperparamètres.

        Paramètres
        ----------
        X_train : array-like de forme (n_samples, n_features)
            Échantillons d'entrée pour l'entraînement.
        X_test : array-like de forme (n_samples, n_features)
            Échantillons d'entrée pour le test.
        y_train : array-like de forme (n_samples,)
            Valeurs cibles pour l'entraînement.
        y_test : array-like de forme (n_samples,)
            Valeurs cibles pour le test.
        models_param_grid : dict
            Dictionnaire contenant les noms des modèles comme clés et un autre dictionnaire comme valeurs.
            Le dictionnaire interne doit avoir 'model' comme estimateur et 'params' comme grille de paramètres.
            Exemple :
            ```python
            model_param_grid = {
                'Lasso': {
                    'model': Lasso(),
                    'params': {
                        'alpha': np.linspace(0.1, 5, 10),
                        'tol': [0.01],
                        'max_iter': [10000]
                    }
                }
            }

        cv : int, par défaut 3
            Nombre de segments pour la validation croisée.
        n_jobs : int, par défaut -1
            Nombre de cœurs CPU à utiliser. -1 signifie utiliser tous les processeurs.
        scoring : str or callable, par défaut None
            La métrique à utiliser pour l'optimisation des hyperparamètres.

        Retourne
        -------
        pd.DataFrame
            Un DataFrame contenant les modèles, les paramètres optimaux, les meilleurs scores d'entraînement (R2),
            le temps d'entraînement, et les métriques de test (R2, RMSE, MAE).
        """
        results = []
        for _model_name, _model_params in tqdm(models_param_grid.items(), desc="Training models"):
            model = _model_params["model"]
            param_grid = _model_params["params"]

            # Entrainement du modèle de base
            base = model
            base.fit(X_train_transformed, y_train)
            base_pred = base.predict(X_train)
            r2_test_base = r2_score(y_train, base_pred)

            model_search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                n_jobs=n_jobs,
                scoring=scoring,
            )

            # Entraînement du modèle avec les meilleurs hyperparamètres
            start_time = timer()
            model_search.fit(X_train, y_train)
            end_time = timer()
            predictions = model_search.predict(X_test)

            # Calcul des fonctions de coûts
            _r2_test = r2_score(y_test, predictions)
            _rmse_test = root_mean_squared_error(y_test, predictions)
            _mae_test = mean_absolute_error(y_test, predictions)

            # Enregistrement dans une liste
            results.append({
                "model_name": _model_name,
                "r2_base": round(r2_test_base, 3),
                "best_params": model_search.best_params_,
                "best_r2_train": round(model_search.best_score_, 3),
                "fit_time": end_time - start_time,
                "r2_test": round(_r2_test, 3),
                "rmse_test": round(_rmse_test, 3),
                "mae_test": round(_mae_test, 3),
            })

        results = pd.DataFrame(results)
        # ITable(results, "Evaluation des modèles de Régression")
        results
        return results

    return (model_comparison,)


@app.cell
def _(px):
    def plots_models_score(df):
        _results_long = df.melt(
            id_vars="model_name",
            value_vars=["r2_base", "best_r2_train", "r2_test"],
            var_name="Dataset",
            value_name="R2",
        )
        _fig = px.bar(
            _results_long,
            x="model_name",
            y="R2",
            color="Dataset",
            barmode="group",
            labels={"model_name": "Model", "R2": "R²"},
            title="Comparaison performance models regression",
            width=600,
            height=400,
        )
        _fig.update_layout(
            legend={
                "title": None,
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            }
        )
        _fig.show()

    return (plots_models_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Les paramètres initiaux des modèles que je souhaite tester.""")
    return


@app.cell
def _(
    DummyRegressor,
    ElasticNet,
    GradientBoostingRegressor,
    Lasso,
    LinearRegression,
    RandomForestRegressor,
    Ridge,
    SVR,
    np,
):
    models_param_grid_2 = {
        "Dummy_regression": {
            "model": DummyRegressor(),
            "params": {"strategy": ["mean"]},
        },
        "Linear_regression": {"model": LinearRegression(), "params": {}},
        "Ridge": {
            "model": Ridge(random_state=42),
            "params": {"alpha": np.linspace(0.001, 10, 30)},
        },
        "Lasso": {
            "model": Lasso(random_state=42),
            "params": {
                "alpha": np.linspace(0.001, 10, 30),
                "tol": [0.01],
                "max_iter": [10000],
            },
        },
        "ElasticNet": {
            "model": ElasticNet(random_state=42),
            "params": {"alpha": np.linspace(0.001, 10, 30)},
        },
        "SVR": {
            "model": SVR(),
            "params": {"C": np.logspace(-2, 3, 6), "gamma": np.logspace(-2, 1, 4)},
        },
        "Random_forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {"n_estimators": np.linspace(10, 200, 20, dtype=int)},
        },
        "GradientBoosting_Reg": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "learning_rate": [1, 0.5, 0.25, 0.1, 0.05, 0.01],
                "n_estimators": np.linspace(10, 400, 30, dtype=int),
            },
        },
    }
    return (models_param_grid_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Voyons les performances des modèles en comparant leurs valeurs de r2 des train et test sets.""")
    return


@app.cell
def _(
    X_test_transformed,
    X_train_transformed,
    mo,
    model_comparison,
    models_param_grid_2,
    y_test,
    y_train,
):
    with mo.persistent_cache(name="model_comparison_param_grid2"):
        results_3 = model_comparison(
            X_train_transformed,
            X_test_transformed,
            y_train,
            y_test,
            models_param_grid_2,
            cv=3,
        )
    return (results_3,)


@app.cell
def _(results_3):
    results_3
    return


@app.cell
def _(plots_models_score, results_3):
    plots_models_score(results_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    `SVR`, `Random_forest` et `Gradient_Boosting_Reg` montrent de l'overfitting (R² de train bien meilleur que R² de test).

    A l'inverse `ElasticNet` montre de l'underfitting.

    Je vais conserver l'algorithme `Ridge` qui semble être le meilleur modèle.

    Le modèle standard est meilleur que le modèle optimisé.
    Peut être dû aux hyperparamètres ou bien à cause du grid search qui réduit le nombre de données d’entraînement.
    Je regarde si je peux l'optimiser.

    ### Optimisation du meilleur modèle `Ridge`

    Les paramètres obtenus précédemment sont: `'alpha': 0.3457931034`

    Je vais tester plus de paramètres et le temps de calcul est long avec GridSearch.
    A la place j'utilise RandomSearchCV qui test aléatoirement des combinaisons d'hyperparamètres pour réduire le temps de calcul.
    Ici le but n'est pas de trouver **LE** meilleurs model mais seulement un meilleur modèle.

    Ainsi ma fonction devient.
    """
    )
    return


@app.cell
def _(
    GridSearchCV,
    ITable,
    RandomizedSearchCV,
    mean_absolute_error,
    pd,
    r2_score,
    root_mean_squared_error,
    timer,
    tqdm,
):
    def model_comparison_v2(
        X_train,
        X_test,
        y_train,
        y_test,
        models_param_grid,
        cv=3,
        n_jobs=-1,
        scoring=None,
        random_state_cv=42,
        randomized_n_iter=10,
        *,
        randomized_state_cv=False,
    ):
        """Compare plusieurs modèles de régression en utilisant la validation croisée.

        Cette fonction teste et compare plusieurs modèles de régression en parallèle
        en utilisant la validation croisée.
        L'argument randomized_cv = True permet de lancer RandomizedSearchCV à la place de GridSearchCV.
        Le random-state et le nombre d'itérations peuvent ensuite être renseignés.

        Paramètres
        ----------
        X_train : array-like de forme (n_samples, n_features)
            Échantillons d'entrée pour l'entraînement.
        X_test : array-like de forme (n_samples, n_features)
            Échantillons d'entrée pour le test.
        y_train : array-like de forme (n_samples,)
            Valeurs cibles pour l'entraînement.
        y_test : array-like de forme (n_samples,)
            Valeurs cibles pour le test.
        models_param_grid : dict
            Dictionnaire contenant les noms des modèles comme clés et un autre dictionnaire comme valeurs.
            Le dictionnaire interne doit avoir 'model' comme estimateur et 'params' comme grille de paramètres.
            Exemple :
            ```python
            model_param_grid = {
                'Lasso': {
                    'model': Lasso(),
                    'params': {
                        'alpha': np.linspace(0.1, 5, 10),
                        'tol': [0.01],
                        'max_iter': [10000]
                    }
                }
            }

        cv : int, par défaut 3
            Nombre de segments pour la validation croisée.
        n_jobs : int, par défaut -1
            Nombre de cœurs CPU à utiliser. -1 signifie utiliser tous les processeurs.
        scoring : str or callable, par défaut None
            La métrique à utiliser pour l'optimisation des hyperparamètres.

        random_state_cv: Contrôle l'aléatoire du RandomizedSearch
        randomized_n_iter: nombre de combinaisons testées

        Retourne
        -------
        pd.DataFrame
            Un DataFrame contenant les modèles, les paramètres optimaux, les meilleurs scores d'entraînement (R2),
            le temps d'entraînement, et les métriques de test (R2, RMSE, MAE).
        """
        results = []
        for _model_name, _model_params in tqdm(models_param_grid.items(), desc="Training models"):
            model = _model_params["model"]
            param_grid = _model_params["params"]

            # Entrainement du modèle de base
            base = model
            base.fit(X_train, y_train)
            base_pred = base.predict(X_train)
            r2_test_base = r2_score(y_train, base_pred)

            # Test les hyperparametres avec RandomizedSearch ou GridSearch
            if randomized_state_cv:
                model_search = RandomizedSearchCV(
                    model,
                    param_grid,
                    cv=cv,
                    n_jobs=n_jobs,
                    random_state=random_state_cv,
                    n_iter=randomized_n_iter,
                    verbose=2,
                    scoring=scoring,
                )
            else:
                model_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=cv,
                    n_jobs=n_jobs,
                    verbose=2,
                )

            # Entraine le modèle avec les meilleurs hyperparamètres
            start_time = timer()
            model_search.fit(X_train, y_train)
            end_time = timer()
            predictions = model_search.predict(X_test)

            # Calculs des fonctions de coûts
            _r2_test = r2_score(y_test, predictions)
            _rmse_test = root_mean_squared_error(y_test, predictions)
            _mae_test = mean_absolute_error(y_test, predictions)

            # Enregistre les résultats dans une liste
            results.append({
                "model_name": _model_name,
                "r2_base": round(r2_test_base, 3),
                "best_params": model_search.best_params_,
                "best_r2_train": round(model_search.best_score_, 3),
                "fit_time": end_time - start_time,
                "r2_test": round(_r2_test, 3),
                "rmse_test": round(_rmse_test, 3),
                "mae_test": round(_mae_test, 3),
            })

        results = pd.DataFrame(results)
        # ITable(results, "Evaluation des modèles de Régression")
        results
        return results

    return (model_comparison_v2,)


@app.cell
def _(Ridge, np):
    models_param_grid_3 = {
        "Ridge": {
            "model": Ridge(random_state=42),
            "params": {
                "alpha": np.logspace(-4, 4, 1000),
            },
        }
    }
    return (models_param_grid_3,)


@app.cell
def _(
    X_test_transformed,
    X_train_transformed,
    mo,
    model_comparison_v2,
    models_param_grid_3,
    y_test,
    y_train,
):
    with mo.persistent_cache(name="model_comparison_param_grid_3"):
        result = model_comparison_v2(
            X_train_transformed,
            X_test_transformed,
            y_train,
            y_test,
            models_param_grid_3,
            cv=5,
            random_state_cv=True,
            randomized_n_iter=200,
        )
    return (result,)


@app.cell
def _(result):
    result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Le Random Search renvoie ces paramètres pour le modèle Ridge:
    `{alpha': 0.8708431498}`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Analyse de la courbe d'apprentissage

    La courbe d'apprentissage (learning curve) permet de tester la précision du modèle sur des échantillons du dataframe de plus en plus grand.
    """
    )
    return


@app.cell
def _():
    from sklearn.model_selection import learning_curve

    return


@app.cell
def _(np, plt):
    from sklearn.model_selection import LearningCurveDisplay

    def evaluation(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        cv=5,
    ):
        model.fit(X_train, y_train)
        ypred = model.predict(X_test)

        learn_disp = LearningCurveDisplay.from_estimator(
            model,
            X_train,
            y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv,
            n_jobs=-1,
            score_name="Score ($R²$)",
            std_display_style=None,
            line_kw={"marker": "o"},
        )

        learn_disp.figure_.set_size_inches(12, 8)
        learn_disp.ax_.legend(loc="lower right")
        learn_disp.ax_.set(title="Courbe d'apprentissage du jeu de données")

        plt.show()

        return ypred

    return (evaluation,)


@app.cell
def _(Ridge):
    ridge_model = Ridge(
        random_state=42,
        alpha=0.8708431498,
    )
    return (ridge_model,)


@app.cell
def _(
    X_test_transformed,
    X_train_transformed,
    evaluation,
    ridge_model,
    y_test,
    y_train,
):
    predictions_v1 = evaluation(
        ridge_model,
        X_train_transformed,
        X_test_transformed,
        y_train,
        y_test,
        cv=5,
    )
    return (predictions_v1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Analyse de l'importance des variables

    L'importance des variables sera ici testé par la méthode de Shapley
    Plus la valeur SHAP est importante, plus la variable affecte la valeur prédite.
    """
    )
    return


@app.cell
def _(X_train_transformed, ridge_model, y_train):
    ridge_model.fit(X_train_transformed, y_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Interprétabilité Locale

    L'interprétation locale des modèles consiste en un ensemble de techniques destinées à répondre à des questions telles que :

    - Pourquoi le modèle a-t-il fait cette prédiction spécifique ?

    - Quel a été l'impact de cette valeur de caractéristique spécifique sur la prédiction ?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Valeurs de Shapley""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Les valeurs de Shapley, dans le contexte de l'interprétation des modèles de machine learning, permettent d'évaluer l'impact de chaque caractéristique sur une prédiction donnée.

    Elles le font en calculant la contribution moyenne de chaque caractéristique à la prédiction, en considérant toutes les combinaisons possibles de caractéristiques.
    """
    )
    return


@app.cell
def _(
    X_test_transformed,
    X_train_transformed,
    feature_names,
    mo,
    ridge_model,
    shap,
):
    with mo.persistent_cache(name="shap_values"):
        # Calcul des valeurs SHAP pour les instances de test
        explainer = shap.Explainer(ridge_model, X_train_transformed, feature_names=feature_names)

        # Calcul des valeurs SHAP pour les instances de test
        shap_values = explainer(X_test_transformed)
    return (shap_values,)


@app.cell
def _(shap, shap_values):
    # Visualiser les valeurs de Shapley pour une prédiction spécifique
    shap.plots.bar(shap_values[0])
    return


@app.cell
def _(shap, shap_values):
    shap.plots.bar(shap_values, max_display=15)
    return


@app.cell
def _(shap, shap_values):
    shap.plots.beeswarm(shap_values)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Ici on peut voir une corrélation positive entre la superficie du bâtiment et l'émission de CO2. Je constate également que j'ai une majorité de bâtiments petite taille."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Entraînement du modèle avec energy star score

    EnergyStarScore contient des valeurs manquantes.
    Pour ne pas influencer le résultat du machine learning, je supprime les bâtiments pour lesquelles l'information est manquante.
    """
    )
    return


@app.cell
def _(data):
    data[["ENERGYSTARScore"]].isna().value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""On réduit le jeu de donnée de moitié.""")
    return


@app.cell
def _(data):
    data_ess = data.dropna(subset="ENERGYSTARScore")
    return (data_ess,)


@app.cell
def _(data_ess, targets):
    features_v2 = data_ess.drop(columns=targets)
    return (features_v2,)


@app.cell
def _(data_ess, features_v2, target, train_test_split):
    # Préparation des données pour l'apprentissage
    X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(
        features_v2,
        data_ess[target],
        test_size=0.2,
        random_state=42,
    )
    return X_test_v2, X_train_v2, y_test_v2, y_train_v2


@app.cell(hide_code=True)
def _(Markdown, X_test_v2, X_train_v2, display, y_train_v2):
    display(Markdown("**Features shapes**"))
    print(X_train_v2.shape)
    print(y_train_v2.shape)

    display(Markdown("**Target shapes**"))
    print(X_test_v2.shape)
    print(y_train_v2.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Identification des colonnes catégorielles et numériques""")
    return


@app.cell
def _(features_v2):
    # Identification des colonnes catégorielles et numériques

    numerical_features_v2 = features_v2.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_features_v2 = features_v2.select_dtypes(include=["object", "category"]).columns.tolist()
    return categorical_features_v2, numerical_features_v2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Définition de l'encodage des features""")
    return


@app.cell
def _(OneHotEncoder, StandardScaler, make_pipeline):
    numerical_pipeline_v2 = make_pipeline(StandardScaler())

    categorical_pipeline_v2 = make_pipeline(OneHotEncoder())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Evaluation des modèles

    ##### Preprocessing des features

    Appliquer les transformations aux ensembles d'entraînement et de test.
    (Standardisation des valeurs numériques)
    """
    )
    return


@app.cell(hide_code=True)
def _(
    ColumnTransformer,
    OneHotEncoder,
    StandardScaler,
    categorical_features_v2,
    numerical_features_v2,
):
    # Définition des transformations pour les colonnes catégorielles et numériques

    preprocessor_v2 = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numerical_features_v2),
            ("cat encode", OneHotEncoder(), categorical_features_v2),
        ]
    )
    preprocessor_v2
    return (preprocessor_v2,)


@app.cell
def _(X_test_v2, X_train_v2, preprocessor_v2):
    # Appliquer les transformations aux ensembles d'entraînement et de test
    X_train_transformed_v2 = preprocessor_v2.fit_transform(X_train_v2)
    X_test_transformed_v2 = preprocessor_v2.transform(X_test_v2)
    return X_test_transformed_v2, X_train_transformed_v2


@app.cell
def _(preprocessor_v2):
    feature_names_v2 = preprocessor_v2.get_feature_names_out()
    return (feature_names_v2,)


@app.cell
def _(ridge_model):
    ridge_model
    return


@app.cell
def _(
    X_test_transformed_v2,
    X_train_transformed_v2,
    evaluation,
    ridge_model,
    y_test_v2,
    y_train_v2,
):
    predictions_v2 = evaluation(
        ridge_model,
        X_train_transformed_v2,
        X_test_transformed_v2,
        y_train_v2,
        y_test_v2,
        cv=5,
    )
    return (predictions_v2,)


@app.cell
def _(
    mean_squared_error,
    np,
    predictions_v2,
    r2_score,
    root_mean_squared_error,
    y_test_v2,
):
    print("$R^2$ (Test Set) :", round(r2_score(y_test_v2, predictions_v2), 3))
    print("MSE (Test Set) :", round(mean_squared_error(y_test_v2, predictions_v2), 3))
    print(
        "RMSE (Test Set) :",
        round(root_mean_squared_error(y_test_v2, predictions_v2), 3),
    )
    print("MAE (Test Set) :", round(np.mean(abs(predictions_v2 - y_test_v2)), 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Comparaison des prédictions avec et sans EnergyStarScore

    Lors de l'exploration du jeu de données j'ai appliqué une transformation logarithmique.
    """
    )
    return


@app.cell
def _(plt, predictions_v1, sns, y_test):
    sns.kdeplot(predictions_v1, label="Without ENERGYSTARScore", fill=True)
    sns.kdeplot(y_test, label="Target", fill=True)

    plt.xlabel("TotalGHGEmissions (log(t CO2eq)) : Predicted")
    plt.ylabel("Density")
    plt.title("Prédictions du modèle sans ENERGYSTARScore")
    plt.legend()
    plt.show()
    return


@app.cell
def _(plt, predictions_v2, sns, y_test_v2):
    sns.kdeplot(predictions_v2, label="With ENERGYSTARScore", fill=True, color="purple")
    sns.kdeplot(y_test_v2, label="Target", fill=True, color="teal")

    plt.xlabel("TotalGHGEmissions (log(t CO2eq)) : Predicted")
    plt.ylabel("Density")
    plt.title("Prédictions du modèle avec ENERGYStarScore")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    La visualisation ci-dessus compare les distributions des prédictions des deux modèles (avec et sans ENERGYSTARScore) en utilisant des courbes KDE (Kernel Density Estimation). Cela permet de voir comment les prédictions sont réparties et de comparer la densité des valeurs prédites par les deux modèles.

    - **Sans ENERGYSTARScore**: La courbe KDE pour les prédictions du modèle sans ENERGYSTARScore.
    - **Avec ENERGYSTARScore**: La courbe KDE pour les prédictions du modèle avec ENERGYSTARScore.

    Lorsque le modèle a comme information la valeur de EnergyStarScore, il semble plus précis. ➡️ La courbe de prédictions s'approche d'une distribution gaussiène.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Voyons maintenant les diagrammes de shap pour voir si mon intuition est juste.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Analyse de l'importance des variables en tenant compte de EnergyStarScore

    L'importance des variables sera ici testée par la méthode de Shapley.
    Plus la valeur SHAP est importante, plus la variable affecte la valeur prédite.
    """
    )
    return


@app.cell(hide_code=True)
def _(X_train_transformed, ridge_model, y_train):
    ridge_model.fit(X_train_transformed, y_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Valeurs de Shapley""")
    return


@app.cell
def _(
    X_test_transformed_v2,
    X_train_transformed_v2,
    feature_names_v2,
    mo,
    ridge_model,
    shap,
):
    with mo.persistent_cache(name="shap_values_v2"):
        # Calcul des valeurs SHAP pour les instances de test
        explainer_v2 = shap.Explainer(ridge_model, X_train_transformed_v2, feature_names=feature_names_v2)

        # Calcul des valeurs SHAP pour les instances de test
        shap_values_v2 = explainer_v2(X_test_transformed_v2)
    return (shap_values_v2,)


@app.cell
def _(shap, shap_values_v2):
    # Visualiser les valeurs de Shapley pour une prédiction spécifique
    shap.plots.bar(shap_values_v2[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Importance Globale des Variables

    L'importance globale des variables permet de comprendre quelles caractéristiques ont le plus d'impact sur les prédictions du modèle en général.
    """
    )
    return


@app.cell
def _(shap, shap_values_v2):
    shap.plots.beeswarm(shap_values_v2)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Conclusion

    Les analyses locales et globales des valeurs de Shapley permettent de comprendre comment chaque caractéristique influence les prédictions du modèle. Les variables avec des valeurs SHAP élevées ont un impact significatif sur les prédictions, tandis que celles avec des valeurs SHAP faibles ont un impact moins marqué.

    Le modèle est moins performant pour prédire les émissions carbones des bâtiments.
    Encore une fois la caractéristique EnergyStarScore améliore la faibilité du modèle.
    """
    )
    return


if __name__ == "__main__":
    app.run()
