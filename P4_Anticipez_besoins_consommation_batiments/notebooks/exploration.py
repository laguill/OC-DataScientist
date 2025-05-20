import marimo

__generated_with = "0.13.7"
app = marimo.App(app_title="P4 Exploration")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - **But de la mission** :

    Aider la ville de Seattle à atteindre son objectif de ville neutre en émission de carbone en 2050.\n
    Pour ce projet, je suis chargé d'étudier les relevés réalisés par les agents de la ville des bâtiments non destinés à l'habitation.\n
    Puis je dois prédire les émissions de CO2  et la consommation totale d'énergie pour des futurs bâtiments sans avoir besoin d'effectuer des relevés.

    - **Détails de la mission** :

    1. Analyser les relevés de 2016 de la ville de Seattle.
    2. Sélectionnes les features les plus pertinentes pour le modèle de prédiction d'émissions.
    3. Identifier les valeurs aberrantes et les bâtiments destinés à l'habitation.
    4. Entraîner un premier modèle de machine learning.
    5. Étudier les points faibles du modèle et chercher à l'améliorer en optimisant la sélection des features.
    6. Entraîner plusieurs modèles et le comparer pour sélectionne l'optimal.
    7. Documenter les forces et faiblesses du modèle conservé.
    8. Analyser l'influence de l'indicateur EnergyStarScore sur le modèle.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Definition des acteurs

    - **Seattle**:

    Seattle, la plus grande ville de l'État de Washington, est située dans le Nord-Ouest des États-Unis.\n
    La ville bénéficie d'un climat tempéré océanique, caractérisé par des hivers doux et humides et des étés semi-arides avec un nombre important de jours de canicule.\n
    Seattle s'est engagée dans plusieurs initiatives pour réduire ses émissions de carbone, notamment en promouvant les énergies renouvelables et en améliorant l'efficacité énergétique des bâtiments **non habitables**.

    - **Emissions de CO2 et consommation totale d'énergie**:

    Le dioxyde de carbone (CO2) est un gaz à effet de serre majeur qui contribue au réchauffement climatique.\n
    Les émissions de CO2 peuvent être d'origine naturelle ou résulter des activités humaines, telles que la combustion de combustibles fossiles pour la production d'énergie.\n
    Dans ce projet, les émissions de CO2 sont estimées en fonction de la consommation totale d'énergie des bâtiments, ce qui permet de cibler les efforts de réduction des émissions de manière plus efficace.

    - **Energy Star Score**:

    L'Energy Star Score est un système de notation américain qui évalue l'efficacité énergétique d'un bâtiment par rapport à d'autres bâtiments similaires.\n
    Ce score, compris entre 0 et 100, est calculé en tenant compte de divers facteurs tels que l'emplacement géographique, le type d'énergie utilisée, et la consommation d'électricité.\n
    Un score de 50 représente une performance médiane, tandis qu'un score de 75 ou plus positionne le bâtiment parmi les 25 % les plus performants.\n
    Les métriques utilisées pour évaluer les bâtiments varient en fonction de leur type, par exemple, les critères pour un appartement diffèrent de ceux pour un cabinet médical ou une piscine.

    ## Formulation de la problématique

    La ville de Seattle a besoin d'outils pour estimer les émissions de CO2 de ses bâtiments non destinés à l'habitation.\n
    Mon étude se basera sur les données structurelles des bâtiments (taille et usage des bâtiments, date de construction, situation géographique, ...) pour proposer un model de machine learning performant.

    1. Analyse exploratoire.

        L'analyse exploratoire visera à identifier les tendances et les corrélations entre les variables structurelles des bâtiments et leurs émissions de CO2. Elle permettra également de repérer les valeurs aberrantes et de comprendre leur impact potentiel sur le modèle.

    2. Modèles de prédiction.

        Nous testerons plusieurs algorithmes de machine learning (ElasticNet, SVM, GradientBoosting, RandomForest), chacun ayant des forces et des faiblesses spécifiques. Par exemple, ElasticNet est utile pour les données avec des variables fortement corrélées, tandis que GradientBoosting peut capturer des relations non linéaires complexes.

    3. Traitement des variables.

        Le traitement des variables inclura la normalisation pour les algorithmes sensibles à l'échelle, ainsi que des transformations logarithmiques pour les variables avec une distribution asymétrique. Nous explorerons également l'ingénierie des caractéristiques pour extraire des informations pertinentes, comme la nature des sources d'énergie utilisées.

    4. Evaluation des performances.

        L'évaluation des performances se fera à l'aide de métriques telles que le RMSE (Root Mean Squared Error) et le R² (coefficient de détermination), qui sont pertinents pour les problèmes de régression. Nous utiliserons la validation croisée pour assurer la robustesse des résultats.

    5. Intégration de l'Energy Star Score.

        L'Energy Star Score sera intégré comme une variable supplémentaire dans nos modèles. Nous évaluerons son impact sur les performances prédictives et déterminerons s'il apporte une valeur ajoutée significative par rapport aux autres variables structurelles.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    # Exploration du jeu de données
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Première lecture du jeu de données

    ### Téléchargement des données

    -   Des relevés minutieux ont été effectués par les agents de la ville en 2016. Voici les [données](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P4/2016_Building_Energy_Benchmarking.csv) et leur [source](https://data.seattle.gov/Built-Environment/Building-Energy-Benchmarking-Data-2015-Present/teqw-tu6e/about_data).
    """
    )
    return


@app.cell(hide_code=True)
def _():
    ### Initialisation et importation des librairies
    return


@app.cell
def _(mo):
    with mo.status.spinner("Importing libraries..."):
        from pathlib import Path

        from itables import init_notebook_mode, show

        init_notebook_mode(all_interactive=True)
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.io as pio
        import seaborn as sns

        from IPython.display import Markdown
        from itables.widget import ITable
        from matplotlib import gridspec
        from ydata_profiling import ProfileReport

        from scripts import analysis_tools as tools

        tools.set_options()

        csv_file: Path = Path("data/raw/2016_Building_Energy_Benchmarking.csv")
    return (
        ITable,
        ProfileReport,
        csv_file,
        gridspec,
        np,
        pd,
        plt,
        px,
        sns,
        tools,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Importation du jeu de données

    Je commence par charger le fichier de données.
    """
    )
    return


@app.cell
def _(csv_file, pd):
    full_data = pd.read_csv(csv_file, sep=",")
    return (full_data,)


@app.cell
def _(full_data):
    full_data.sample(frac=0.25, random_state=42)
    return


@app.cell
def _(full_data):
    full_data.describe()
    return


@app.cell
def _(full_data):
    full_data.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Structure et contenu du jeu de données

    Une description de chaque caractéristiques est fourni à cette adresse: [about data](https://data.seattle.gov/Built-Environment/Building-Energy-Benchmarking-Data-2015-Present/teqw-tu6e/about_data)

    ### ydata-profiling

    Je commence par afficher un rapport du jeu de données avec l'outils ydata-profiling.
    Ainsi j'aurais rapidement un aperçu clair et détaillé de son contenu.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Il y a 46 caractéristiques décrivant les propriétés (géographiques, architecturales, usage, consommations, émissions).

    Je conserve les caractéristiques suivantes:

    -   Géographiques:

        -   `Latitude`

        -   `Longitude`

        -   `CouncilDistrictCode`

        -   `Neighborhood` (Quartier)

    -   Architecturales:

        -   `YearBuilt` (Année de construction)
        -   `NumberofBuildings` (Nombre de bâtiments sur la propriété)
        -   `NumberofFloors` (Nombre d'étages de la propriété)
        -   `PropertyGFATotal` (Surface Totale)
        -   `PropertyGFABuildings` (Surface des bâtiment de la propriété)
        -   `PropertyGFAParking` (Surface parking)
        -   `OSEBuildingID` (identifiant unique)

    -   Usage

        -   `PrimaryPropertyType` (Type de propriété principale)
        -   `BuildingType`

    -   Emissions/Conso

        -   `TotalGHGEmissions`

        -   `SiteEnergyUseWN(kBtu)`

        -   `SteamUse(kBtu)`

        -   `Electricity(kBtu)`

        -   `NaturalGas(kBtu)`

        -   `ENERGYSTARScore`

    -   Commentaires:

        -   `DefaultData`

        -   `Comments`

        -   `ComplianceStatus`

        -   `Outlier`

    | Column Name | Description |
    |-------------------------|-----------------------------------------------|
    | ENERGYSTARScore | An EPA calculated 1-100 rating that assesses a property’s overall energy performance, based on national data to control for differences among climate, building uses, and operations. A score of 50 represents the national median.Read more |
    | EPAPropertyType | The primary use of a property (e.g. office, retail store). Primary use is defined as a function that accounts for more than 50% of a property. This is the Property Type - EPA Calculated field from Portfolio Manager.Read more |
    | GHGEmissionsIntensity | Total Greenhouse Gas Emissions divided by property's gross floor area, measured in kilograms of carbon dioxide equivalent per square foot. |
    | Latitude | Property latitude |
    | Longitude | Property longitude |
    | Neighborhood | Property neighborhood area defined by the City of Seattle Department of Neighborhoods. |
    | NumberofBuilding(s) | Number of buildings as part of each property. |
    | NumberofFloors | Number of occupiable floors at or above grade level. n |
    | OSEBuildingID | A unique identifier assigned to each property covered by the Seattle Benchmarking Ordinance for tracking and identification purposes. |
    | PropertyGFATotal | Total building and parking gross floor area. Figure reflects Seattle OSE's confirmed property gross floor area as reported in public records like King County Assessor Data, or otherwise verified.Read more |
    | PropertyGFABuildings | Total floor space in square feet between the outside surfaces of a building’s enclosing walls. This includes all areas inside the building(s), such as tenant space, common areas, stairwells, basements, storage, etc... but does not include parking. Figure reflects Seattle OSE's confirmed property gross floor area as reported in public records like King County Assessor Data, or otherwise verified. |
    | PropertyGFAParking | Total space in square feet of all types of parking (Fully Enclosed, Partially Enclosed, and Open). Figure reflects Seattle OSE's confirmed property gross floor area as reported in public records like King County Assessor Data, or otherwise verified. |
    | SiteEnergyUseWN(kBtu) | The annual amount of energy consumed by the property from all sources of energy, adjusted to what the property would have consumed during 30-year average weather conditions.Read more |
    | SteamUse(kBtu) | The annual amount of district steam consumed by the property on-site, measured in thousands of British thermal units (kBtu). |
    | Electricity(kBtu) | The annual amount of electricity consumed by the property on-site, including electricity purchased from the grid and generated by onsite renewable systems, measured in thousands of British thermal units (kBtu). |
    | NaturalGas(kBtu) | The annual amount of utility-supplied natural gas consumed by the property, measured in thousands of British thermal units (kBtu). |
    | TotalGHGEmissions | The total amount of greenhouse gas emissions, including carbon dioxide, methane, and nitrous oxide gases released into the atmosphere as a result of energy consumption at the property, measured in metric tons of carbon dioxide equivalent. This calculation employs the utility-specific emissions factors defined in the 2023 Building Emissions Performance Standard law. |
    | YearBuilt | Year in which a property was constructed. |
    | ComplianceStatus | Whether a property met energy benchmarking requirements for the specified reporting year, as of the end of that reporting year's enforcement grace period. |
    | ComplianceIssue | If known, what compliance issues were present for the specified reporting year at the end of that reporting year's enforcement grace period. |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Je recharge le jeu de données avec uniquement les colonnes sélectionnées.""")
    return


@app.cell(hide_code=True)
def _():
    ## Exploration
    return


@app.cell
def _():
    col_geo = [
        "CouncilDistrictCode",
        "Neighborhood",
        "Latitude",
        "Longitude",
    ]
    col_archi = [
        "YearBuilt",
        "NumberofBuildings",
        "NumberofFloors",
        "PropertyGFATotal",
        "PropertyGFABuilding(s)",
        "PropertyGFAParking",
        "OSEBuildingID",
    ]

    col_usage = [
        "PrimaryPropertyType",
        "BuildingType",
    ]

    col_conso = [
        "SiteEnergyUseWN(kBtu)",
        "TotalGHGEmissions",
        "SteamUse(kBtu)",
        "Electricity(kBtu)",
        "NaturalGas(kBtu)",
        "ENERGYSTARScore",
    ]

    col_commentaires = [
        "DefaultData",
        "Comments",
        "ComplianceStatus",
        "Outlier",
    ]
    col_conservees = col_geo + col_archi + col_usage + col_conso + col_commentaires
    return (col_conservees,)


@app.cell(hide_code=True)
def _(col_conservees, csv_file, pd):
    data = pd.read_csv(csv_file, sep=",", usecols=col_conservees)
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Propriétés en doublons

    Dans un premier temps, je m'assure que je n'ai pas de propriétés en doublons.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    propriete_doublons = data["OSEBuildingID"].duplicated(keep=False).sum()
    Markdown(f"Nombre total de doublons (toutes occurrences) : **{propriete_doublons}**")
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Il n'y a pas de propriété en doublons.

    Je peux supprimer la colonne indicateur.
    """
    )
    return


@app.cell
def _(data):
    data_1 = data.drop("OSEBuildingID", axis=1)
    return (data_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Usage de la propriété

    Je commence par étudier uniquement les caractéristiques d'usages.
    """
    )
    return


@app.cell(hide_code=True)
def _(ProfileReport, data_1):
    profile_1 = ProfileReport(
        data_1[["BuildingType", "PrimaryPropertyType"]],
        title="Profiling Report - Usages des propriétés",
        correlations=None,
        samples=None,
    )
    profile_1.to_notebook_iframe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""`BuildingType` définie le type de propriété alors que `PrimaryPropertyType` définie son usage principale."""
    )
    return


@app.cell(hide_code=True)
def _(ITable, data_1):
    # ITable(data_1["BuildingType"].unique(), "Mots dans BuildingType")
    data_1["BuildingType"].unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Je vais conserver uniquement les types de propriétés suivants et les renommer en `NonResidential`:

    -   `NonResidential`

    -   `Nonresidential COS` (Non résidentiel construction)

    -   `Nonresidential WA` (Non résidentiel Washington)
    """
    )
    return


@app.cell
def _(data_1):
    buildingtype_map = {
        "NonResidential": "NonResidential",
        "Nonresidential COS": "NonResidential",
        "Nonresidential WA": "NonResidential",
    }
    data_1["new_bldtp"] = data_1["BuildingType"].map(buildingtype_map)
    filter = "NonResidential"
    data_2 = data_1.query(f"`new_bldtp` == '{filter}'")
    return (data_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Je peux supprimer la colonne `BuildingType` et `new_bldtp`.""")
    return


@app.cell
def _(data_2):
    data_3 = data_2.drop("BuildingType", axis=1)
    data_3 = data_3.drop("new_bldtp", axis=1)
    return (data_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Type des caractéristiques

    Je vérifie que le type des colonnes sont correctement détectés par pandas.
    """
    )
    return


@app.cell(hide_code=True)
def _(ITable, data_3):
    # ITable(data_3.dtypes)
    data_3.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""J'essaie la conversion automatique de pandas""")
    return


@app.cell
def _(data_3):
    data_4 = data_3.convert_dtypes()
    return (data_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Pour faciliter la modélisation des données je convertie certaines caractéristiques en variables catégorielles."""
    )
    return


@app.cell
def _(data_4):
    data_4["CouncilDistrictCode"] = data_4["CouncilDistrictCode"].astype("category")
    data_4["PrimaryPropertyType"] = data_4["PrimaryPropertyType"].astype("category")
    data_4["Neighborhood"] = data_4["Neighborhood"].astype("category")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Caractéristiques numériques

    J'analyse les variables numériques (`YearBuilt`,`SiteEnergyUseWN(kBtu)`) pour identifier les valeurs manquantes, outliers, incohérences.
    """
    )
    return


@app.cell(hide_code=True)
def _(ProfileReport, data_4):
    profile_2 = ProfileReport(
        data_4[["YearBuilt", "SiteEnergyUseWN(kBtu)", "NumberofFloors", "NumberofBuildings", "PropertyGFATotal"]],
        title="Profiling Report - Usages des propriétés",
        samples=None,
    )
    profile_2.to_notebook_iframe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Je supprime les résidences avec une consommation énergétique nulle et les propriétés sans bâtiments.""")
    return


@app.cell(hide_code=True)
def _(data_4):
    data_5 = data_4.query("`SiteEnergyUseWN(kBtu)` > 0")
    data_5 = data_5.query("`NumberofBuildings` > 0")
    return (data_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Le jeu de données est fourni avec des champs de commentaires pour avertir sur la qualité des données ou sur la présence d'outlier."""
    )
    return


@app.cell(hide_code=True)
def _(ProfileReport, data_5):
    profile_3 = ProfileReport(
        data_5[["DefaultData", "Comments", "ComplianceStatus", "Outlier"]],
        title="Profiling Report - Usages des propriétés",
        samples=None,
    )
    profile_3.to_notebook_iframe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Je supprime les propriétés où la colonne `Outlier` contient "High outlier" ou "Low outlier".""")
    return


@app.cell
def _(data_5):
    OUTLIERS = ["High outlier", "Low outlier"]
    pattern = "|".join(OUTLIERS)
    data_6 = data_5[~data_5["Outlier"].str.contains(pattern, case=False, na=False)]
    return (data_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Voyons voir maintenant la variable `ComplianceStatus`. Je conserver que les propriétés indiquée comme "Compliant"
    """
    )
    return


@app.cell
def _(data_6):
    data_7 = data_6[data_6["ComplianceStatus"].str.contains("Compliant")]
    return (data_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Je peux supprimer les colonnes commentaires qui ne me sont plus utiles.""")
    return


@app.cell
def _(data_7):
    data_8 = data_7.drop(columns=["DefaultData", "Comments", "ComplianceStatus", "Outlier"])
    return (data_8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Valeurs manquantes""")
    return


@app.cell
def _(px):
    def proportion_valeurs_manquantes(data):
        """Calcule la proportion de valeurs manquantes par colonne."""
        return data.isna().mean() * 100

    def visualiser_valeurs_manquantes(data, threshold=None):
        """Visualise la proportion de valeurs manquantes par colonne avec un graphique à barres interactif.

        Paramètres :
        - data : DataFrame pandas contenant les données à analyser.
        - threshold : Seuil pour la ligne de référence sur le graphique (optionnel).

        Retourne :
        - fig : Objet Plotly Figure représentant le graphique.
        """
        valeurs_manquantes_data = proportion_valeurs_manquantes(data).reset_index()
        valeurs_manquantes_data.columns = ["Colonne", "Proportion"]

        # Création du graphique à barres avec Plotly
        fig = px.bar(
            valeurs_manquantes_data,
            x="Colonne",
            y="Proportion",
            title="Proportions de valeurs manquantes par colonne",
            labels={"Proportion": "Proportion de valeurs manquantes (%)"},
            color="Proportion",
            color_continuous_scale=["#66c2a5", "#fc8d62"],
            range_color=[0, 100],  # Fixe l'échelle de couleurs de 0 à 100
        )

        fig.update_layout(
            xaxis_title="Colonnes",
            yaxis_title="Proportion de valeurs manquantes (%)",
            xaxis_tickangle=-45,
            width=800,
            height=500,
        )

        # Ajout d'une ligne de seuil si spécifiée
        if threshold is not None:
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(valeurs_manquantes_data) - 0.5,
                y0=threshold,
                y1=threshold,
                line={"color": "black", "dash": "dash"},
            )
            fig.add_annotation(
                x=len(valeurs_manquantes_data) - 0.5,
                y=threshold,
                text=f"Manquants < {threshold}%",
                showarrow=False,
                align="right",
                xanchor="right",
            )

        fig.show()

    return (visualiser_valeurs_manquantes,)


@app.cell
def _(data_8, visualiser_valeurs_manquantes):
    visualiser_valeurs_manquantes(data_8)
    return


@app.cell(hide_code=True)
def _(ITable, data_8):
    # ITable(data_8.isna().sum(), "Nb valeurs manquantes par colonne")
    data_8.isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Il n'y a pas de valeurs manquantes dans le jeu de données sauf pour `EnergyStartScore` que j'utiliserai plus tard.

    ### Valeurs aberrantes

    La variable property GFA total décrit la superficie de la propriété. Je vérifie s'il n'y a pas de valeurs négatives.
    """
    )
    return


@app.cell(hide_code=True)
def _(data_8):
    data_8.describe(include="all")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    La colonne `TotalGHGEmissions` contient une valeur minimale de -0.8. Il s'agit d'une somme des émissions rejetées sans tenir compte de l'auto-consommation qui donnerait une valeur négative.

    Je supprime les propriétés avec des émissions négatives.

    On peut voir que certaines propriétés utilisent de l'électricité auto produite (`Electricity(kBtu)` négative).
    """
    )
    return


@app.cell
def _(data_8):
    data_9 = data_8.query("TotalGHGEmissions > 0")
    return (data_9,)


@app.cell(hide_code=True)
def _(ProfileReport, data_9):
    profile_4 = ProfileReport(data_9, title="Profiling Report", minimal=True)
    profile_4.to_notebook_iframe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Surface

    Les variables PropertyGFAParking et PropertyGFABuilding(s) représentent une partie de la superficie, alors que PropertyGFATotal, représente la superficie total. Par conséquent, la somme 2 premières variables ne peut pas être supérieures à la variable total.
    """
    )
    return


@app.cell(hide_code=True)
def _(data_9):
    filtered_data = data_9.query("PropertyGFAParking + `PropertyGFABuilding(s)` > PropertyGFATotal")
    filtered_data.shape[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Aucune propriété n'a la somme de ces surfaces supérieurs à la superficie totale renseignée.

    Voyons maintenant si une des superficie est supérieures à la superficie totale renseignée.
    """
    )
    return


@app.cell(hide_code=True)
def _(data_9):
    filtered_data_1 = data_9.query(
        "PropertyGFAParking > PropertyGFATotal | `PropertyGFABuilding(s)` > PropertyGFATotal"
    )
    filtered_data_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Aucune superficie Parking ou d'habitation n'est supérieure à la superficie totale renseignée.

    ### Feature engineering

    Caractérisation des mix énergétiques. Je calcule les proportions des types d'énergies utilisés.

    Je définie les colonnes qui caractérisent le type d'énergie et la consommation d'énergie totale.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    cols_conso = [
        "SteamUse(kBtu)",
        "Electricity(kBtu)",
        "NaturalGas(kBtu)",
    ]

    col_energy_tot = "SiteEnergyUseWN(kBtu)"
    return col_energy_tot, cols_conso


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Je crée de nouvelles colonnes pour représenter les proportions d'énergies utilisées par bâtiment. Ainsi je pourrai comparer les bâtiment plus facilement."""
    )
    return


@app.cell
def _(col_energy_tot, cols_conso, data_9):
    for col in cols_conso:
        data_9[f"{col}_Proportion"] = data_9[col] / data_9[col_energy_tot]
    return


@app.function
# Déterminer la source d'énergie principale pour chaque bâtiment
def determine_primary_energy_source(row):
    if row["SteamUse(kBtu)_Proportion"] == max(row):
        return "Steam"
    elif row["Electricity(kBtu)_Proportion"] == max(row):
        return "Electricity"
    elif row["NaturalGas(kBtu)_Proportion"] == max(row):
        return "Gas"


@app.cell
def _(data_9):
    cols_ratio_conso = ["SteamUse(kBtu)_Proportion", "Electricity(kBtu)_Proportion", "NaturalGas(kBtu)_Proportion"]
    data_9["PrimaryEnergySource"] = data_9[cols_ratio_conso].apply(determine_primary_energy_source, axis=1)
    return (cols_ratio_conso,)


@app.cell(hide_code=True)
def _(data_9, px):
    energy_counts = data_9["PrimaryEnergySource"].value_counts()
    fig = px.pie(
        values=energy_counts.values, names=energy_counts.index, title="Distribution des sources d'énergies utilisées"
    )
    fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Les bâtiments non résidentiels utilise majoritairement de l'électricité (75%).

    Je supprime les features de mesure d'énergie pour ne conserver plus que `PrimaryEnergySource`.
    """
    )
    return


@app.cell
def _(cols_conso, data_9):
    data_10 = data_9.drop(cols_conso, axis=1)
    return (data_10,)


@app.cell
def _(cols_ratio_conso, data_10):
    data_11 = data_10.drop(cols_ratio_conso, axis=1)
    return (data_11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Année de construction

    Je peux regrouper les bâtiments en fonction de leurs années de construction. Les années de construction.
    """
    )
    return


@app.cell(hide_code=True)
def _(data_11):
    data_11["YearBuilt"].plot.hist(bins=50)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Les années de construction des bâtiments sont distribuées entre 1900 et 2016.

    Je groupes les bâtiments par tranche de 50 ans.
    """
    )
    return


@app.cell
def _(data_11, pd):
    bins = [float("-inf"), 1950, 2000, float("inf")]
    labels = ["Avant 1950", "1950-2000", "Après 2000"]
    data_11["AgeGroup"] = pd.cut(data_11["YearBuilt"], bins=bins, labels=labels)
    return


@app.cell(hide_code=True)
def _(data_11, px):
    age_group_counts = data_11["AgeGroup"].value_counts()
    fig_1 = px.pie(
        values=age_group_counts.values,
        names=age_group_counts.index,
        title="Distribution des Groupes d'Années de Construction des Bâtiments",
        labels={"names": "Groupe d'Années"},
        color=age_group_counts.index,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_1.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Il y majoritairement des bâtiments construits entre 1950 et 2000.

    Je supprime la colonne `YearBuilt` pour ne conserver que `AgeGroup`.
    """
    )
    return


@app.cell(hide_code=True)
def _(data_11):
    data_12 = data_11.drop("YearBuilt", axis=1)
    return (data_12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Géographie

    #### Neighborhood

    Voyons la répartition des résidences dans les quartiers de Seattle.
    """
    )
    return


@app.cell(hide_code=True)
def _(ITable, data_12):
    # ITable(data_12["Neighborhood"].groupby(data_12["Neighborhood"], observed=True).count().sort_values())
    data_12["Neighborhood"].groupby(data_12["Neighborhood"], observed=True).count().sort_values()
    return


@app.cell
def _(mo):
    mo.md(r"""Il y des différences de casses pour les différents quartiers. Je les normalise.""")
    return


@app.cell(hide_code=True)
def _(data_12):
    data_12["Neighborhood"] = data_12["Neighborhood"].str.upper()
    return


@app.cell
def _(mo):
    mo.md(r"""Je remplace de `DELRIDGE NEIGHBORHOODS` par `DELRIDGE`""")
    return


@app.cell(hide_code=True)
def _(data_12):
    data_12["Neighborhood"] = data_12["Neighborhood"].replace("DELRIDGE NEIGHBORHOODS", "DELRIDGE")
    return


@app.cell
def _(mo):
    mo.md(r"""Je vérifie à nouveaux la répartition des bâtiments.""")
    return


@app.cell(hide_code=True)
def _(ITable, data_12):
    # ITable(data_12["Neighborhood"].groupby(data_12["Neighborhood"], observed=True).count().sort_values())
    data_12["Neighborhood"].groupby(data_12["Neighborhood"], observed=True).count().sort_values()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### District code

    Répartition des bâtiments selon les codes administratifs des quartiers de Seattle.
    """
    )
    return


@app.cell(hide_code=True)
def _(ITable, data_12):
    # ITable(data_12["CouncilDistrictCode"].groupby(data_12["CouncilDistrictCode"], observed=True).count().sort_values())
    data_12["CouncilDistrictCode"].groupby(data_12["CouncilDistrictCode"], observed=True).count().sort_values()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    La répartition est différentes de Neighborhoods. C'est intéressant de la conserver pour peut être affiner le modèle.

    #### Latitude-Longitude

    Répartition des bâtiments selon leurs coordonnées GPS.
    """
    )
    return


@app.cell
def _(data_12, px):
    fig_2 = px.histogram(data_12, "Latitude", title="Distribution des Latitudes")
    fig_2.show()
    return


@app.cell(hide_code=True)
def _(data_12, px):
    fig_3 = px.histogram(data_12, "Longitude", title="Distribution des Longitudes")
    fig_3.show()
    return


@app.cell
def _(mo):
    mo.md(r"""Visualisation des bâtiments en fonction de leurs quartiers (`PrimaryEnergySource`).""")
    return


@app.cell(hide_code=True)
def _(data_12, px):
    data_13 = data_12.sort_values(by="PrimaryEnergySource")
    fig_4 = px.scatter_map(
        data_13,
        lat="Latitude",
        lon="Longitude",
        color="PrimaryEnergySource",
        center={"lat": data_13["Latitude"].mean(), "lon": data_13["Longitude"].mean()},
        zoom=9,
        height=500,
        map_style="carto-positron",
        title="Répartition des bâtiments par Energy de Seattle",
    )
    fig_4.show()
    return (data_13,)


@app.cell
def _(mo):
    mo.md(r"""Les bâtiments qui utilisent de l'énergie issue de la vapeur se trouvent en centre ville.""")
    return


@app.cell(hide_code=True)
def _(ITable, data_13):
    # ITable(data_13.query("PrimaryEnergySource == 'Steam'"))
    data_13.query("PrimaryEnergySource == 'Steam'")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Difficile de déterminer l'usage de ces bâtiments...

    Je visualise la carte également avec la période de construction des bâtiments.

    Visualisation des bâtiments en fonction de leurs quartiers (`AgeGroup`).
    """
    )
    return


@app.cell(hide_code=True)
def _(data_13, px):
    data_14 = data_13.sort_values(by="AgeGroup")
    fig_5 = px.scatter_map(
        data_14,
        lat="Latitude",
        lon="Longitude",
        color="AgeGroup",
        center={"lat": data_14["Latitude"].mean(), "lon": data_14["Longitude"].mean()},
        zoom=9,
        height=500,
        map_style="carto-positron",
        title="Répartition des bâtiments par année de construction à Seattle",
    )
    fig_5.show()
    return (data_14,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""La majorité des bâtiments ont été construit entre 1950 et 2000.""")
    return


@app.cell
def _(data_14, px):
    age_group_counts_1 = data_14["AgeGroup"].value_counts(normalize=True).round(4) * 100
    age_group_counts_1 = age_group_counts_1.reset_index()
    age_group_counts_1.columns = ["AgeGroup", "pourcentage"]
    fig_6 = px.bar(
        age_group_counts_1,
        x="AgeGroup",
        y="pourcentage",
        color="AgeGroup",
        text_auto=".2f",
        title="Pourcentage de bâtiments par années de construction",
    ).update_xaxes(categoryorder="total descending")
    fig_6.show()
    return


@app.cell(hide_code=True)
def _(data_14, px):
    fig_7 = px.density_map(
        data_14,
        lat="Latitude",
        lon="Longitude",
        z="NumberofBuildings",
        center={"lat": data_14["Latitude"].mean(), "lon": data_14["Longitude"].mean()},
        zoom=9,
        height=500,
        map_style="carto-positron",
        title="Densité des bâtiments par propriété à Seattle",
    )
    fig_7.show()
    return


@app.cell
def _(data_14):
    data_14.columns
    return


@app.cell
def _(data_14, px):
    fig_86 = px.density_map(
        data_14,
        lat="Latitude",
        lon="Longitude",
        z="PropertyGFABuilding(s)",
        center={"lat": data_14["Latitude"].mean(), "lon": data_14["Longitude"].mean()},
        zoom=9,
        height=500,
        map_style="carto-positron",
        title="Surface des bâtiments par propriété à Seattle",
    )
    fig_86.show()
    return


@app.cell
def _(data_14, sns):
    sns.kdeplot(data_14[["PropertyGFABuilding(s)"]], fill=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Architecture

    Création d'une colonne montrant le rapport entre la taille du parking et la taille totale.
    """
    )
    return


@app.cell
def _(data_14):
    data_14["Parking_ratio"] = (data_14["PropertyGFAParking"] / data_14["PropertyGFATotal"]).round(3)
    return


@app.cell(hide_code=True)
def _(data_14):
    data_15 = data_14.drop("PropertyGFAParking", axis=1)
    return (data_15,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Création d'une colonne montrant le rapport entre la taille d'usage majoritaire et la taille totale.""")
    return


@app.cell
def _(data_15):
    data_15["Building_ratio"] = (data_15["PropertyGFABuilding(s)"] / data_15["PropertyGFATotal"]).round(3)
    return


@app.cell(hide_code=True)
def _(data_15):
    data_16 = data_15.drop("PropertyGFABuilding(s)", axis=1)
    return (data_16,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Distributions et corrélations

    Je visualise la distribution des variables du jeu de données.
    """
    )
    return


@app.cell(hide_code=True)
def _(data_16, np, px):
    df_corr = data_16.select_dtypes(include="number").corr(method="spearman").round(2)
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    df_corr_viz = df_corr.mask(mask).dropna(how="all").dropna(axis="columns", how="all")
    fig_8 = px.imshow(
        df_corr_viz,
        text_auto=True,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title="Matrice de correlation suivant le coefficient de spearman",
    )
    fig_8.update_xaxes(showgrid=False).update_yaxes(showgrid=False)
    fig_8.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""`Parking_ratio` et `Building_ration` ont une corrélation négative parfaite de -1. Cela signifie que l'information est redondante. Afin d'éviter le bruit lors de l’entraînement je n'en conserve que une `Parking_ratio`."""
    )
    return


@app.cell(hide_code=True)
def _(data_16):
    data_17 = data_16.drop("Building_ratio", axis=1)
    return (data_17,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Voyons la distribution des colonnes pour déterminer si des transformations sont nécessaires.""")
    return


@app.cell(hide_code=True)
def _(data_17, px):
    for column in data_17.select_dtypes(include="number"):
        fig_9 = px.histogram(data_17, x=column, title=f"Distribution de {column}", marginal="box")
        fig_9.show()
    return


@app.cell
def _(data_16, data_17, gridspec, plt, sns):
    plt.figure(figsize=(12, 5 * 5))
    _gs = gridspec.GridSpec(8, 5, hspace=0.7)

    for _i, _col in enumerate(data_17.select_dtypes(include="number")):
        _ax = plt.subplot(_gs[_i])
        sns.kdeplot(data_16[_col], color="blue", label=_col, fill=True)
        _ax.set_xlabel("")
        _ax.set_title(_col)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Plusieurs caractéristiques numériques ont une distribution asymétrique vers la droite. (`PropertyGFATotal`, `SiteEnergyUseWN(kBtu)`,`TotalGHGEmissions` et `Parking_ratio`)
    J'applique une transformation logarithmique pour centrer ces features.
    """
    )
    return


@app.cell(hide_code=True)
def _(data_17, np):
    log_columns = ["PropertyGFATotal", "SiteEnergyUseWN(kBtu)", "TotalGHGEmissions", "Parking_ratio"]
    for column_1 in log_columns:
        data_17[column_1] = np.log1p(data_17[column_1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Voyons à nouveau la distribution des colonnes numériques.""")
    return


@app.cell
def _(data_17, px):
    for column_2 in data_17.select_dtypes(include="number"):
        fig_10 = px.histogram(data_17, x=column_2, title=f"Distribution de {column_2}", marginal="box")
        fig_10.show()
    return


@app.cell
def _(data_17, gridspec, plt, sns):
    plt.figure(figsize=(12, 5 * 5))
    _gs = gridspec.GridSpec(8, 5, hspace=0.7)

    for _i, _col in enumerate(data_17.select_dtypes(include="number")):
        _ax = plt.subplot(_gs[_i])
        sns.kdeplot(data_17[_col], color="green", label=_col, fill=True)
        _ax.set_xlabel("")
        _ax.set_title(_col)
    plt.show()
    return


@app.cell
def _(data_17, tools):
    tools.save_intermediate_data(data_17, "feature_eng")
    return


if __name__ == "__main__":
    app.run()
