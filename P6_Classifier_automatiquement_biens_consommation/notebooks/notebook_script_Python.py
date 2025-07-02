import marimo

__generated_with = "0.14.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Script python pour tester API""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Dans ce notebook, je me concentre sur la réalisation du test de collecte de produit à base de champagne via l'API de edamam accessible par rapidapi  :

    > "Nous souhaitons élargir notre gamme de produits, en particulier dans l’épicerie fine.
    > Pourrais-tu tester la collecte de produits à base de “**champagne**” via l’API disponible ici ?
    > Je souhaiterais que tu puisses nous fournir une extraction des 10 premiers produits dans un fichier “.csv”, contenant pour chaque produit les données suivantes : **foodId**, **label**, **category**, **foodContentsLabel**, **imag**".

    - API = https://rapidapi.com/edamam/api/edamam-food-and-grocery-database
    - utilisation de variable d'environnement pour ne pas partager ma clée d'accès à l'API

    /// admonition | Editer pyproject.toml

        [tool.marimo.runtime]
        dotenv = [".env"]

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Import des librairies""")
    return


@app.cell
def _():
    import requests

    import pandas as pd

    import os
    return os, pd, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Script""")
    return


@app.cell
def _(os, requests):
    url = "https://edamam-food-and-grocery-database.p.rapidapi.com/api/food-database/v2/parser"

    # Paramètre pour rechercher "champagne"
    querystring = {"ingr": "champagne"}

    api_key = os.environ["X-RapidAPI-Key"]

    # En-têtes HTTP avec la clé et l'hôte RapidAPI
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "edamam-food-and-grocery-database.p.rapidapi.com"
    }

    # Requête
    response = requests.get(url, headers=headers, params=querystring)
    return (response,)


@app.cell
def _(response):
    response.json()
    return


@app.cell
def _(pd, response):
    # Après ton traitement des données
    if response.status_code == 200:
        data = response.json()
        filtered_data = []

        if "hints" in data:
            for item in data["hints"]:
                food = item.get("food", {})
                relevant_info = {
                    "foodId": food.get("foodId"),
                    "label": food.get("label"),
                    "category": food.get("category"),
                    "foodContentsLabel": food.get("foodContentsLabel"),
                    "image": food.get("image")
                }
                filtered_data.append(relevant_info)

        # Création du DataFrame pandas
        df = pd.DataFrame(filtered_data)
    else:
        print("Erreur API :", response.status_code, response.text)

    return (df,)


@app.cell
def _(df):
    # Affichage du DataFrame
    df.head(10)
    return


@app.cell
def _(df):
    df.head(10).to_csv("data/processed/top_10_products_infos.csv")
    return


if __name__ == "__main__":
    app.run()
