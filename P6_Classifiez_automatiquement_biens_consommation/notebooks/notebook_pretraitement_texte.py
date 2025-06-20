import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Notebook de prétaitement des valeurs textuelles""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("notebooks/public/Projet_textimage_logo.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Contexte

    Vous êtes Data Scientist au sein de l’entreprise "Place de marché”, qui souhaite lancer une marketplace e-commerce.

    - La place de marché est anglophone ➡️ Photo et description des articles en anglais.
    - Objectif: automatiser la catégorisation des articles dans un but de fiabilité et d'amélioration de l'expérience utilisateur.

    Pour automatiser la catégorisation d'un produit, je vais devoir analyser séparement sa description et le contenu de l'image. 

    Bien entendu je ne vais pas povoir utiliser l'information brut, un prétratement est nécessaire.

    Je commence par prétraiter les descriptions des produits.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Définition

    - **NLP:** L'analyse du language humain par un ordinateur est appelé en inteligence artificiel la **NLP** (Natural Language Processing).

    - **NLTK:** Le NLTK, ou Natural Language Toolkit, est une suite de bibliothèques logicielles et de programmes. Elle est conçue pour le traitement naturel symbolique et statistique du langage anglais en langage Python.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Chargement des librairies et importation du jeu de données""")
    return


@app.cell
def _():
    # Text processing
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Télécharger les ressources NLTK
    nltk.download('punkt') # Utilisé pour la tokenization, c'est-à-dire diviser le texte en phrases ou en mots.
    nltk.download('stopwords') # Fournit une liste de mots courants (like "and", "the", etc.) souvent filtrés dans le traitement du texte.
    nltk.download('wordnet') # Utilisé pour la lemmatisation et pour accéder à des synonymes et des relations sémantiques entre les mots.

    # Graphiques
    import seaborn as sns
    sns.set()

    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv("data/raw/flipkart_com-ecommerce_sample_1050.csv")
    df
    return


if __name__ == "__main__":
    app.run()
