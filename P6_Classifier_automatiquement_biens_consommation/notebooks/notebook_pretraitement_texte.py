import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium", app_title="P6 prétraitement descriptions")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Notebook de prétraitement des valeurs textuelles""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("notebooks/public/Projet_textimage_logo.png").center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Contexte

    Vous êtes Data Scientist au sein de l’entreprise "Place de marché”, qui souhaite lancer une marketplace e-commerce.

    - La place de marché est anglophone ➡️ Photo et description des articles en anglais.
    - Objectif: automatiser la catégorisation des articles dans un but de fiabilité et d'amélioration de l'expérience utilisateur.

    Pour automatiser la catégorisation d'un produit, je vais devoir analyser séparément sa description et le contenu de son image.

    Bien entendu je ne vais pas pouvoir utiliser l'information brut, un prétraitement est nécessaire.

    Je commence par prétraiter les descriptions des produits.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Définition

    - **NLP:** L'analyse du language humain par un ordinateur est appelé en intelligence artificiel la **NLP** (Natural Language Processing).

    La NLP nécessite une approche méthodique pour analyser du language humain par un ordinateur:

    0. **Nettoyage des données**: Supprime les caractères spéciaux, les balises HTML, supprimes les majuscules, ...

    1. **Tokenization** : C'est la première étape où le texte est divisé en unités plus petites, comme des mots ou des phrases. NLTK fournit des fonctions pour faire cela, comme word_tokenize() pour les mots et sent_tokenize() pour les phrases.

    2. **Suppression des stopwords** : Les stopwords sont des mots fréquents qui n'apportent généralement pas beaucoup de sens, comme "and", "the". NLTK a des listes de stopwords pour les filtrer.

    3. **Stemming/Lemmatisation** : Ces processus réduisent les mots à leur forme de base.

        1. Le stemming réduit les mots à leurs racines.
        2. La lemmatisation utilise un vocabulaire et une analyse morphologique pour obtenir la forme de base des mots (running ➡️ run, happily ➡️ happy).
        3. Le stemming est plus rapide que la lemmatisation mais peut créer des mots qui n'existent pas (happily ❌ happili au lieu de happy).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Les librairies principalement pour faire de la NLP sont:

    1. **Transformers (Hugging Face)** :

    - Points forts : Offre une large gamme de modèles pré-entraînés de pointe et une grande flexibilité pour le fine-tuning.
    - Points faibles : Peut être complexe pour les débutants et nécessite souvent des ressources computationnelles importantes.

    2. **spaCy** :

    - Points forts : Efficace pour le traitement rapide et efficace du texte, avec un bon support pour le traitement de pipeline NLP.
    - Points faibles : Moins adapté pour les modèles de deep learning avancés comparé à d'autres bibliothèques.

    3. **NLTK (Natural Language Toolkit)** :

    - Points forts : Excellente pour l'enseignement et l'apprentissage des concepts de base du NLP grâce à sa simplicité et sa documentation complète.
    - Points faibles : Moins performante pour les tâches nécessitant des modèles avancés et à grande échelle.

    /// admonition | Info

        Étant débutant en NLP, je vais utiliser principalement **spacy** et **NLTK**

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Important
        type: warning

        Avec uv, le paquet de langue utilisé doit être installé avec la commande suivante. https://spacy.io/usage/models
        ```python
        uv add pip
        source .venv/bin/activate
        uv add en_core_web_sm $(spacy info en_core_web_sm --url)
        ```

    ///
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
    import matplotlib.pyplot as plt
    import nltk

    # Import spacy library
    import spacy

    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # wordcloud
    from wordcloud import WordCloud

    # Load language model
    nlp = spacy.load("en_core_web_sm")
    nltk.download('stopwords')

    # Word2Vec
    # tensorflow
    import os

    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import time

    # Graphiques
    import seaborn as sns
    import tensorflow as tf

    # USE
    import tensorflow_hub as hub

    # transformer huggingface
    import transformers
    from datasets import Dataset

    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    # sklearn
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.manifold import TSNE

    # Metriques
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import LabelEncoder
    from tf_keras.initializers import Constant
    from tf_keras.layers import Embedding, GlobalAveragePooling1D, Input
    from tf_keras.models import Model
    from tf_keras.preprocessing.sequence import pad_sequences
    from tf_keras.preprocessing.text import Tokenizer
    from transformers import AutoTokenizer, TFAutoModel

    sns.set_theme(style="white", palette="Set2")

    import re

    # Nettoyage du texte
    import string

    import numpy as np
    import pandas as pd

    from ydata_profiling import ProfileReport
    return (
        AutoTokenizer,
        Constant,
        CountVectorizer,
        Embedding,
        GlobalAveragePooling1D,
        Input,
        KMeans,
        LabelEncoder,
        Model,
        PCA,
        PorterStemmer,
        ProfileReport,
        SentenceTransformer,
        TFAutoModel,
        TSNE,
        TfidfVectorizer,
        Tokenizer,
        Word2Vec,
        WordCloud,
        adjusted_rand_score,
        hub,
        nlp,
        nltk,
        np,
        pad_sequences,
        pd,
        plt,
        re,
        simple_preprocess,
        sns,
        tf,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Téléchargement des ressources NLTK""")
    return


@app.cell
def _(nltk):
    # Télécharger les ressources NLTK
    nltk.download("punkt")  # Utilisé pour la tokenization, c'est-à-dire diviser le texte en phrases ou en mots.
    nltk.download(
        "stopwords"
    )  # Fournit une liste de mots courants (like "and", "the", etc.) souvent filtrés dans le traitement du texte.
    nltk.download(
        "wordnet"
    )  # Utilisé pour la lemmatisation et pour accéder à des synonymes et des relations sémantiques entre les mots.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Chargement du jeu de données""")
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/raw/flipkart_com-ecommerce_sample_1050.csv")
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Caractéristiques du jeu de données""")
    return


@app.cell
def _(df):
    df.count()
    return


@app.cell
def _(mo):
    button = mo.ui.run_button("warn", tooltip="Click to run expensive cells")
    button
    return (button,)


@app.cell
def _(ProfileReport, button, df, mo):
    # Explore datframe using ydata-profiling
    mo.stop(not button.value, "Click the button to continue")

    _profile = ProfileReport(
        df,
        title="Profiling Report",
        explorative=True,
    )
    _profile.to_file("notebooks/public/ydata_reports/Profile_report.html")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Pour ce projet, je vais utiliser que les colonnes **product_name**,**description** et **product_category_tree** du jeu de données.

    Il y a 1050 produits/description et aucun doublons.

    J' analyserai product_category_tree plus tard.
    """
    )
    return


@app.cell
def _(df, plt, sns):
    sns.barplot(df[["product_name", "description", "product_category_tree", "image"]].count())
    plt.tight_layout()
    plt.title("Nombre de valeurs par caractéristiques")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Colonne description""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Nettoyage des données""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Je commence par ajouter le product name à le description produit.

    Puis je passe à la partie nettoyage.

    1. mise en minuscule
    2. tokenisation
        - sépare tous les mots
        - supprime les stopwords
        - supprime la ponctuation
        - supprime les chiffres/nombres
        - supprime les mots/tokens inférieur à 2 caractères

    3. Regroupe les tokens par radicaux avec un Stemmer ou un Lemmer
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Fusion des colonnes product_name et description_produit

      ```python
      df["description"] = df["product_name"] + df["description"]
      ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Mise en minuscule

        ```pyhton
        # Convertit le texte en minuscules
        text.lower()
        ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Tokenisation c'est à liste chaque mot individuellement

    ```python
    def tokenize(text):
        doc = nlp(text)
        return [token.text for token in doc]
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Suppression des mots de liaisons ou stopwords en anglais
    - Suppression des nombres
    spacy contient une liste de mots de liaisons prédéfinis que je pourrais étendre si besoin.

    Pour les supprimer d'un document on utilise le code suivant

    ```python
    # suppression des mots de liaisons
    print([token.text for token in doc if not token.is_stop and token.is_alpha]])
    # ajout d'un mot de liaison à la liste
    nlp.Defaults.stop_words.add("ergo")
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Suppression de la ponctuation

        ```python
        import re

        text = re.sub(r"[^\w\s]", "", text)  # suppression ponctuation
        ```
    """
    )
    return


@app.cell
def _(nlp):
    # Print spacy's stop word list
    print(nlp.Defaults.stop_words)
    print(len(nlp.Defaults.stop_words))
    return


@app.cell
def _(df):
    # Ajoute le nom du produit à la description
    df["description"] = df["product_name"] + " " + df["description"]
    df["description"][0]
    return


@app.cell
def _(df, re):
    def preprocess_text(text):
        """Convert text to lowercase and remove punctuation."""
        MAX_LEN_WORDS = 2

        text = text.lower()  # Convert to lowercase

        # Remplacer les signes de ponctuation par des espaces
        text = re.sub(r"[^\w\s]", " ", text)

        # Supprimer les espaces excédentaires
        text = re.sub(r"\s+", " ", text)

        # Supprimer les mots de moins de 2 caractères
        text = " ".join(word for word in text.split() if len(word) > MAX_LEN_WORDS)

        # Supprimer les espaces en début et fin de chaîne
        text = text.strip()

        return text


    # Exemple d'utilisation
    _text = df["description"][0]
    _cleaned_text = preprocess_text(_text)
    print(_text)
    print(_cleaned_text)
    return (preprocess_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Stemmatisation ou Racinisation

    Consiste à réduire un mot à sa racine.

    Peut utiliser désormais car peut créer des mots qui n'existe pas.

    Spacy ne propose pas de méthode, il faut utiliser nltk.

    ```python
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    stemmer = PorterStemmer()

    def stem_text(text):
        doc = nlp(text)
        return [stemmer.stem(token.text) for token in doc]
    ```
    """
    )
    return


@app.cell
def _(PorterStemmer, df, nlp):
    def stem_text(text):
        """Apply Porter Stemming to text and filter out stop words and non-alphabetic tokens."""
        # Instanciate a stemmer from ntlk
        stemmer = PorterStemmer()
        doc = nlp(text)

        return [stemmer.stem(token.text) for token in doc if not token.is_stop and token.is_alpha]


    # Exemple d'utilisation
    _text = df["description"][0]
    stemed_text = stem_text(_text)
    print(_text)
    print(stemed_text)
    return (stem_text,)


@app.cell
def _(df, nlp):
    def lemmatize_text(text):
        """Lemmatize the input text and filter out stop words and non-alphabetic tokens."""
        doc = nlp(text.lower())
        return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]


    # Exemple d'utilisation
    _text = df["description"][0]
    lemmatized_text = lemmatize_text(_text)
    print(_text)
    print(lemmatized_text)
    return (lemmatize_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Fonction finale

    La fonction de preprocessing du texte prend un texte en entrée puis effectue la tokenisation selon le stemming ou le stemming et enfin les tokens de moins de 2 caractères sont supprimés eg: "cm".
    """
    )
    return


@app.cell
def _(df, lemmatize_text, preprocess_text, stem_text):
    def clean_text(text, lemmer=True, rejoin=True):
        """Perform text cleaning.

        1. lower_case, remove punctuation and word smaller than 2 letters
        2. extrat lem or stem from strings
        3. return cleaned text or a list of tokens
        """
        processed_text = preprocess_text(text)

        tokens = lemmatize_text(processed_text) if lemmer else stem_text(processed_text)

        if rejoin:
            return " ".join(tokens)
        return tokens


    # Exemple d'utilisation
    _text = df["description"][0]
    _cleaned_text = clean_text(df["description"][0], lemmer=True, rejoin=True)
    print(_text)
    print(_cleaned_text)
    return (clean_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""J'applique la fonction sur la totalité des produits.""")
    return


@app.cell
def _(clean_text, df):
    df["preprocessed_descr"] = df["description"].apply(clean_text, rejoin=False)
    df[["description", "preprocessed_descr"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Analyse des catégories""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Catégories de produits

    Voyons les catégories principales de chaque produits.
    """
    )
    return


@app.cell
def _(df):
    df["product_category_tree"].str.lower().value_counts().head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    L'arbre de catégorie est difficile à analyser.

    Je commence par généraliser en utilisant simplement la première catégorie de chaque produit.
    """
    )
    return


@app.cell
def _(df):
    # Split et conserve la première catégorie
    df["main_category"] = df["product_category_tree"].str.split(pat=" >>").str[0]

    # mis en minuscule
    df["main_category"] = df["main_category"].str.lower()

    # supprime ["
    df["main_category"] = df["main_category"].str.replace(pat='["', repl="")
    return


@app.cell
def _(df):
    df["main_category"].value_counts()
    return


@app.cell
def _(df):
    df["main_category"].count()
    return


@app.cell
def _(df, plt, sns):
    # Compter les valeurs de chaque catégorie
    counts = df["main_category"].value_counts()

    # Créer un graphique en secteurs avec Seaborn
    plt.figure(figsize=(6, 6))
    colors = sns.color_palette("pastel")[0 : len(counts)]
    plt.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct=lambda p: "{:.0f}".format(p * sum(counts) / 100),
        startangle=140,
    )

    plt.title("Nombre de produit par catégorie")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Chaque produit est assigné à une catégorie.

    il n'y a pas de valeurs manquantes et la répartition des produits par catégorie est parfaitement équilibré.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Voyons les mots qui sont le plus présents dans les descriptions de produit.""")
    return


@app.cell
def _(df, mo, plt, sns):
    df_exploded = df.explode("preprocessed_descr")[["main_category", "preprocessed_descr"]]

    # Compter la fréquence des mots
    word_frequencies = df_exploded["preprocessed_descr"].value_counts()

    # Sélectionner les 10 mots les plus fréquents
    top_10_words = word_frequencies.head(10)

    # Créer un DataFrame pour le tracé
    df_word_frequencies = top_10_words.reset_index()
    df_word_frequencies.columns = ["Word", "Frequency"]

    # Tracer les fréquences des mots
    _fig, _ax = plt.subplots(figsize=(5, 3))
    sns.barplot(x="Frequency", y="Word", data=df_word_frequencies, ax=_ax)
    _ax.set_title("Top 10 des mots les plus fréquents dans les descriptions de produits")
    _ax.set_xlabel("Fréquence")
    _ax.set_ylabel("Mots")
    mo.hstack([_fig, mo.ui.table(df_word_frequencies)])
    return (df_exploded,)


@app.cell(hide_code=True)
def _(df, mo):
    # Cell 2: Create a dropdown to select a category
    category_dropdown = mo.ui.dropdown(
        options=df["main_category"].unique().tolist(), value=df["main_category"].unique()[0], label="Select category"
    )

    category_dropdown
    return (category_dropdown,)


@app.cell(hide_code=True)
def _(WordCloud, category_dropdown, df, plt):
    # 1. Sélectionne la catégorie à analyser
    _selected_category = category_dropdown.value

    # 2. Filtrer le DataFrame sur la catégorie choisie
    _filtered_descr = df.query(f"main_category == '{category_dropdown.value}'")["preprocessed_descr"]

    # 3. Aplatir la liste de listes en une seule chaîne de texte
    _corpus = " ".join(" ".join(tokens) for tokens in _filtered_descr)

    # 4. Générer le nuage de mots
    _wordcloud = WordCloud(width=400, height=200, background_color="white", max_words=10, collocations=False).generate(
        _corpus
    )

    # 5. Afficher
    plt.figure(figsize=(8, 4))
    plt.imshow(_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for Category: {_selected_category}")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Voyons les mots qui sont le plus présent dans le jeu de données.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Voyons s'il y a des mots présents dans plusieurs catégories.""")
    return


@app.cell
def _(df_exploded):
    # Créer un ensemble de mots pour chaque catégorie
    word_sets = df_exploded.groupby("main_category")["preprocessed_descr"].apply(set)

    # Trouver l'intersection des ensembles de mots pour toutes les catégories
    common_words = set.intersection(*word_sets)

    # Afficher les mots communs
    print("Mots présents dans toutes les catégories :", common_words)
    return (common_words,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Il y a 65 mots présents dans les 7 catégories.

    Je les supprime du jeu de données.
    """
    )
    return


@app.cell
def _(common_words, df):
    final_df = df.copy()
    # Elimination des tokens (communs aux 7 catégories)
    final_df["preprocessed_descr"] = final_df["preprocessed_descr"].apply(
        lambda tokens: [token for token in tokens if token not in common_words]
    )
    return (final_df,)


@app.cell
def _(WordCloud, final_df, np, plt):
    # liste des catégories
    _categories = final_df["main_category"].unique().tolist()

    # Détermine le nombre de lignes et de colonnes pour les subplots
    _n_cols = 3
    _n_rows = int(np.ceil(len(_categories) / _n_cols))

    # Crée une figure pour contenir tous les subplots
    _fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 8, _n_rows * 4))

    # Aplatis le tableau de sous-graphiques pour faciliter l'itération
    _axes = _axes.flatten()

    for _idx, _category in enumerate(_categories):
        # 1. Sélectionne la catégorie à analyser
        _selected_category = _category

        # 2. Filtre le DataFrame sur la catégorie choisie
        _filtered_descr = final_df.query(f"main_category == '{_selected_category}'")["preprocessed_descr"]

        # 3. Aplatir la liste de listes en une seule chaîne de texte
        _corpus = " ".join(" ".join(tokens) for tokens in _filtered_descr)

        # 4. Générer le nuage de mots
        _wordcloud = WordCloud(width=400, height=200, background_color="white", max_words=10, collocations=False).generate(
            _corpus
        )

        # 5. Affiche le nuage de mots dans le sous-graphe correspondant
        _axes[_idx].imshow(_wordcloud, interpolation="bilinear")
        _axes[_idx].axis("off")
        _axes[_idx].set_title(f"Word Cloud for Category: {_selected_category}", fontsize=20)

    # Masque les sous-graphiques inutilisés
    for j in range(len(_categories), len(_axes)):
        _axes[j].axis("off")

    # Ajuste l'espacement entre les sous-graphiques
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Les mots principaux utilisés dans chaque catégorie semblent bien les décrire.

    Je vais m'en tenir là pour l'instant.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Vectoring et clustering""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Maintenant que le corpus de texte est nettoyé, l'objectif est de transformer les descriptions textuelles en vecteurs interprétables par une machine.

    Cela peut être un vecteur unique ou une matrice de vecteurs, selon la méthode utilisée.

    Je vais employer plusieurs méthodes de vectorisation, y compris des approches de type bag-of-words, TF-IDF, et des méthodes d'embedding comme Word2Vec, BERT, et USE (Universal Sentence Encoder).

    Voici la méthodologie que je vais suivre pour chaque model :

    - Label encoding: Encodage des 7 catégories en valeurs numériques.

    - Vectorisation : Transformation des descriptions textuelles en vecteurs en utilisant la méthode choisie, avec des étapes de nettoyage supplémentaires comme le seuil de fréquence des mots et la normalisation.

    - Clustering : Application d'un algorithme de clustering (K-means) avec 7 clusters, correspondant aux 7 catégories cibles.

    - Projection : Réduction de la dimensionnalité à deux dimensions à l'aide de t-SNE pour visualiser les résultats.

    - Visualisation : Affichage des résultats du clustering et comparaison avec les catégories réelles pour évaluer la performance visuelle.

    - Évaluation : Calcul du score ARI (Adjusted Rand Index) pour quantifier la similarité entre notre clustering et les catégories réelles.

    - Vérification de la propriété intellectuelle : Assurer que les textes traités ne relèvent pas d'une propriété intellectuelle protégée.

    Cette approche me permettra de comparer l'efficacité des différentes méthodes de vectorisation pour classifier les produits.

    Je m'assurerais également de respecter les droits de propriété intellectuelle.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Encodage des catégories""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Je commence par créer un nouveau dataframe avec uniquement les colonnes **main_category** et **preprocessed_descr**""")
    return


@app.cell
def _(final_df):
    description_df = final_df[["main_category", "preprocessed_descr"]].copy()
    description_df.head()
    return (description_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""J'encode les catégories pour pouvoir les utiliser avec les différents modèles.""")
    return


@app.cell
def _(LabelEncoder, description_df, final_df):
    encoder = LabelEncoder()
    description_df["label"] = encoder.fit_transform(description_df["main_category"])
    final_df["label"] = description_df["label"]
    description_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Sauvegarde du dataframe après le nettoyage du texte""")
    return


@app.cell
def _(final_df):
    final_df.to_csv("data/intermediate/clean_description.csv", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Afin de comparer la performances des différentes méthode de classification, je vais utiliser l'indice ARI et le temps de calcul.""")
    return


@app.cell
def _(pd):
    scores_df = pd.DataFrame(columns=["methode", "ARI", "time"])
    return (scores_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Bag of words

    Le "Bag of Words" (BoW) est une technique simple pour représenter du texte.

    Elle compte combien de fois chaque mot apparaît dans un document, sans tenir compte de l'ordre des mots. Cela transforme le texte en un ensemble de nombres.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### CountVectorizer

    CountVectorizer crée un vocabulaire de tous les mots uniques présents dans les descriptions produit, puis compte combien de fois chaque mot de ce vocabulaire apparaît par description. 

    Cela permet de transformer le texte en une représentation numérique qui peut être utilisée dans des modèles d'apprentissage automatique.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    J'ai itérer plusieurs fois en modifiant les paramètres de CountVectorizer sans réussir à obtenir un meilleur score ARI qu'avec les paramètres par défauts.

    > Je spécifie les paramètres de CountVectorizer **min_df=0.05** et **max_df=0.95**.

    > Ainsi j'espère filtrer les mots très peu présent et les mots présents dans plusieurs catégories.

    > Après plusieurs itération, capturer le contexte en paramétrant **ngram_range=(2,3)** semble améliorer le clustering.

    > J'ai également limiter la vectorisation aux 50 mots les plus présents dans le corpus (**max_features=50**). 

    /// admonition | Info ngram_range

        Bigramme (2-gramme) : C'est une paire de mots consécutifs. Dans la même phrase, les bigrammes seraient **"C'est un"** et **"un exemple"**.

        Trigramme (3-gramme) : C'est une séquence de trois mots consécutifs. Dans notre exemple, le trigramme serait **"C'est un exemple"**.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Je peux créer une liste de mots basés sur les mots les plus présents par catégorie de produit.""")
    return


@app.cell
def _(df_exploded):
    top_words = (
        df_exploded.groupby("main_category")["preprocessed_descr"]
        .value_counts()
        .groupby(level=0)  # group by category
        .head(10)  # top 10 mots par catégorie
        .reset_index(name="count")
    )
    vocabulary = top_words["preprocessed_descr"].unique()
    print(vocabulary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Erreur
        type: danger

        Ne pas utiliser la liste de vocabulaire basée sur les données. Cela conduit à une fuite de données du modèle.

    ///
    """
    )
    return


@app.cell
def _(CountVectorizer, description_df, pd, time):
    # chargement du corpus qui est tokeniser
    corpus_tokens = description_df["preprocessed_descr"]

    # Convertir chaque liste de tokens en une chaîne de caractères
    corpus = corpus_tokens.apply(" ".join)

    _start_timer = time.time()
    # Initialiser le vectoriseur BoW
    # count_vectorizer = CountVectorizer(ngram_range=(2, 2))
    count_vectorizer = CountVectorizer()

    # Appliquer le BoW au corpus bow = term_frequency
    _term_frequency = count_vectorizer.fit_transform(corpus)

    # Création d'un dataframe
    bow_df = pd.DataFrame(_term_frequency.toarray(), columns=count_vectorizer.get_feature_names_out())
    bow_df.head()

    elapsed_cvectorizer = time.time() - _start_timer
    return bow_df, corpus, elapsed_cvectorizer


@app.cell
def _(bow_df):
    bow_df.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Réductions en deux dimensions

    Afin de visualiser la qualité des clusters, il est nécessaire de projeter les données dans un espace à deux dimensions. Or, le jeu de données d'origine possède autant de dimensions que de mots, ce qui rend la visualisation impossible en l’état.

    Pour remédier à cela, deux techniques de réduction de dimensionnalité sont à notre disposition :

    - **PCA** (Analyse en Composantes Principales)

    - **t-SNE** (t-distributed Stochastic Neighbor Embedding)

    Dans un premier temps, on va appliquer une réduction via PCA afin d’éliminer les dimensions corrélées et de réduire la complexité computationnelle de t-SNE, tout en conservant l'information essentielle.

    A l'étape précédente, j'ai plus de 4000 dimensions...

    Je définis n_components=0.99 dans la PCA afin de conserver le minimum de composantes expliquant 99 % de la variance totale des données.

    Concernant t-SNE, j’ai testé différentes valeurs du paramètre perplexity pour ajuster le niveau de sensibilité aux structures locales dans les données. 
    Cela permet d’observer comment les clusters réagissent face au bruit ou à la densité locale.

    Je teste les valeurs suivantes de perplexity [5,30,50].

    Une perplexity faible (ex. 5) révèle des micro-clusters, tandis qu’une perplexity plus élevée (ex. 50) donne une vue plus globale, mais peut fusionner des groupes.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Fonctions communes à l'analyse des clusters

        Je vais avoir besoin de deux fonctions.

        La première pour entraîner KMeans après avoir vectoriser le corpus de texte, calculer le temps de process et analyser la qualité des clusters avec l'indice ARI.

        Une deuxième pour visualiser en deux dimensions la répartition des produits dans les clusters.
    ///
    """
    )
    return


@app.cell
def _(KMeans, PCA, TSNE, adjusted_rand_score, time):
    def cluster_and_evaluate(data, n_clusters, category_encoded, perplexity=30):
        """Réduit la dimension des données à 2 dimensions avec t-SNE.

        Effectue un clustering sur les données réduites avec KMeans,
        et calcule le score ARI entre les catégories attendues et les clusters déterminés par K-means.

        Arguments :
            data (array-like) : Matrice de données générée à partir des documents.
            n_clusters (int) : Nombre de clusters à former.
            category_encoded (array-like) : Étiquettes des catégories réelles pour calculer le score ARI.
            perplexity (float) : Paramètre de voisinage pour t-SNE (défaut = 30).

        Retourne :
            ARI (float) : Score ARI entre les catégories réelles et les clusters prédits.
            execution_time (float) : Temps d'exécution de la fonction.
            data_TSNE (array) : Données réduites par t-SNE.
            clusters (array) : Étiquettes des clusters déterminés par K-means.
        """
        start_time = time.time()

        print("Shape des données d'entrée :", data.shape)

        # PCA pour réduction préalable
        pca = PCA(n_components=0.99, svd_solver="auto")
        feat_pca = pca.fit_transform(data)
        print("Shape après PCA :", feat_pca.shape)

        # Réduction avec t-SNE (perplexity paramétrable)
        tsne = TSNE(n_components=2, perplexity=perplexity, init="random", random_state=42)
        data_TSNE = tsne.fit_transform(feat_pca)

        # Clustering avec KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data_TSNE)

        # Score ARI
        ARI = round(adjusted_rand_score(category_encoded, kmeans.labels_), 4)
        execution_time = round(time.time() - start_time, 4)

        print(f"Score ARI : {ARI}")
        print(f"Temps d'exécution : {execution_time} secondes")

        return ARI, execution_time, data_TSNE, kmeans.labels_
    return (cluster_and_evaluate,)


@app.cell
def _(plt):
    def plot_comparaison(tsne, cat_num, labels, category_list=None):
        """Trace une comparaison entre les vraies étiquettes et les étiquettes générées par le clustering.

        Arguments :
            tsne (array) : Données réduites à 2 dimensions.
            cat_num (list) : Vraies catégories.
            labels (array) : Étiquettes générées par le clustering.
            category_list (list, optionnel) : Liste des noms de catégories pour la légende.
        """
        # Définir la taille de la figure à tracer
        fig = plt.figure(figsize=(15, 6))

        # Ajouter un premier sous-graphe pour les vraies catégories
        ax = fig.add_subplot(1, 2, 1)
        scatter = ax.scatter(tsne[:, 0], tsne[:, 1], c=cat_num, cmap="Set1")
        # Ajouter une légende si category_list est fourni
        if category_list:
            ax.legend(handles=scatter.legend_elements()[0], labels=category_list, loc="best", title="Catégories")
        # Ajouter un titre
        ax.set_title("Représentation des articles par catégories réelles")

        # Ajouter un second sous-graphe pour les clusters
        ax = fig.add_subplot(1, 2, 2)
        scatter = ax.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap="Set1")
        # Ajouter une légende
        ax.legend(handles=scatter.legend_elements()[0], labels=set(labels), loc="best", title="Clusters")
        # Ajouter un titre
        ax.set_title("Représentation des articles par clusters")

        # Afficher le graphique final
        plt.tight_layout()
        plt.show()
    return (plot_comparaison,)


@app.cell
def _(bow_df, cluster_and_evaluate, description_df, pd):
    print("BOW:")
    ARI_bow, time_bow, tnse_bow, labels_bow = cluster_and_evaluate(
        data=bow_df, n_clusters=7, category_encoded=description_df["label"], perplexity=20
    )
    pd.Series(labels_bow).value_counts()
    return ARI_bow, labels_bow, time_bow, tnse_bow


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Un clustering parfait devrait retrouver les catégories initiales.

    J'ai testé plusieurs valeurs de perplexity et le meilleur résultat est pour perplexity = 20.
    Le score ARI est de 0,4597.

    Comme on peut le voir sur la représentation graphique, les catégories computers, home furnishing et home decor & festive needs sont assez bien représentées.
    """
    )
    return


@app.cell
def _(ARI_bow, elapsed_cvectorizer, scores_df, time_bow):
    # Add score model to the scores dataframe
    _new_row = {"methode": "BOW", "ARI": ARI_bow, "time": elapsed_cvectorizer + time_bow}
    scores_df.loc[0] = _new_row
    scores_df
    return


@app.cell
def _(description_df, labels_bow, plot_comparaison, tnse_bow):
    _categories = description_df["main_category"].unique().tolist()

    plot_comparaison(tsne=tnse_bow, cat_num=description_df["label"], labels=labels_bow, category_list=_categories)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Maintenant je vais voir si j'obtient de meilleurs résultats avec l'utilisation de TF-IDF.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### TF-IDF

    Alors que la méthode **Bag of Words (BoW)** se contente de compter le nombre d’occurrences des mots dans chaque document, **TF-IDF (Term Frequency – Inverse Document Frequency)** va plus loin : 

    > elle pondère chaque mot en fonction de son importance relative dans l'ensemble du corpus de descriptions.

    Concrètement :

    - **TF (Term Frequency**) mesure à quel point un mot est fréquent dans une description donnée.

    - **IDF (Inverse Document Frequency)** réduit le poids des mots trop communs en les divisant par leur fréquence dans l'ensemble des descriptions.

    Cela signifie qu’un mot comme **online**, qui apparaît presque dans toutes les descriptions de produits, aura un poids faible. En revanche, un mot plus spécifique comme **computer**, qui n’apparaît que dans la description de produits de la catégorie **computer**, sera jugé plus informatif et recevra un poids plus élevé.
    """
    )
    return


@app.cell
def _(TfidfVectorizer, corpus, pd, time):
    _start_timer = time.time()

    # Crée une instance de TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Applique le TF-IDF au corpus de descriptions de produits
    _term_frequency = tfidf_vectorizer.fit_transform(corpus)

    # Crée un DataFrame pandas à partir de ce tableau pour une visualisation plus facile
    tfidf_df = pd.DataFrame(_term_frequency.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_df.head()

    elapsed_tfidf = time.time() - _start_timer
    return elapsed_tfidf, tfidf_df


@app.cell
def _(cluster_and_evaluate, description_df, pd, tfidf_df):
    print("TF-IDF:")
    ARI_tfidf, time_tfidf, tnse_tfidf, labels_tfidf = cluster_and_evaluate(
        data=tfidf_df, n_clusters=7, category_encoded=description_df["label"], perplexity=30
    )
    pd.Series(labels_tfidf).value_counts()
    return ARI_tfidf, labels_tfidf, time_tfidf, tnse_tfidf


@app.cell
def _(ARI_tfidf, elapsed_tfidf, scores_df, time_tfidf):
    # Add score model to the scores dataframe
    _new_row = {"methode": "TF-IDF", "ARI": ARI_tfidf, "time": elapsed_tfidf + time_tfidf}
    scores_df.loc[1] = _new_row
    scores_df
    return


@app.cell
def _(description_df, labels_tfidf, plot_comparaison, tnse_tfidf):
    _categories = description_df["main_category"].unique().tolist()

    plot_comparaison(tsne=tnse_tfidf, cat_num=description_df["label"], labels=labels_tfidf, category_list=_categories)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Le temps de calcul est plus long de 2 secondes par rapport à BOW mais le score ARI est meilleur: **+0,5** ce qui donne un ARI de 0,5899.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Word2Vec

    Word2Vec est une technique utilisée pour transformer des mots en vecteurs numériques. Voici comment cela fonctionne :

    - **Apprentissage par le Contexte :** Word2Vec ne connaît pas la définition des mots comme un dictionnaire. Au lieu de cela, il apprend à partir des mots qui apparaissent autour d'un mot donné dans un texte. Par exemple, si les mots "chien" et "chat" apparaissent souvent dans des contextes similaires, Word2Vec les placera près l'un de l'autre dans l'espace vectoriel.

    - **Vecteurs de Mots :** Chaque mot est représenté par un vecteur, qui est essentiellement une liste de nombres. Ces vecteurs sont ajustés de sorte que les mots qui partagent des contextes similaires dans le texte ont des vecteurs similaires.

    - **Réseau de Neurones :** Word2Vec utilise un réseau de neurones pour apprendre ces vecteurs. Le réseau est entraîné sur un grand volume de texte et ajuste les vecteurs pour capturer les relations entre les mots.

    - **Représentation Sémantique :** Bien que Word2Vec ne comprenne pas les mots comme le ferait un humain, il capture des relations sémantiques. Par exemple, les mots "roi" et "reine" pourraient être proches dans l'espace vectoriel car ils apparaissent souvent dans des contextes similaires.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On va utiliser la bibliothèque gensim pour vectoriser les mots des descriptions et Keras pour entrainer un réseau de neurones.

    Gensim apprend les représentations de mots (word embeddings) à partir du contexte statistique des mots dans ton corpus.

    Donc peu importe la langue : français, arabe, japonais, etc.
    Tant que qu'on lui donne un corpus cohérent, il peut apprendre les relations entre les mots.

    Un nettoyage est recommandé avec spacy avant son utililsation pour tenir compte des subtiltités de la langue.

    Par rapport aux techniques de clustering précédentes, ici on aura deux étapes de préprocessing.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Avant de traiter les textes, on cherche à comprendre leurs caractéristiques :

    - Longueur maximale en nombre de caractères.
    - Longueur maximale en nombre de mots.

    Cela permet ensuite d’utiliser le padding, une technique qui consiste à compléter les textes plus courts avec des éléments neutres pour que toutes les séquences aient la même longueur, ce qui est nécessaire pour les modèles de machine learning.
    """
    )
    return


@app.cell
def _(corpus):
    # Nombre max de caractères dans une description
    max_length = corpus.str.len().max()
    print(f"Nombre max de caractères dans une description: {max_length}")

    # Nombre max de mots dans une description
    max_words = corpus.str.split(" ").str.len().max()
    print(f"Nombre max de mots dans une description: {max_words}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Définition des hyperparamètres de Word2Vec:

    - `w2v_size` : taille du vecteur de sortie pour chaque mot.
    - `w2v_window` : fenêtre de contexte autour du mot cible.
    - `w2v_min_count` : fréquence minimale pour considérer un mot.
    - `w2v_epochs` : nombre d’itérations d’apprentissage.
    - `maxlen` : taille fixe pour les séquences de mots (utile pour les réseaux).
    """
    )
    return


@app.cell
def _():
    w2v_size = 100  # 50 à 100 pour petit corpus sinon entre 100 et 300
    w2v_window = 5  # 5 ou 7
    w2v_min_count = 1  # garde tous les mots ok pour taille moyenne de corpus
    w2v_epochs = 100  # tester 100-200-300
    maxlen = 204  # 75% de max words
    return maxlen, w2v_epochs, w2v_min_count, w2v_size, w2v_window


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On transforme les descriptions textuelles en listes de mots (tokens) en minuscules.
    Cette étape est essentielle pour entraîner le modèle Word2Vec.
    """
    )
    return


@app.cell
def _(corpus, simple_preprocess):
    # Préparation des sentences (tokenization)
    descriptions = corpus.to_list()

    descriptions = [simple_preprocess(text) for text in descriptions]
    return (descriptions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Entraînement du modèle Word2Vec

    On entraîne Word2Vec sur les descriptions pour apprendre des représentations vectorielles des mots.

    Ces vecteurs encodent les similarités sémantiques entre mots selon leur contexte.
    """
    )
    return


@app.cell
def _(
    Word2Vec,
    descriptions,
    time,
    w2v_epochs,
    w2v_min_count,
    w2v_size,
    w2v_window,
):
    _start_timer = time.time()
    print("Création et entrainement du modèle Word2Vec")
    w2v_start_time = time.time()
    # Initialize the model
    w2v_model = Word2Vec(min_count=w2v_min_count, window=w2v_window, vector_size=w2v_size)

    # Create the vocabulary from the list of descriptions
    w2v_model.build_vocab(descriptions)

    # Train the model
    w2v_model.train(descriptions, total_examples=w2v_model.corpus_count, epochs=w2v_epochs)

    # Get vectors
    model_vectors = w2v_model.wv
    # Get vocabulary
    w2v_words = model_vectors.index_to_key

    print("Vocabulary size:", len(w2v_words))
    print("Word2Vec trained")

    train_w2v = time.time() - _start_timer
    return model_vectors, train_w2v, w2v_words


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Instanciation du tokenizer Keras

    On utilise un tokenizer pour convertir chaque mot en entier unique.

    Cela permet de représenter une phrase comme une séquence de nombres.

    Le padding est appliqué pour que toutes les séquences aient la même longueur (`maxlen`).
    """
    )
    return


@app.cell
def _(Tokenizer, descriptions, maxlen, pad_sequences, time):
    _start_timer = time.time()
    print("Instantiation du tokenizer")
    # Initialize a tokenizer
    tokenizer = Tokenizer()

    # Train the tokenizer with our descriptions
    tokenizer.fit_on_texts(descriptions)

    # Extract tokens
    tokenized_descriptions = pad_sequences(tokenizer.texts_to_sequences(descriptions), maxlen=maxlen, padding="post")

    # Extract number of unique words from the tokenizer
    num_words = len(tokenizer.word_index) + 1
    print("Nombre de mots uniques:", num_words)

    token_w2v = time.time() - _start_timer
    return num_words, token_w2v, tokenized_descriptions, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Création de la matrice d'embedding

    On crée une matrice où chaque ligne correspond au vecteur Word2Vec d’un mot.

    Cette matrice sera utilisée pour initialiser la couche d’embedding du modèle.

    On mesure aussi le **taux de couverture** : la proportion de mots du vocabulaire ayant un vecteur appris.
    """
    )
    return


@app.cell
def _(model_vectors, np, num_words, time, tokenizer, w2v_size, w2v_words):
    _start_timer = time.time()
    print("Création de la matrice d'embedding'")
    # Create an empty matrix with the sized defined previously
    embedding_matrix = np.zeros((num_words, w2v_size))

    # Loop to fill the matrix with the calculated vectors
    for word, idx in tokenizer.word_index.items():
        if word in w2v_words:
            embedding_vector = model_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[idx] = model_vectors[word]

    word_rate = np.sum(embedding_matrix.sum(axis=1) != 0) / num_words
    print("Word embedding rate : ", word_rate)

    embed_w2v = time.time() - _start_timer
    return (embedding_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On crée un modèle Keras simple :

    - Une couche `Embedding` initialisée avec la matrice Word2Vec (non entraînable).
    - Une couche de **GlobalAveragePooling** pour obtenir un vecteur unique par phrase.

    Ce modèle sert à extraire les vecteurs de représentation pour les descriptions.
    """
    )
    return


@app.cell
def _(
    Constant,
    Embedding,
    GlobalAveragePooling1D,
    Input,
    Model,
    embedding_matrix,
    maxlen,
    num_words,
    time,
    w2v_size,
):
    _start_timer = time.time()

    word_input = Input(shape=(maxlen,), dtype="float64")
    word_embedding = Embedding(
        input_dim=num_words, output_dim=w2v_size, trainable=False, embeddings_initializer=Constant(embedding_matrix)
    )(word_input)
    word_vec = GlobalAveragePooling1D()(word_embedding)
    embed_model = Model(inputs=word_input, outputs=word_vec)
    embed_model.summary()

    pre_keras_w2v = time.time() - _start_timer
    return embed_model, pre_keras_w2v


@app.cell
def _(embed_model, time, tokenized_descriptions):
    _start_timer = time.time()
    # Entraînement du modèle d'embedding
    word2vec = embed_model.predict(tokenized_descriptions)
    print("Shape of embeddings:", word2vec.shape)

    train_keras_w2v = time.time() - _start_timer
    return train_keras_w2v, word2vec


@app.cell
def _(
    cluster_and_evaluate,
    description_df,
    pre_keras_w2v,
    token_w2v,
    train_keras_w2v,
    train_w2v,
    word2vec,
):
    print("Word2Vec:")

    ARI_w2v, time_w2v, tnse_w2v, labels_w2v = cluster_and_evaluate(
        data=word2vec, n_clusters=7, category_encoded=description_df["label"], perplexity=30
    )

    elapsed_w2v = train_w2v + token_w2v + pre_keras_w2v + train_keras_w2v + time_w2v
    return ARI_w2v, elapsed_w2v, labels_w2v, tnse_w2v


@app.cell
def _(ARI_w2v, elapsed_w2v, scores_df):
    # Add score model to the scores dataframe
    _new_row = {"methode": "Word2Vec", "ARI": ARI_w2v, "time": elapsed_w2v}
    scores_df.loc[2] = _new_row
    scores_df
    return


@app.cell
def _(description_df, labels_w2v, plot_comparaison, tnse_w2v):
    _categories = description_df["main_category"].unique().tolist()

    plot_comparaison(tsne=tnse_w2v, cat_num=description_df["label"], labels=labels_w2v, category_list=_categories)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### BERT

    BERT, un modèle de langage de Google, utilise les transformers pour traiter le langage naturel en considérant le contexte avant et après chaque mot. 

    Pré-entraîné sur de vastes textes, il prédit des mots masqués et la suite des phrases.

    Il peut ensuite être ajusté pour des tâches spécifiques, comme la classification de texte, ce qui le rend très efficace pour diverses applications
    """
    )
    return


@app.cell
def _(np, tf, time):
    # Fonction d'embedding pour scinder en batch le corpus et réduire puissance de calcul
    def generate_embeddings(texts, tokenizer, model, max_length=64, batch_size=8):
        embeddings = []
        start = time.time()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, max_length=max_length, padding="max_length", truncation=True, return_tensors="tf")

            outputs = model(inputs)
            # Moyenne sur les tokens (sauf padding)
            batch_embeds = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
            embeddings.append(batch_embeds)

        elapsed = round(time.time() - start, 2)
        return np.vstack(embeddings), elapsed
    return (generate_embeddings,)


@app.cell
def _(AutoTokenizer, TFAutoModel, corpus, generate_embeddings, mo):
    with mo.persistent_cache(name="distilbert_cache"):
        # Paramètres
        MODEL_NAME = "distilbert-base-uncased"
        MAX_LENGTH = 64
        BATCH_SIZE = 10

        # Initialisation
        # Chargement du tokenizer et du modèle
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = TFAutoModel.from_pretrained(
            MODEL_NAME, from_pt=True
        )  # from_pt=True because using tensorflow and not pytorch

        # Préparation des données
        _texts = corpus.to_list()

        # Génération des embeddings
        bert_embeddings, bert_duration = generate_embeddings(
            texts=_texts, tokenizer=_tokenizer, model=_model, max_length=MAX_LENGTH, batch_size=BATCH_SIZE
        )

    print(f"Embeddings shape: {bert_embeddings.shape}")
    print(f"Temps d'exécution : {bert_duration} secondes")
    return bert_duration, bert_embeddings


@app.cell
def _(bert_duration, bert_embeddings, cluster_and_evaluate, description_df):
    print("Bert:")

    ARI_bert, time_bert, tnse_bert, labels_bert = cluster_and_evaluate(
        data=bert_embeddings, n_clusters=7, category_encoded=description_df["label"], perplexity=30
    )

    elapsed_bert = bert_duration + time_bert
    return ARI_bert, elapsed_bert, labels_bert, tnse_bert


@app.cell
def _(ARI_bert, elapsed_bert, scores_df):
    # Add score model to the scores dataframe
    _new_row = {"methode": "Bert", "ARI": ARI_bert, "time": elapsed_bert}
    scores_df.loc[3] = _new_row
    scores_df
    return


@app.cell
def _(description_df, labels_bert, plot_comparaison, tnse_bert):
    _categories = description_df["main_category"].unique().tolist()

    plot_comparaison(tsne=tnse_bert, cat_num=description_df["label"], labels=labels_bert, category_list=_categories)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### USE""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    USE est un modèle de traitement de langage naturel (NLP) développé par Google. Contrairement à BERT qui se concentre sur la représentation de mots, USE est spécifiquement conçu pour encoder des phrases et des paragraphes entiers.

    Ce qui distingue USE, c'est sa capacité à générer des représentations vectorielles universelles pour des phrases de différentes langues et de différentes longueurs. Ces représentations sont apprises à partir de données non annotées et sont conçues pour capturer des informations sémantiques et syntaxiques importantes dans le text, etc.
    """
    )
    return


@app.cell
def _(np, time):
    def embedding_use_fct(sentences, embed_model, b_size):
        """Génère des embeddings USE par batch."""
        print("Encodage par lots avec Universal Sentence Encoder...")
        embeddings = []
        start = time.time()

        for i in range(0, len(sentences), b_size):
            batch = sentences[i : i + b_size]
            batch_embeddings = embed_model(batch).numpy()
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)
        duration = time.time() - start
        print(f"Encodage terminé en {duration} secondes. Taille : {embeddings.shape}")
        return embeddings
    return (embedding_use_fct,)


@app.cell
def _(corpus, embedding_use_fct, hub, mo, time):
    print("Chargement du modèle USE...")
    _start_timer = time.time()
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # Génération des embeddings
    print("Encodage des descriptions produits...")
    with mo.persistent_cache(name="load_use_cache"):
        use_embeddings = embedding_use_fct(corpus.to_list(), use_model, b_size=32)

        duration_use = time.time() - _start_timer

    print(f"Embeddings shape : {use_embeddings.shape}")
    print(f"Temps total : {duration_use} secondes")
    return duration_use, use_embeddings


@app.cell
def _(cluster_and_evaluate, description_df, duration_use, use_embeddings):
    print("USE:")

    ARI_use, time_use, tnse_use, labels_use = cluster_and_evaluate(
        data=use_embeddings, n_clusters=7, category_encoded=description_df["label"], perplexity=30
    )

    elapsed_use = duration_use + time_use
    return ARI_use, elapsed_use


@app.cell
def _(ARI_use, elapsed_use, scores_df):
    # Add score model to the scores dataframe
    _new_row = {"methode": "USE", "ARI": ARI_use, "time": elapsed_use}
    scores_df.loc[4] = _new_row
    scores_df
    return


@app.cell
def _(description_df, labels_bert, plot_comparaison, tnse_bert):
    _categories = description_df["main_category"].unique().tolist()

    plot_comparaison(tsne=tnse_bert, cat_num=description_df["label"], labels=labels_bert, category_list=_categories)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### all-mpnet-base-v2""")
    return


@app.cell
def _(SentenceTransformer, corpus, mo, time):
    with mo.persistent_cache(name="mpnet_cache"):
        _start_timer = time.time()
        _model = SentenceTransformer("all-mpnet-base-v2")
        mpnet_embeddings = _model.encode(corpus.to_list(), batch_size=16, show_progress_bar=True)

        mpnet_duration = time.time() - _start_timer
    return mpnet_duration, mpnet_embeddings


@app.cell
def _(cluster_and_evaluate, description_df, mpnet_duration, mpnet_embeddings):
    print("Mpnet:")

    ARI_mpnet, time_mpnet, tnse_mpnet, labels_mpnet = cluster_and_evaluate(
        data=mpnet_embeddings, n_clusters=7, category_encoded=description_df["label"], perplexity=30
    )

    elapsed_mpnet = mpnet_duration + time_mpnet
    return ARI_mpnet, elapsed_mpnet, labels_mpnet, tnse_mpnet


@app.cell
def _(ARI_mpnet, elapsed_mpnet, scores_df):
    # Add score model to the scores dataframe
    _new_row = {"methode": "Mpnet", "ARI": ARI_mpnet, "time": elapsed_mpnet}
    scores_df.loc[5] = _new_row
    scores_df
    return


@app.cell
def _(description_df, labels_mpnet, plot_comparaison, tnse_mpnet):
    _categories = description_df["main_category"].unique().tolist()

    plot_comparaison(tsne=tnse_mpnet, cat_num=description_df["label"], labels=labels_mpnet, category_list=_categories)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Comparaison des résultats

    J'ai testé 5 méthodes de vectorisation pour classifier les produits en fonction de leurs noms et de leurs descriptions.

    Voyons les résultats obtenus.
    """
    )
    return


@app.cell
def _(plt, scores_df, sns):
    # Define the figure we will use to plot
    _fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    # Define the grid
    gs = _fig.add_gridspec(nrows=1, ncols=2)

    # Add a subplot for the first plot
    _fig_ax1 = _fig.add_subplot(gs[0, 0])
    sns.barplot(data=scores_df, x="methode", y="ARI", label="Score")
    plt.title("Score ARI par méthode de vectorisation", fontweight="bold")
    plt.ylabel("Score ARI")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Méthode utilisée")

    # Add a subplot for the second plot
    _fig_ax2 = _fig.add_subplot(gs[0, 1])
    sns.barplot(data=scores_df, x="methode", y="time", label="Temps")
    plt.title("Temps d'entrainement par méthode", fontweight="bold")
    plt.ylabel("Temps d'entrainement (s)")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Méthode utilisée")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Les performances du modèles Mpnet sont assez impressionnantes.

    Si la temps d'entrainement doit être rapide en revanche il vaut mieux utiliser le méthode TF-IDF qui est moins performant de 13%.

    Classifier automatiquements des produits grâce à leurs descriptions semble tout à fait réalisable.

    Voyons si j'obtient un résultat similaire en analysant uniquement les images.
    """
    )
    return


if __name__ == "__main__":
    app.run()
