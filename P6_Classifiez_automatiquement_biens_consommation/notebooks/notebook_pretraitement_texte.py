import marimo

__generated_with = "0.14.6"
app = marimo.App(width="medium")


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
        python -m spacy download en_core_web_sm
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

    # Bag of words and TF-IDF
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    # Graphiques
    import seaborn as sns

    sns.set_theme()

    import pandas as pd

    from ydata_profiling import ProfileReport

    # Nettoyage du texte
    import string
    import re

    return (
        CountVectorizer,
        PorterStemmer,
        ProfileReport,
        TfidfVectorizer,
        WordCloud,
        nlp,
        nltk,
        pd,
        plt,
        re,
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
def _(df):
    df.nunique()
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
    Pour ce projet, je vais utiliser que les colonnes **image**,**description** et **product_category_tree** du jeu de données.

    Il y a 1050 produits/description et aucun doublons.

    J' analyserai product_category_tree plus tard.
    """
    )
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
    Je commence par le nettoyage des données c'est à dire:

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
    mo.md(r"""Voyons les mots qui sont le plus présents par catégories.""")
    return


@app.cell(hide_code=True)
def _(category_dropdown, df):
    df_exploded = df.explode("preprocessed_descr")[["main_category", "preprocessed_descr"]]
    top_words = (
        df_exploded.groupby("main_category")["preprocessed_descr"]
        .value_counts()
        .groupby(level=0)  # group by category
        .head(10)  # top 10 mots par catégorie
        .reset_index(name="count")
    )
    top_words.query(f"main_category == '{category_dropdown.value}'")
    return (df_exploded,)


@app.cell(hide_code=True)
def _(df, mo):
    # Cell 2: Create a dropdown to select a category
    category_dropdown = mo.ui.dropdown(
        options=df["main_category"].unique().tolist(), value=df["main_category"].unique()[0], label="Select category"
    )

    category_dropdown
    return (category_dropdown,)


@app.cell
def _(WordCloud, category_dropdown, df, plt):
    # 1. Sélectionne la catégorie pà analyser
    _selected_category = category_dropdown.value

    # 2. Filtrer le DataFrame sur la catégorie choisie
    filtered_descr = df.query(f"main_category == '{category_dropdown.value}'")["preprocessed_descr"]

    # 3. Aplatir la liste de listes en une seule chaîne de texte
    all_tokens = [token for sublist in filtered_descr for token in sublist]
    _text = " ".join(all_tokens)

    # 4. Générer le nuage de mots
    wordcloud = WordCloud(width=400, height=200, background_color="white", max_words=10).generate(_text)

    # 5. Afficher
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for Category: {_selected_category}")
    plt.show()
    return


@app.cell
def _(category_dropdown, df_exploded):
    _selected_category = category_dropdown.value
    _filtered_descr = df_exploded[df_exploded["main_category"] == _selected_category]["preprocessed_descr"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Voyons les mots qui sont le plus présent dans le jeu de données.""")
    return


@app.cell
def _(df_exploded):
    df_exploded["preprocessed_descr"].value_counts().head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Après analyse, je n'ai pas l'impression qu'il y ait des mots bannir car trop présents dans plusieurs catégories.

    Je ne vais pas bannir de mots pour le moment.
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

    - Vectorisation : Transformation des descriptions textuelles en vecteurs en utilisant la méthode choisie, avec des étapes de nettoyage supplémentaires comme le seuil de fréquence des mots et la normalisation.

    - Test avec un exemple : Illustration des étapes de nettoyage et de vectorisation avec une phrase ou un court texte d'exemple.

    - Clustering : Application d'un algorithme de clustering (K-means) avec 7 clusters, correspondant aux 7 catégories cibles.

    - Projection : Réduction de la dimensionnalité à deux dimensions à l'aide de t-SNE pour visualiser les résultats.

    - Visualisation : Affichage des résultats du clustering et comparaison avec les catégories réelles pour évaluer la performance visuelle.

    - Évaluation : Calcul du score ARI (Adjusted Rand Index) pour quantifier la similarité entre notre clustering et les catégories réelles.

    - Vérification de la propriété intellectuelle : Assurer que les textes traités ne relèvent pas d'une propriété intellectuelle protégée.

    Cette approche me permettra de comparer l'efficacité des différentes méthodes de vectorisation pour classifier les produits.

    Je m'assurais également de respecter les droits de propriété intellectuelle.
    """
    )
    return


@app.cell
def _():
    # # Define a list with all the categories
    # category_list = list(set(df['category'].lower()))
    # # Associate a number for each category of our dataframe
    # category_number = [(category_list.index(df.iloc[i]['category'])) for i in range(len(df))]
    return


@app.cell
def _(mo):
    mo.md(r"""Définissons des listes pour conserver les différents scores""")
    return


@app.cell
def _():
    ari_scores = []
    time_scores = []
    return


@app.cell
def _(CountVectorizer, TfidfVectorizer, df):
    # Bag-of-Words
    corpus = df["preprocessed_descr"]
    vectorizer = CountVectorizer(min_df=1, max_df=0.95)
    X_bow = vectorizer.fit_transform(corpus)

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(min_df=1, max_df=0.95)
    X_tfidf = tfidf_vectorizer.fit_transform(corpus)

    # Afficher les features
    print("Bag-of-Words features:")
    print(X_bow.toarray())
    print(vectorizer.get_feature_names_out())

    print("\nTF-IDF features:")
    print(X_tfidf.toarray())
    print(tfidf_vectorizer.get_feature_names_out())
    return


if __name__ == "__main__":
    app.run()
