import marimo

__generated_with = "0.14.10"
app = marimo.App(app_title="P6 prétraitement images", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Notebook de prétraitement des images""")
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

    Dans ce notebook, je vais me concentrer sur la réalisation de la faisabilité à partir d'images.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("notebooks/public/Projet_textimage_logo.png").center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Initialisation des librairies et import des images""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Installation d'opencv
        type: warning

        La librairie libgthread-2_0-0 est nécessaire. Penser à vérifier sa présence sur le système.
        ```bash
        # openSUSE Tumbleweed
        sudo zypper in libgthread-2_0-0
        ```

    ///
    """
    )
    return


@app.cell
def _():
    # Plot images
    import time

    from pathlib import Path

    # Process images
    import cv2
    import matplotlib.pyplot as plt

    # Graphiques
    import seaborn as sns
    import tensorflow as tf

    from matplotlib.image import imread
    from PIL import Image, ImageFilter, ImageOps
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.decomposition import PCA

    # sklearn
    from sklearn.manifold import TSNE

    # Metriques
    from sklearn.metrics import adjusted_rand_score
    from tf_keras.applications.vgg16 import VGG16, preprocess_input
    from tf_keras.layers import Dense, Flatten
    from tf_keras.models import Model, Sequential
    from tf_keras.preprocessing.image import img_to_array, load_img
    from tf_keras.utils import to_categorical

    sns.set_theme()

    import numpy as np
    import pandas as pd

    return (
        Image,
        ImageFilter,
        ImageOps,
        KMeans,
        MiniBatchKMeans,
        Model,
        PCA,
        Path,
        TSNE,
        VGG16,
        adjusted_rand_score,
        cv2,
        img_to_array,
        load_img,
        np,
        pd,
        plt,
        preprocess_input,
        sns,
        time,
    )


@app.cell
def _(pd):
    df = pd.read_csv("data/intermediate/clean_description.csv")
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Nettoyage des données

    La mission est de réaliser une étude de la faisabilité d'un moteur de classification automatique d'articles, en utilisant leur image et leur description.

    Je réutilise le jeu de données filtrés dans le notebook précédent et conserve uniquement les colonnes **image**, **main_category** et **label**.
    """
    )
    return


@app.cell
def _(df):
    images_df = df[["image", "main_category", "label"]]
    return (images_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Je vais appliquer différente méthodes pour nettoyer les images.

    Pour visualiser les images, je n'aurais qu'à concaténer le chemin du dossier image avec le nom de l'image.

    ```python
    images_path + images_df["images"][0]
    ```

    Je vais aussi avoir besoin d'un dataframe pour stocker les résultats.
    """
    )
    return


@app.cell
def _(Path, pd):
    images_path = Path("data/raw/Images/")

    scores_df = pd.DataFrame(columns=["methode", "ARI", "time"])
    return images_path, scores_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Analyse des images

    Afin de m'assurer de la qualité des images, je vais visualiser leurs histogrammes associés.

    L'histogramme d'une image permet de voir comment les intensités lumineuses sont réparties dans cette image.

    Cela aide à comprendre si l'image est trop sombre, trop claire ou si elle manque de contraste. En observant l'histogramme, je pourrai décider des ajustements nécessaires pour améliorer la qualité visuelle de l'image.
    """
    )
    return


@app.cell
def _(images_df, images_path, mo):
    _image_path = images_path / images_df["image"][1]
    mo.image(_image_path)
    print(_image_path.name)
    return


@app.cell
def _(np, plt):
    def plot_image(image):
        """Affiche une image et son histogramme.

        Arguments :
            image
        """
        # Convertir l'image en tableau numpy pour calculer l'histogramme
        img_array = np.array(image)

        # Calculer l'histogramme
        hist, _ = np.histogram(img_array.flatten(), bins=256, range=[0, 256])

        # Créer une figure avec deux sous-graphiques
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)

        # Définir la grille
        gs = fig.add_gridspec(nrows=1, ncols=2)

        # Ajouter un sous-graphique pour l'image
        fig_ax1 = fig.add_subplot(gs[0, 0])
        fig_ax1.imshow(image, cmap="gray")
        fig_ax1.set_title("Image")
        fig_ax1.axis("off")  # Masquer les axes pour l'image

        # Ajouter un sous-graphique pour l'histogramme
        fig_ax2 = fig.add_subplot(gs[0, 1])
        fig_ax2.plot(hist)
        fig_ax2.set_title("Histogramme de l'image")
        fig_ax2.set_xlabel("Intensité de pixel")
        fig_ax2.set_ylabel("Fréquence")

        plt.show()

    return (plot_image,)


@app.cell
def _(Image, images_df, images_path):
    _image_path = images_path / images_df["image"][1]
    Image.open(_image_path)
    return


@app.cell
def _(Image, images_df, images_path, plot_image):
    # Charger une image en niveaux de gris
    _image_path = images_path / images_df["image"][0]
    image = Image.open(_image_path)
    plot_image(image)
    return (image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Traitement d'image""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Conversion de l'image en niveau de gris afin de réduire le nombre de dimensions de couleur dans nos images, nous allons tout d'abord les convertir en niveau de gris.""")
    return


@app.function
def convert_to_grayscale(image):
    """Convertit une image en niveaux de gris.

    Arguments :
        imag
    """
    return image.convert("L")


@app.cell
def _(image, plot_image):
    grey_image = convert_to_grayscale(image)
    plot_image(convert_to_grayscale(grey_image))
    return (grey_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Redimensionnement des images

    Une taille couramment utilisée pour réduire la dimension des images tout en conservant suffisamment d'informations est 224x224, ce qui est souvent utilisé dans les modèles de réseaux de neurones convolutionnels pré-entraînés comme VGG, ResNet, etc.
    """
    )
    return


@app.cell
def _(ImageOps):
    def resize_image(image, size=(224, 224)):
        """Redimensionne une image pour qu'elle soit carrée.

        Arguments :
            image : Image -- l'image à traiter
            size : tuple -- (largeur, hauteur) les nouvelles dimensions pour l'image
        """
        return ImageOps.pad(image, size)

    return (resize_image,)


@app.cell
def _(grey_image, plot_image, resize_image):
    resized_image = resize_image(grey_image)
    plot_image(resized_image)
    return (resized_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Correction de l'exposition""")
    return


@app.cell
def _(ImageOps):
    def correct_exposure(image):
        """Normalise le contraste d'une image.

        Arguments :
            image : Image -- l'image à traiter
        """
        return ImageOps.autocontrast(image)

    return (correct_exposure,)


@app.cell
def _(correct_exposure, plot_image, resized_image):
    exposed_image = correct_exposure(resized_image)
    plot_image(exposed_image)
    return (exposed_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Amélioration du contraste""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Cette fonction est destinée à normaliser le contraste d'une image, ce qui signifie qu'elle ajuste les intensités des pixels pour améliorer la qualité visuelle de l'image.

    La distribution se répartie entre 0 et environ 250 ce qui montre une exposition correcte de l'image. Lorsqu'une image est sous-exposée, la distribution est concentrée vers des niveaux faibles de gris donc proche de 0, et inversement pour les images sur-exposés.
    """
    )
    return


@app.cell
def _(ImageOps):
    def correct_contrast(image):
        """Normalise le contraste d'une image.

        Arguments :
            image : Image -- l'image à traiter
        """
        return ImageOps.equalize(image)

    return (correct_contrast,)


@app.cell
def _(correct_contrast, exposed_image, plot_image):
    contrasted_image = correct_contrast(exposed_image)
    plot_image(contrasted_image)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Elimination du bruit""")
    return


@app.cell
def _(ImageFilter):
    def correct_noise(image):
        """Supprime le bruit d'une image.

        Arguments :
            image : Image -- l'image à traiter
        """
        return image.filter(ImageFilter.BoxBlur(1))

    return (correct_noise,)


@app.cell
def _(correct_noise, exposed_image, plot_image):
    image_noised = correct_noise(exposed_image)
    plot_image(image_noised)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""On observe une nette amélioration de l'image après le réglage du contraste""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##### Fonction pour appliquer les filtres""")
    return


@app.cell
def _(
    Image,
    correct_contrast,
    correct_exposure,
    correct_noise,
    np,
    resize_image,
):
    def clean_image(
        path_to_image, resize=True, grayscale=True, exposure=True, contrast=True, noise=True, size=(224, 224)
    ):
        """Nettoie et normalise une image.

        Arguments :
            path_to_image : str -- le chemin de l'image à traiter
            size : tuple -- (largeur, hauteur) les dimensions souhaitées pour l'image
        """
        # Ouvrir l'image
        # Si l'image est envoyée sous forme de tableau (si elle a été ouverte avec OpenCV par exemple)
        image = Image.fromarray(path_to_image) if isinstance(path_to_image, np.ndarray) else Image.open(path_to_image)

        # Redimensionner l'image si nécessaire
        if resize:
            image = resize_image(image, size)

        # Convertir l'image en niveaux de gris si nécessaire
        if grayscale:
            image = convert_to_grayscale(image)

        # Corriger l'exposition si nécessaire
        if exposure:
            image = correct_exposure(image)

        # Corriger le contraste si nécessaire
        if contrast:
            image = correct_contrast(image)

        # Supprimer le bruit si nécessaire
        if noise:
            image = correct_noise(image)

        return image

    return (clean_image,)


@app.cell
def _(clean_image, images_df, images_path, plot_image):
    _image_path = images_path / images_df["image"][0]
    _image = clean_image(_image_path, resize=True, grayscale=True, exposure=False, contrast=False, noise=False)
    plot_image(_image)
    return


@app.cell
def _(clean_image, images_df, images_path, plot_image):
    _image_path = images_path / images_df["image"][0]
    _image = clean_image(_image_path, resize=True, grayscale=True, exposure=True, contrast=False, noise=True)
    plot_image(_image)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""L'application des ces différents filtres nous permet de remarquer l'atout de l'application d'un filtre sur la qualité de l'image. Nous poursuivrons avec le filtre median""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### SIFT

    Extraction de features d'image avec SIFT sur l'ensemble des images.

    L'algorithme SIFT (Scale-Invariant Feature Transform) est une méthode qui permet d'extraire des caractéristiques (ou points d'intérêt) d'une image et de calculer leurs descripteurs.

    Un descripteur est un vecteur qui décrit le voisinage de la caractéristique à laquelle il est associé.

    Il est utilisé pour repérer les paires de caractéristiques qui se ressemblent le plus dans deux images.

    Pour faciliter cette étape de correspondance, le descripteur doit présenter de nombreuses propriétés d'invariance (rotation, échelle, exposition).

    Ainsi, les descripteurs de deux caractéristiques identiques à un changement géométrique ou photométrique près doivent être aussi proches que possible.

    L'étape de correspondance revient alors à comparer les descripteurs.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Extraction des descripteurs des images""")
    return


@app.cell
def _(clean_image, cv2, mo, np, time):
    def generate_sift_descriptors(
        image_paths, resize=True, grayscale=True, exposure=False, contrast=False, noise=False
    ):
        """Génère les descripteurs SIFT pour une collection d'images.

        Arguments :
            image_paths : list -- liste des chemins complets des images
            resize, grayscale, exposure, contrast, noise : bool -- options de prétraitement
        Retourne :
            sift_descriptors_list : list -- liste des descripteurs SIFT par image
            sift_descriptors_all : np.ndarray -- concaténation de tous les descripteurs
            duration : float -- temps total d'exécution
        """
        # Enregistrer le temps de début pour calculer la durée totale d'exécution
        start_time = time.time()

        # Créer l'objet SIFT pour l'extraction des caractéristiques
        sift = cv2.SIFT_create()

        # Initialiser une liste pour sauvegarder les descripteurs SIFT de chaque image
        sift_descriptors_list = []

        # Parcourir chaque chemin d'image dans la liste des chemins d'images
        for img_path in mo.status.progress_bar(image_paths):
            # Charger l'image à partir du chemin spécifié
            image = cv2.imread(img_path)

            # Vérifier si l'image a été correctement chargée
            if image is None:
                print(f"Erreur : impossible de charger {img_path}")
                continue

            # Nettoyer et prétraiter l'image selon les options spécifiées
            image = clean_image(image, resize, grayscale, exposure, contrast, noise)

            # Détecter les points clés et calculer les descripteurs SIFT
            keypoints, descriptors = sift.detectAndCompute(np.array(image), None)

            # Si aucun descripteur n'est trouvé, créer un tableau de zéros par défaut
            if descriptors is None:
                descriptors = np.zeros((2, 128), dtype=np.float32)

            # Ajouter les descripteurs de l'image courante à la liste
            sift_descriptors_list.append(descriptors)

        # Filtrer les images qui n'ont pas échoué à générer des descripteurs
        valid_descriptors = [d for d in sift_descriptors_list if d is not None]

        # Concaténer tous les descripteurs valides en un seul tableau numpy
        sift_descriptors_all = np.concatenate(valid_descriptors, axis=0)

        # Calculer la durée totale d'exécution
        duration = time.time() - start_time

        # Retourner les descripteurs par image, tous les descripteurs concaténés, et la durée d'exécution
        return sift_descriptors_list, sift_descriptors_all, duration

    return (generate_sift_descriptors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""J'ajoute une nouvelle colonne au dataframe pour ajouter le chemin d'accès complet aux images.""")
    return


@app.cell
def _(images_df, images_path):
    images_df["image_path"] = [str(images_path / x) for x in images_df["image"]]
    images_df["image_path"].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Objectif: extraire des descripteurs (vecteurs de 128 valeurs) qui capturent les motifs visuels importants de chaque image avant de les regrouper pour créer un vocabulaire visuel commun.""")
    return


@app.cell
def _(generate_sift_descriptors, images_df, mo):
    with mo.persistent_cache("sift_descriptors_cache"):
        # Construction des descripteurs SIFT pour notre collection d'images
        sift_descriptors_list, sift_descriptors_all, descriptors_extraction_time = generate_sift_descriptors(
            images_df["image_path"]
        )
    return sift_descriptors_all, sift_descriptors_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Regroupement des descripteurs par clusters

    Après l'extraction des descripteurs des images, ceux-ci sont utilisés pour déterminer des clusters.

    Ces clusters de descripteurs servent à générer les caractéristiques des images.

    Pour ce faire, les descripteurs de toutes les images sont concaténés et un algorithme de clustering est appliqué.
    """
    )
    return


@app.cell
def _(MiniBatchKMeans, np, sift_descriptors_all, time):
    # Determination du nombre de cluster
    _start_timer = time.time()

    k = int(round(np.sqrt(len(sift_descriptors_all)), 0))
    print("Nombre de clusters estimés : ", k)

    # Clustering
    descriptors_kmeans = MiniBatchKMeans(n_clusters=k, init_size=3 * k, init="k-means++", random_state=42)
    descriptors_kmeans.fit(sift_descriptors_all)

    descriptors_clustering_time = time.time() - _start_timer
    print(f"temps de traitement kmeans : {descriptors_clustering_time:.2f} secondes")
    return descriptors_clustering_time, descriptors_kmeans


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Création des features des images

    Pour cette étape, l'objectif est de prédire les numéros de clusters pour chacun des descripteurs des images.

    Par la suite, un histogramme sera généré afin de représenter la distribution du nombre de descripteurs de l'image au sein de chaque cluster.

    Le but étant de symboliser chaque vecteur par une image.
    """
    )
    return


@app.cell
def _(np):
    def build_histogram(kmeans, descriptors, image_index):
        """Construit un histogramme normalisé.

        Construit un histogramme normalisé (vecteur de caractéristiques) à partir
        des descripteurs SIFT d'une image et d'un modèle KMeans entraîné.

        Arguments :
            kmeans : KMeans -- modèle de clustering (sac de mots visuel)
            descriptors : np.ndarray -- descripteurs SIFT de l'image (n x 128)
            image_index : int -- index ou numéro de l'image (utilisé pour les messages d'erreur)

        Retourne :
            histogram : np.ndarray -- vecteur de caractéristiques (taille = nb de clusters)
        """
        # Prédire à quel cluster appartient chaque descripteur
        cluster_indices = kmeans.predict(descriptors)

        # Initialiser l'histogramme avec des zéros, une case par cluster
        histogram = np.zeros(len(kmeans.cluster_centers_))

        # Nombre total de descripteurs pour l'image
        num_descriptors = len(descriptors)

        # Avertir s'il n'y a pas de descripteur (image vide ou non détectée)
        if num_descriptors == 0:
            print("Problème histogramme image :", image_index)

        # Remplir l'histogramme : pour chaque cluster assigné, incrémenter la case correspondante
        for i in cluster_indices:
            histogram[i] += 1.0 / num_descriptors  # Normalisation par le nombre de descripteurs

        # Retourner l'histogramme normalisé (vecteur de features)
        return histogram

    return (build_histogram,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Le code suivant calcule les caractéristiques SIFT pour chaque image en utilisant un modèle k-means.""")
    return


@app.cell
def _(build_histogram, descriptors_kmeans, np, sift_descriptors_list, time):
    # Initialise le temps au début de la fonction
    _start_time = time.time()

    # Crée une liste pour stocker les caractéristiques de chaque image
    sift_features_by_image = []

    # Parcours chaque ensemble de descripteurs SIFT
    # Construction de l'histogramme (vecteur de caractéristiques) par image
    # Ajout du vecteur de caractéristiques à la liste
    # Conversion de la liste en un tableau NumPy (matrice n_images x n_clusters)
    sift_features_by_image = np.array([
        build_histogram(descriptors_kmeans, descriptors, i) for i, descriptors in enumerate(sift_descriptors_list)
    ])

    # Calcule la durée totale de l'exécution de la fonction
    vectorize_image_time = round(time.time() - _start_time, 0)

    print(f"temps de traitement sift par image par cluster : {vectorize_image_time:.2f} secondes")
    return sift_features_by_image, vectorize_image_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Réduction dimensionnelle PCA/t-SNE des features

    Les images sont décrites par 464 features, je vais réduire le nombre de dimensions du jeu de données afin de faciliter les traitements de clustering et d'affichage.
    """
    )
    return


@app.cell
def _(PCA, TSNE, sift_features_by_image, time):
    _start_timer = time.time()
    _data = sift_features_by_image
    print("Dimensions du jeu de données avant réduction PCA : ", _data.shape)

    # PCA pour réduction préalable
    _pca = PCA(n_components=0.99, svd_solver="auto")
    _feat_pca = _pca.fit_transform(_data)
    print("Shape après PCA :", _feat_pca.shape)

    # Réduction avec t-SNE
    _tsne = TSNE(n_components=2, perplexity=30, init="random", random_state=42)
    tsne_sift = _tsne.fit_transform(_feat_pca)

    time_reduce_sift = time.time() - _start_timer

    print("Dimensions du jeu de données après réduction : ", tsne_sift.shape)
    return time_reduce_sift, tsne_sift


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Clustering

    Réalisons à présent un clustering avec un nombre de cluster égale à 7 (nombre de catégorie du jeu de données) puis le calcul du score ARI par rapport aux attentes.
    """
    )
    return


@app.cell
def _(KMeans, adjusted_rand_score, images_df, time, tsne_sift):
    _start_timer = time.time()

    # Initialise un modèle KMeans pour déterminer les clusters
    _kmeans = KMeans(n_clusters=7, random_state=42)

    # Entraîne le modèle KMeans sur les données réduites
    _kmeans.fit(tsne_sift)

    labels_sift = _kmeans.labels_

    # Calcule le score ARI (Adjusted Rand Index) entre les numéros de catégorie définis précédemment et les labels prédits par KMeans
    ARI_sift = round(adjusted_rand_score(images_df["label"], labels_sift), 4)

    # Calcule la durée totale de l'exécution de la fonction et l'arrondit à l'entier le plus proche
    cluster_sift_time = round(time.time() - _start_timer, 0)
    return ARI_sift, cluster_sift_time, labels_sift


@app.cell
def _(
    ARI_sift,
    cluster_sift_time,
    descriptors_clustering_time,
    scores_df,
    time_reduce_sift,
    vectorize_image_time,
):
    # Add score model to the scores dataframe
    _time = descriptors_clustering_time + vectorize_image_time + time_reduce_sift + cluster_sift_time
    _new_row = {"methode": "SIFT", "ARI": ARI_sift, "time": _time}
    scores_df.loc[0] = _new_row
    scores_df
    return


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
def _(df, labels_sift, plot_comparaison, tsne_sift):
    _categories = df["main_category"].unique().tolist()

    plot_comparaison(tsne=tsne_sift, cat_num=df["label"], labels=labels_sift, category_list=_categories)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    La classification des images selon leur catégorie via SIFT donne de mauvais résultats, que ce soit au niveau visuel ou au niveau du score ARI (0.029).

    Je vais essayer une autre méthode, le CNN Transfer Learning.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### CNN Transfer Learning

    Le CNN Transfer Learning est une technique de deep learning où un modèle pré-entraîné sur une grande quantité de données est utilisé comme point de départ pour résoudre une tâche similaire ou différente.

    Plutôt que de construire et d'entraîner un modèle à partir de zéro, ce qui peut nécessiter beaucoup de données et de temps de calcul, on utilise un modèle déjà entraîné.

    Ensuite, on adapte ce modèle aux nouvelles données ou à la nouvelle tâche en réglant ses poids, souvent uniquement sur les dernières couches, tout en conservant les poids appris dans les couches précédentes.

    Cela permet d'exploiter les connaissances apprises par le modèle pré-entraîné, ce qui peut améliorer considérablement les performances du modèle sur la nouvelle tâche, même avec moins de données d'entraînemenant.

    #### Création du modèle pré-entrainé
    """
    )
    return


@app.cell
def _(Model, VGG16):
    # Charge le modèle
    vgg = VGG16()

    # Extrait l'avant dernière couche du model
    vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

    print(vgg.summary())
    return (vgg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Création des features des images""")
    return


@app.cell
def _(images_df, img_to_array, load_img, mo, np, preprocess_input, time, vgg):
    # Initialize the time at the beginning of the function
    _start_timer = time.time()

    # Create a list to store the features
    vgg_features_by_image = []

    with mo.persistent_cache("vgg_features_by_image"):
        for image_file in mo.status.progress_bar(images_df["image_path"]):
            # charge l'image en indiquant la taille en pixel requise par le model
            _image = load_img(image_file, target_size=(224, 224))
            # convertis l'image en une matrice numpy
            _image = img_to_array(_image)
            _image = np.expand_dims(_image, axis=0)
            _image = preprocess_input(_image)
            vgg_features_by_image.append(vgg.predict(_image, verbose=0)[0])
        vgg_features_by_image = np.asarray(vgg_features_by_image)
        time_vgg_features = round(time.time() - _start_timer, 0)

    print(vgg_features_by_image.shape)
    return time_vgg_features, vgg_features_by_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Réduction dimension et analyse

    Les images sont décrites par 4096 features.
    Afin de réduire le temps de calcul, je vais réduire le nombre de dimensions.

    Comme précédemment, j'applique une ACP en conservant 99% de la variance expliquée, avant d'appliquer un T-SNE pour réduire à 2 dimensions.
    """
    )
    return


@app.cell
def _(PCA, TSNE, time, vgg_features_by_image):
    _start_timer = time.time()
    _data = vgg_features_by_image
    print("Dimensions du jeu de données avant réduction PCA : ", _data.shape)

    # PCA pour réduction préalable
    _pca = PCA(n_components=0.99, svd_solver="auto")
    _feat_pca = _pca.fit_transform(_data)
    print("Shape après PCA :", _feat_pca.shape)

    # Réduction avec t-SNE
    _tsne = TSNE(n_components=2, perplexity=30, init="random", random_state=42)
    tsne_vgg = _tsne.fit_transform(_feat_pca)

    time_reduce_vgg = time.time() - _start_timer

    print("Dimensions du jeu de données après réduction : ", tsne_vgg.shape)
    return time_reduce_vgg, tsne_vgg


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Clusters

    Réalisation des clusters et mesure des performances avec le score ARI.
    """
    )
    return


@app.cell
def _(KMeans, adjusted_rand_score, images_df, time, tsne_vgg):
    _start_timer = time.time()

    # Initialise un modèle KMeans pour déterminer les clusters
    _kmeans = KMeans(n_clusters=7, random_state=42)

    # Entraîne le modèle KMeans sur les données réduites
    _kmeans.fit(tsne_vgg)

    labels_vgg = _kmeans.labels_

    # Calcule le score ARI (Adjusted Rand Index) entre les numéros de catégorie définis précédemment et les labels prédits par KMeans
    ARI_vgg = round(adjusted_rand_score(images_df["label"], labels_vgg), 4)

    # Calcule la durée totale de l'exécution de la fonction et l'arrondit à l'entier le plus proche
    cluster_vgg_time = round(time.time() - _start_timer, 0)
    return ARI_vgg, cluster_vgg_time, labels_vgg


@app.cell
def _(
    ARI_vgg,
    cluster_vgg_time,
    scores_df,
    time_reduce_vgg,
    time_vgg_features,
):
    # Add score model to the scores dataframe
    _time = time_vgg_features + time_reduce_vgg + cluster_vgg_time
    _new_row = {"methode": "VGG", "ARI": ARI_vgg, "time": _time}
    scores_df.loc[1] = _new_row
    scores_df
    return


@app.cell
def _(df, labels_vgg, plot_comparaison, tsne_vgg):
    _categories = df["main_category"].unique().tolist()

    plot_comparaison(tsne=tsne_vgg, cat_num=df["label"], labels=labels_vgg, category_list=_categories)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Comparaison des résultats""")
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
    ### CONCLUSION

    * Le modèle CNN Transfer Learning est beaucoup plus performant que SIFT, que ce soit au niveau du score ARI (0.45) ou au niveau visuel.
    * L'analyse graphique montre visuellement qu'il est réalisable de séparer automatiquement les images selon leurs vraies classes
    * Ceci suffit à démontrer la faisabilité de réaliser ultérieurement une classification supervisée pour déterminer automatiquement les classes des images
    * Cette étape 1 est très rapide à mettre en oeuvre. Une conclusion négative sur la faisabilité aurait éviter de réaliser des traitements beaucoup plus lourd de classification supervisée
    """
    )
    return


if __name__ == "__main__":
    app.run()
