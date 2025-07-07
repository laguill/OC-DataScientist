import marimo

__generated_with = "0.14.10"
app = marimo.App(
    app_title="P6 Classification supervisée",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Classification supervisée à partir d'images""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    La faisabilité de regrouper automatiquement des produits de même catégorie a pu être démontré dans les deux notebooks précédents.

    Vous continuez votre travail au sein de "Place du marché". Vous avez partagé le travail effectué lors de votre mission précédente avec Lead Data Scientist, Linda. Elle vous invite désormais à aller plus loin dans l’analyse d’images.

    Voici le mail qu’elle vous a envoyé.

    > Bonjour,

    > Merci beaucoup pour ton travail ! Voici la suite de ta mission :

    > Pourrais-tu réaliser une **classification supervisée à partir des images** ? Je souhaiterais que tu mettes en place une **data augmentation** afin d’optimiser le modèle.

    > [...]

    > Merci encore, bon courage !

    > Linda
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Importation des librairies et chargement des données""")
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from matplotlib.image import imread


    sns.set_theme()
    import json
    import os
    import shutil
    import time

    from pathlib import Path

    import cv2
    import tensorflow as tf

    from PIL import Image, ImageFilter, ImageOps
    from plot_keras_history import plot_history, show_history
    from sklearn import cluster, metrics
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import accuracy_score, adjusted_rand_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tf_keras.applications.vgg16 import VGG16, preprocess_input
    from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
    from tf_keras.layers import (
        Dense,
        Dropout,
        Flatten,
        GlobalAveragePooling1D,
        GlobalAveragePooling2D,
        RandomFlip,
        RandomRotation,
        RandomZoom,
        Rescaling,
    )
    from tf_keras.models import Model, Sequential, load_model
    from tf_keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
    from tf_keras.utils import image_dataset_from_directory, to_categorical
    return (
        Dense,
        Dropout,
        EarlyStopping,
        GlobalAveragePooling2D,
        ImageDataGenerator,
        Model,
        ModelCheckpoint,
        Path,
        RandomFlip,
        RandomRotation,
        RandomZoom,
        Rescaling,
        Sequential,
        VGG16,
        classification_report,
        confusion_matrix,
        image_dataset_from_directory,
        img_to_array,
        json,
        load_img,
        load_model,
        np,
        os,
        pd,
        plot_history,
        plt,
        preprocess_input,
        show_history,
        shutil,
        sns,
        tf,
        time,
        to_categorical,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Lecture du dataset""")
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/intermediate/clean_description.csv", usecols=["image", "main_category", "label"])
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Remaniement du champ image""")
    return


@app.cell
def _(Path, df):
    images_path = Path("data/raw/Images/")
    df["image_path"] = [str(images_path / x) for x in df["image"]]
    df["image_path"].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Encodage de la catégorie""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Info

        J'ai précédemment encoder les catégories.
        ```python
        encoder = LabelEncoder()
        df["label"] = encoder.fit_transform(df["main_category"])
        ```
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Préparation et division des données en ensembles d'entraînement/validation et de test""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Je vais diviser le jeu de données en trois ensembles distincts : entraînement, validation et test.

    - La proportion est 80% pour l'entrainement, 10% pour la validation et 10% pour le test.

    - Utilisation du paramètre stratify pour m'assurer d'avoir la même distribution des classes entre les données d'entrainement et de tests.
    """
    )
    return


@app.cell
def _(train_test_split):
    def data_split(X, y, train_frac=0.8, random_state=42):
        """Split data into train,val and test
        param data:       Data to be split
        param train_frac: Ratio of train set to whole dataset

        Randomly split dataset, based on these ratios:
            'train': train_frac
            'valid': (1-train_frac) / 2
            'test':  (1-train_frac) / 2

        Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
        """

        assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, stratify=y, train_size=train_frac, random_state=random_state
        )

        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, train_size=0.5, random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test
    return (data_split,)


@app.cell
def _(data_split, df, to_categorical):
    X = df["image_path"]
    y = to_categorical(df["label"])
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(X, y)
    return X_test, X_train, X_val, y_test, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Classification supervisée""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Création du modèle de classification""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    En premier lieu, définissons une fonction qui nous permettra de d'implémenter et modifier un modèle préentrainé pour l'adapter à ce que nous allons faire.

    Le modèle utilisé ici est un modèle VGG16 préentrainé sur les images "imagenet" sur lequel nous allons :

    - Supprimer les 3 dernières couches du modèle (include_top=False)
    - Rendre les couches non entrainables, afin de conserver les poids du modèle tel qu'ils ont été définis dans son entrainement
    - Ajouter des couches afin de modifier la sortie du modèle et lui permettre de prédire nos 7 classes.

    Le modèle sera compilé avec comme fonction de perte une entropie croisée et comme optimiseur adam.
    """
    )
    return


@app.cell
def _(Dense, Dropout, GlobalAveragePooling2D, Model, VGG16):
    def create_model():
        # Implement the pretrained model
        model0 = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

        # Set layers to non trainable to keep the weights of the pretrained model
        for layer in model0.layers:
            layer.trainable = False

        # Get the output layer of the model
        output = model0.output
        # Upgrade the model
        output = GlobalAveragePooling2D()(output)
        output = Dense(256, activation="relu")(output)
        output = Dropout(0.5)(output)
        # Define the new output with 7 classes and a softmax function
        predictions = Dense(7, activation="softmax")(output)

        # Redefine the whole model
        model = Model(inputs=model0.input, outputs=predictions)
        # Compile the new model
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        print(model.summary())

        return model
    return (create_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Création des listes de scoring""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Créons des listes pour stocker les scores des différents modèles, nous les réutiliserons pour faire un comparatif en fin d'étude."""
    )
    return


@app.cell
def _(pd):
    # Create dataframe to store scores
    scores_df = pd.DataFrame(
        columns=["methode", "training_times", "loss_validations", "loss_tests", "accuracy_validation", "accuracy_tests"]
    )
    return (scores_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Première approche: Traitement simple des images""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Pour la première approche, nous ferons une préparation initiale simple de l'ensemble des images avant une classification supervisée."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Préparation des images""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Le préprocessing de nos images sera ici uniquement une remise à la taille attendue par le modèle et une conversion en niveau de gris.

    Enfin l'image sera adaptée à la forme attendue par le modèle.
    """
    )
    return


@app.cell
def _(img_to_array, load_img, np, preprocess_input):
    def prepare_images(data):
        # Créer une liste pour stocker les images préparées
        prepared_images = []

        # Pour chaque chemin d'image dans les données
        for image_path in data:
            # Charger l'image à partir du fichier
            image = load_img(image_path, target_size=(224, 224))

            # Convertir l'image en un tableau numpy
            image = img_to_array(image)

            # Redimensionner l'image pour qu'elle corresponde aux besoins du modèle
            image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))

            # Prétraiter l'image (normalisation, etc.)
            image = preprocess_input(image)

            # Ajouter l'image convertie en tableau à la liste
            prepared_images.append(image)

        # Retourner les images préparées sous forme de tableau numpy
        return np.array(prepared_images)
    return (prepare_images,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Appliquons notre fonction de préprocessing sur nos images.""")
    return


@app.cell
def _(X_test, X_train, X_val, mo, prepare_images, time):
    with mo.persistent_cache("preprocessed_image_model1"):
        _start_time = time.time()
        X_train_preprocessed = prepare_images(X_train)
        X_val_preprocessed = prepare_images(X_val)
        X_test_preprocessed = prepare_images(X_test)
        time_preprocessed_method_1 = round(time.time() - _start_time, 0)
    return (
        X_test_preprocessed,
        X_train_preprocessed,
        X_val_preprocessed,
        time_preprocessed_method_1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Définition du modèle""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Premièrement, nous définirons notre modèle, puis :

    - un checkpoint permettant l'enregistrement du modèle dans un fichier
    - les conditions d'arrêt anticipé de l'entrainement
    """
    )
    return


@app.cell
def _(EarlyStopping, ModelCheckpoint):
    # Chemin pour sauvegarder les meilleurs poids du modèle
    model1_save_path = "models/model1_best_weights.keras"
    model1_log_path = "models/model1_training_log"
    training_time1_path = "models/model1_train_time.json"

    # Callback pour sauvegarder les meilleurs poids du modèle
    _checkpoint = ModelCheckpoint(
        model1_save_path,
        monitor="val_loss",  # Surveiller la perte de validation
        verbose=1,  # Afficher les messages de sauvegarde
        save_best_only=True,  # Sauvegarder uniquement les meilleurs poids
        mode="min",  # Mode de minimisation de la perte
    )

    # Callback pour arrêter l'entraînement si la performance ne s'améliore pas
    _early_stopping = EarlyStopping(
        monitor="val_loss",  # Surveiller la perte de validation
        mode="min",  # Mode de minimisation de la perte
        verbose=1,  # Afficher les messages d'arrêt précoce
        patience=5,  # Nombre d'époques sans amélioration avant l'arrêt
    )

    # Liste des callbacks à utiliser pendant l'entraînement
    callbacks_list_m1 = [_checkpoint, _early_stopping]
    return (
        callbacks_list_m1,
        model1_log_path,
        model1_save_path,
        training_time1_path,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - **Layer(Type)** nous montre le nom et le type de chaque couche du modèle. Par exemple, **input_1** est une couche d'entrée, alors que **block1_conv1** est une couche de convolution.

    - **Output Shape** est la forme de sortie de chaque couche. Par exemple "**(None, 224, 224, 64)**" signifie qu'après cette couche, la sortie sera un tenseur de dimension (None, 224, 224, 64), où **None** est la taille du lot (batch size), et les dimensions suivantes correspondent aux dimensions spatiales de la sortie de la couche et le dernier nombre représente le nombre de canaux

    - **Param #** représente le nombre de paramètre de chaque couche

    - **Total params** représente le nombre total de paramètres dans le modèle incluant
    **Trainable params** et **Non-trainable params**

    - **Trainable params** représente les paramètres que le modèle peut apprendre pendant l'entraînement, généralement les poids des couches ajouté (par exemple, les poids des couches Denses).

    - **Non-trainable para** représentent les paramètres qui proviennent de couches pré-entraînées et qui ne seront pas modifiés pendant l'entraînement (les poids des couches de convolution de VGG).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Entrainement du modèle""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Pour l'entrainement du modèle nous utilisons notre set d'entrainement et pour la validation le set de validation. Nous choisissons d'entrainer notre modèle sur 50 epochs, bien qu'avec l'earlystopping défini précédemment, celui-ci sera arrêté avant la fin des 50 epochs.

    Enfin, nous sauvegarderons le modèle dans un fichier. A noter que si le modèle a déjà été entrainé, nous réimporterons celui-ci et son historique d'entrainement, principalement pour gagner du temps si nous avions besoin de rejouer le notebook.
    """
    )
    return


@app.cell
def _(tf):
    # Exécuter le modèle sur le CPU ou GPU si dispo en remplaçant par gpu
    device_name = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    return (device_name,)


@app.cell
def _(
    Path,
    X_train_preprocessed,
    X_val_preprocessed,
    callbacks_list_m1,
    create_model,
    device_name,
    json,
    load_model,
    model1_log_path,
    model1_save_path,
    np,
    tf,
    time,
    training_time1_path,
    y_train,
    y_val,
):
    # Vérifie si le modèle a déjà été entraîné (et sauvegardé)
    # Si c'est le cas, charge le modèle à partir de la sauvegarde
    # Utilisé principalement pour économiser du temps lors de la réexécution du notebook
    if os.path.exists(model1_save_path):
        # Ouvrir le modèle à partir du fichier
        model1 = load_model(model1_save_path)
        # Ouvrir le fichier de log d'entraînement
        history1 = np.load(model1_log_path + ".npy", allow_pickle=True).item()
        with open(training_time1_path, "r") as f:
            time_training_method_1 = json.load(f)["training_time_m1"]
        print("Modèle chargé depuis les fichiers.")
    else:
        # Initialiser le temps au début de la fonction
        _start_time = time.time()

        # Entraînement du modèle sur des données
        with tf.device(device_name):
            model1 = create_model()

            _history = model1.fit(
                X_train_preprocessed,
                y_train,
                epochs=50,
                batch_size=64,
                callbacks=callbacks_list_m1,
                validation_data=(X_val_preprocessed, y_val),
                verbose=1,
            )

        # Calculer la durée de la fonction
        time_training_method_1 = round(time.time() - _start_time, 0)
        history1 = _history.history

        # Sauvegarde
        model1.save(model1_save_path)
        np.save(model1_log_path, history1)
        with open(training_time1_path, "w") as f:
            json.dump({"training_time_m1": time_training_method_1}, f)
        print(f"Modèle entraîné en {time_training_method_1} secondes.")
    return history1, model1, time_training_method_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Le processus continue pour chaque époque continue jusqu'à ce que les critères d'arrêt soient satisfaits. L'entraînement a été arrêté prématurément après la 17ème époque car il n'y avait pas d'amélioration de la perte de validation pendant 5 époques consécutives (selon les critères définis par EarlyStopping)."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Scoring et analyse""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Training Accuracy: La précision sur l'ensemble des données d'entraînement est de presque 100%, ce qui signifie que le modèle a réussi à classifier correctement toutes les images de l'ensemble d'entraînement.

    Validation Accuracy : La précision sur l'ensemble de validation est de 82.67%, ce qui signifie que le modèle a réussi à classifier correctement environ 82,67% des images de l'ensemble de validation.

    Un écart entre la précision sur l'ensemble d'entraînement et sur l'ensemble de validation peut indiquer un overfitting. Cependant, la différence ici n'est pas significative, ce qui suggère que le modèle généralise bien.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calculons à présent les scores et affichons les.""")
    return


@app.cell
def _(X_train_preprocessed, X_val_preprocessed, mo, model1, y_train, y_val):
    with mo.persistent_cache("evaluate_model1"):
        # Évaluation sur du dernier epoch
        train_loss_m1, train_accuracy_m1 = model1.evaluate(X_train_preprocessed, y_train, verbose=True)

        # Évaluation sur l'ensemble de validation
        val_loss_m1, val_accuracy_m1 = model1.evaluate(X_val_preprocessed, y_val, verbose=True)

    print(f"Training Accuracy: {train_accuracy_m1:.4f}")
    print()
    print(f"Validation Accuracy:  {val_accuracy_m1:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Training Accuracy :** La précision sur l'ensemble des données d'entraînement est de presque 100%, ce qui signifie que le modèle a réussi à classifier correctement toutes les images de l'ensemble d'entraînement.

    **Validation Accuracy :** La précision sur l'ensemble de validation est de 81%, ce qui signifie que le modèle a réussi à classifier correctement environ 81% des images de l'ensemble de validation.

    Un écart entre la précision sur l'ensemble d'entraînement et sur l'ensemble de validation peut indiquer un overfitting. 

    Cependant, la différence ici n'est pas significative, ce qui suggère que le modèle généralise bien.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calcul des scores de l'epoch optimal.""")
    return


@app.cell
def _(
    X_test_preprocessed,
    X_val_preprocessed,
    mo,
    model1,
    model1_save_path,
    y_test,
    y_val,
):
    with mo.persistent_cache("evaluate_best_model1"):
        # Charger les poids du meilleur modèle
        model1.load_weights(model1_save_path)

        # Évaluation sur l'ensemble de validation
        val_loss_final_m1, val_accuracy_final_m1 = model1.evaluate(X_val_preprocessed, y_val, verbose=False)

        # Évaluation sur l'ensemble de test
        test_loss_m1, test_accuracy_m1 = model1.evaluate(X_test_preprocessed, y_test, verbose=False)

    print(f"Validation Accuracy: {val_accuracy_final_m1:.4f}")
    print(f"Validation Loss: {val_loss_final_m1:.4f}")
    print(f"Test Accuracy: {test_accuracy_m1:.4f}")
    print(f"Test Loss: {test_loss_m1:.4f}")
    return (
        test_accuracy_m1,
        test_loss_m1,
        val_accuracy_final_m1,
        val_loss_final_m1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Validation Accuracy (Précision de validation) :** 0,8190

    Cela signifie que votre modèle a correctement prédit environ 81,90 % des échantillons dans l'ensemble de validation.
    C'est une bonne indication de la performance du modèle sur des données qu'il n'a pas vues pendant l'entraînement.

    **Validation Loss (Perte de validation) :** 0,7921

    La perte de validation mesure l'erreur commise par le modèle sur l'ensemble de validation.
    Une perte plus faible indique un meilleur ajustement du modèle aux données. Dans ce cas, la perte de validation est relativement faible, ce qui est un bon signe.

    **Test Accuracy (Précision de test) :** 0,8095

    La précision de test montre comment le modèle se comporte sur un ensemble de données complètement nouveau et non utilisé pendant l'entraînement ou la validation.
    Ici, le modèle a une précision de 80,95 %, ce qui est légèrement inférieur à la précision de validation. Cela peut indiquer que le modèle généralise bien, mais il peut y avoir une légère variance due à la différence dans les ensembles de données.

    **Test Loss (Perte de test) :** 0,9067

    La perte de test est légèrement plus élevée que la perte de validation, ce qui est normal car le modèle n'a jamais vu les données de test.
    Une perte de test plus élevée que la perte de validation peut indiquer un léger sur-ajustement (overfitting), mais la différence ici n'est pas très grande.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualisons les scores du modèle durant son entrainement.""")
    return


@app.cell
def _(Path, history1, plot_history, plt, show_history):
    # Plot the history of training
    show_history(history1)
    plot_history(history1, path=Path("models/model1_history.png"))
    plt.close()
    return


@app.cell
def _(
    X_test_preprocessed,
    classification_report,
    confusion_matrix,
    df,
    mo,
    model1,
    np,
    pd,
    plt,
    sns,
    y_test,
):
    with mo.persistent_cache("report_model1"):
        _categories = df["main_category"].unique().tolist()

        # Obtenir les vraies et les prédictions du modèle
        _y_test_true = np.argmax(y_test, axis=1)
        _y_test_pred = np.argmax(model1.predict(X_test_preprocessed), axis=1)

        # Créer la matrice de confusion
        _conf_mat = confusion_matrix(_y_test_true, _y_test_pred)

        # Affichage sous forme de DataFrame avec labels explicites
        _df_conf_mat = pd.DataFrame(
            _conf_mat,
            index=[label for label in _categories],
            columns=[_i for _i in "0123456"],
        )

        # Affichage de la heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(_df_conf_mat, annot=True, cmap="Blues")
        plt.xlabel("Prédictions")
        plt.ylabel("Vérités")
        plt.title("Matrice de Confusion")
        plt.tight_layout()
        ax = plt.gca()  # Get the current Axes

        _report_dict = classification_report(_y_test_true, _y_test_pred, target_names=_categories, output_dict=True)
        df_report_m1 = pd.DataFrame(_report_dict).transpose()

    # Affichage d'un rapport de classification complet
    # mo.vstack([
    #     mo.md("**Evaluation model 1: Traitement simple des images**"),
    #     mo.hstack([ax])
    #     mo.ui.table(df_report_m1),
    # ])
    print("**Evaluation model 1: Traitement simple des images**")

    print(df_report_m1)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Sauvegarde des scores""")
    return


@app.cell
def _(
    scores_df,
    test_accuracy_m1,
    test_loss_m1,
    time_preprocessed_method_1,
    time_training_method_1,
    val_accuracy_final_m1,
    val_loss_final_m1,
):
    _duration = time_preprocessed_method_1 + time_training_method_1

    _new_row = {
        "methode": "functional",
        "training_times": _duration,
        "loss_validations": val_loss_final_m1,
        "loss_tests": test_loss_m1,
        "accuracy_validation": val_accuracy_final_m1,
        "accuracy_tests": test_accuracy_m1,
    }

    scores_df.loc[0] = _new_row
    scores_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Seconde approche: Data generator""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Pour la seconde approche, nous ferons une data augmentation avec ImageDataGenerator pour le preprocessing de nos images."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Création d'un jeu de données de validation""")
    return


@app.cell
def _(df, train_test_split):
    data_train, data_test = train_test_split(df, test_size=0.15, random_state=42)
    return data_test, data_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Préparation des images""")
    return


@app.cell
def _(ImageDataGenerator, preprocess_input):
    # Create the ImageDataGenerator for the train/validation set
    image_data_generator_train = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.25,
        preprocessing_function=preprocess_input,
    )

    # Create the ImageDataGenerator for the test set
    image_data_generator_test = ImageDataGenerator(validation_split=0, preprocessing_function=preprocess_input)
    return image_data_generator_test, image_data_generator_train


@app.cell
def _(
    data_test,
    data_train,
    image_data_generator_test,
    image_data_generator_train,
    time,
):
    _start_time = time.time()
    train_flow = image_data_generator_train.flow_from_dataframe(
        data_train,
        directory="",
        x_col="image_path",
        y_col="main_category",
        weight_col=None,
        target_size=(224, 224),
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=42,
        subset="training",
    )
    validation_flow = image_data_generator_train.flow_from_dataframe(
        data_train,
        directory="",
        x_col="image_path",
        y_col="main_category",
        weight_col=None,
        target_size=(224, 224),
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=42,
        subset="validation",
    )
    test_flow = image_data_generator_test.flow_from_dataframe(
        data_test,
        directory="",
        x_col="image_path",
        y_col="main_category",
        weight_col=None,
        target_size=(224, 224),
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=False,
        seed=42,
        subset=None,
    )
    time_prepare_method_2 = round(time.time() - _start_time, 0)
    return test_flow, time_prepare_method_2, train_flow, validation_flow


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Affichons quelques images transformées pour voir le résultat""")
    return


@app.cell
def _(plt, train_flow):
    batch_images, batch_labels = next(train_flow)
    for _i in range(5):
        img = batch_images[_i]
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Définition du modèle""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Premièrement, nous définirons notre modèle, puis :

    - un checkpoint permettant l'enregistrement du modèle dans un fichier
    - les conditions d'arrêt anticipé de l'entrainement
    """
    )
    return


@app.cell
def _(EarlyStopping, ModelCheckpoint):
    # Chemin pour sauvegarder les meilleurs poids du modèle
    model2_save_path = "models/model2_best_weights.keras"
    model2_log_path = "models/model2_training_log"
    training_time2_path = "models/model2_train_time.json"

    # Callback pour sauvegarder les meilleurs poids du modèle
    _checkpoint = ModelCheckpoint(
        model2_save_path,
        monitor="val_loss",  # Surveiller la perte de validation
        verbose=1,  # Afficher les messages de sauvegarde
        save_best_only=True,  # Sauvegarder uniquement les meilleurs poids
        mode="min",  # Mode de minimisation de la perte
    )

    # Callback pour arrêter l'entraînement si la performance ne s'améliore pas
    _early_stopping = EarlyStopping(
        monitor="val_loss",  # Surveiller la perte de validation
        mode="min",  # Mode de minimisation de la perte
        verbose=1,  # Afficher les messages d'arrêt précoce
        patience=5,  # Nombre d'époques sans amélioration avant l'arrêt
    )

    # Liste des callbacks à utiliser pendant l'entraînement
    callbacks_list_m2 = [_checkpoint, _early_stopping]
    return (
        callbacks_list_m2,
        model2_log_path,
        model2_save_path,
        training_time2_path,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Entrainement du modèle""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Pour l'entrainement du modèle nous utilisons notre set d'entrainement et pour la validation le set de validation. Nous choisissons d'entrainer notre modèle sur 50 epochs, bien qu'avec l'earlystopping défini précédemment, celui-ci sera arrêté avant la fin des 50 epochs.

    Enfin, nous sauvegarderons le modèle dans un fichier. A noter que si le modèle a déjà été entrainé, nous réimporterons celui-ci et son historique d'entrainement, principalement pour gagner du temps si nous avions besoin de rejouer le notebook.
    """
    )
    return


@app.cell
def _(
    Path,
    callbacks_list_m2,
    create_model,
    device_name,
    json,
    load_model,
    model2_log_path,
    model2_save_path,
    np,
    test_flow,
    tf,
    time,
    train_flow,
    training_time2_path,
):
    # Vérifie si le modèle a déjà été entraîné (et sauvegardé)
    # Si c'est le cas, charge le modèle à partir de la sauvegarde
    # Utilisé principalement pour économiser du temps lors de la réexécution du notebook
    if os.path.exists(model2_save_path):
        # Ouvrir le modèle à partir du fichier
        model2 = load_model(model2_save_path)

        # Ouvrir le fichier de log d'entraînement
        history2 = np.load(model2_log_path + ".npy", allow_pickle=True).item()

        with open(training_time2_path, "r") as _f:
            time_training_method_2 = json.load(_f)["training_time_m2"]
        print("Modèle chargé depuis les fichiers.")
    else:
        # Initialiser le temps au début de la fonction
        _start_time = time.time()

        # Entraînement du modèle sur l'ensemble des données
        with tf.device(device_name):
            model2 = create_model()

            _history = model2.fit(
                train_flow,
                epochs=50,
                batch_size=64,
                callbacks=callbacks_list_m2,
                validation_data=test_flow,
                verbose=1,
            )

        # Calculer la durée de la fonction
        time_training_method_2 = round(time.time() - _start_time, 0)
        history2 = _history.history

        # Sauvegarde
        model2.save(model2_save_path)
        np.save(model2_log_path, history2)
        with open(training_time2_path, "w") as _f:
            json.dump({"training_time_m2": time_training_method_2}, _f)
        print(f"Modèle entraîné en {time_training_method_2} secondes.")
    return history2, model2, time_training_method_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Scoring et analyse""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calculons à présent les scores et affichons les.""")
    return


@app.cell
def _(mo, model2, train_flow, validation_flow):
    with mo.persistent_cache("evaluate_last_model2"):
        train_loss_m2, train_accuracy_m2 = model2.evaluate(train_flow, verbose=True)

        val_loss_m2, val_accuracy_m2 = model2.evaluate(validation_flow, verbose=True)

    print(f"Training Accuracy: {train_accuracy_m2:.4f}")
    print()
    print(f"Validation Accuracy: {val_accuracy_m2:.4f}")
    return


@app.cell
def _(mo, model2, model2_save_path, test_flow, validation_flow):
    with mo.persistent_cache("evaluate_best_model2"):
        # Charger les poids du meilleur modèle
        model2.load_weights(model2_save_path)

        # Évaluation sur l'ensemble de validation
        val_loss_final_m2, val_accuracy_final_m2 = model2.evaluate(validation_flow, verbose=False)

        # Évaluation sur l'ensemble de test
        test_loss_m2, test_accuracy_m2 = model2.evaluate(test_flow, verbose=False)

    print(f"Validation Accuracy: {val_accuracy_final_m2:.4f}")
    print(f"Validation Loss: {val_loss_final_m2:.4f}")
    print(f"Test Accuracy: {test_accuracy_m2:.4f}")
    print(f"Test Loss: {test_loss_m2:.4f}")
    return (
        test_accuracy_m2,
        test_loss_m2,
        val_accuracy_final_m2,
        val_loss_final_m2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualisons les scores du modèle durant son entrainement.""")
    return


@app.cell
def _(history2, plot_history, plt, show_history):
    # Plot the history of training
    show_history(history2)
    plot_history(history2, path="models/model2_history.png")
    plt.close()
    return


@app.cell
def _(
    classification_report,
    confusion_matrix,
    mo,
    model2,
    np,
    pd,
    plt,
    sns,
    test_flow,
):
    with mo.persistent_cache("report_model2"):
        # Important : shuffle=False dans test_flow lors de sa création sinon erreur lors de la confusion matrix

        # Prédictions du modèle
        _y_pred_proba = model2.predict(test_flow)
        _y_pred = np.argmax(_y_pred_proba, axis=1)

        # Valeurs réelles
        _y_true = test_flow.classes

        # Récupérer le mapping index <-> nom de classe
        _labels_ordered = list(test_flow.class_indices.keys())

        # Créer une matrice de confusion avec noms explicites
        _conf_mat = confusion_matrix(_y_true, _y_pred)
        _df_conf_mat = pd.DataFrame(_conf_mat, index=_labels_ordered, columns=[_i for _i in "0123456"])

        # Heatmap de la matrice de confusion
        plt.figure(figsize=(6, 4))
        _ax = sns.heatmap(_df_conf_mat, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Prédictions")
        plt.ylabel("Vérités")
        plt.title("Matrice de Confusion")
        plt.tight_layout()

        # Rapport de classification par classe
        _report_dict = classification_report(_y_true, _y_pred, target_names=_labels_ordered, output_dict=True)
        df_report_m2 = pd.DataFrame(_report_dict).transpose()

    # mo.vstack([
    #     mo.md("**Evaluation model 2: Data Augmentation**"),
    #     mo.hstack([_ax]),
    #     mo.ui.table(df_report_m2),
    # ])
    print("**Evaluation model 2: Data Augmentation**")

    print(df_report_m2)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Sauvegarde des scores""")
    return


@app.cell
def _(
    scores_df,
    test_accuracy_m2,
    test_loss_m2,
    time_prepare_method_2,
    time_training_method_2,
    val_accuracy_final_m2,
    val_loss_final_m2,
):
    _duration = time_prepare_method_2 + time_training_method_2

    _new_row = {
        "methode": "data generator",
        "training_times": _duration,
        "loss_validations": val_loss_final_m2,
        "loss_tests": test_loss_m2,
        "accuracy_validation": val_accuracy_final_m2,
        "accuracy_tests": test_accuracy_m2,
    }

    scores_df.loc[1] = _new_row
    scores_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Troisième approche: Dataset sans dataaugmentation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Cette fois-ci, nous opterons pour une approche par dataset, sans data augmentation

    Cette approche travaillant sur les données depuis le disque dur, nous allons devoir scinder notre jeu de données en jeu d'entrainement/validation et test directement sur les fichiers.

    Les images doivent également être rangées dans des dossiers suivant la classe à laquelle elles appartiennent.
    """
    )
    return


@app.cell
def _(Path, df, os, shutil, train_test_split):
    # Sépare le DataFrame 'df' en deux ensembles : 85% pour l'entraînement, 15% pour le test
    _data, _data_test = train_test_split(df, test_size=0.15, random_state=42)

    # Définition des chemins vers les dossiers d'images d'entraînement et de test
    path_train = "data/raw/Images_train/"
    path_test = "data/raw/Images_test/"

    # create image_train dir
    if not os.path.exists(path_train):
        os.makedirs(path_train)
        # Parcourir les données d'entraînement
        for idx, ser in _data.iterrows():
            # Créer un sous-dossier pour chaque catégorie principale si nécessaire
            category_path = os.path.join(path_train, ser["main_category"])
            if not os.path.exists(category_path):
                os.makedirs(category_path)

            # Copier l'image dans le bon dossier
            image_name = ser["image"]
            shutil.copy(ser["image_path"], os.path.join(category_path, image_name))

    # Parcourir les données de test
    for idx, ser in _data_test.iterrows():
        # Créer un sous-dossier pour chaque catégorie principale si nécessaire
        category_path = os.path.join(path_test, ser["main_category"])
        if not os.path.exists(category_path):
            os.makedirs(category_path)

        # Copier l'image dans le bon dossier
        image_name = ser["image"]
        shutil.copy(ser["image_path"], os.path.join(category_path, image_name))
    return path_test, path_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Préparation des images""")
    return


@app.cell
def _(image_dataset_from_directory, np, path_test, path_train, time):
    _beginning_time = time.time()
    dataset_train = image_dataset_from_directory(
        path_train,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=42,
        validation_split=0.25,
        subset="training",
    )
    dataset_validation = image_dataset_from_directory(
        path_train,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=42,
        validation_split=0.25,
        subset="validation",
    )
    dataset_test = image_dataset_from_directory(
        path_test,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=42,
        validation_split=0,
        subset=None,
    )
    time_prepare_method_3 = np.round(time.time() - _beginning_time, 0)
    return (
        dataset_test,
        dataset_train,
        dataset_validation,
        time_prepare_method_3,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Définition du modèle""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Premièrement, nous définirons notre modèle, puis :

    - un checkpoint permettant l'enregistrement du modèle dans un fichier
    - les conditions d'arrêt anticipé de l'entrainement
    """
    )
    return


@app.cell
def _(EarlyStopping, ModelCheckpoint):
    # Chemin pour sauvegarder les meilleurs poids du modèle
    model3_save_path = "models/model3_best_weights.keras"
    model3_log_path = "models/model3_training_log"
    training_time3_path = "models/model3_train_time.json"

    # Callback pour sauvegarder les meilleurs poids du modèle
    _checkpoint = ModelCheckpoint(
        model3_save_path,
        monitor="val_loss",  # Surveiller la perte de validation
        verbose=1,  # Afficher les messages de sauvegarde
        save_best_only=True,  # Sauvegarder uniquement les meilleurs poids
        mode="min",  # Mode de minimisation de la perte
    )

    # Callback pour arrêter l'entraînement si la performance ne s'améliore pas
    _early_stopping = EarlyStopping(
        monitor="val_loss",  # Surveiller la perte de validation
        mode="min",  # Mode de minimisation de la perte
        verbose=1,  # Afficher les messages d'arrêt précoce
        patience=5,  # Nombre d'époques sans amélioration avant l'arrêt
    )

    # Liste des callbacks à utiliser pendant l'entraînement
    callbacks_list_m3 = [_checkpoint, _early_stopping]
    return (
        callbacks_list_m3,
        model3_log_path,
        model3_save_path,
        training_time3_path,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Entrainement du modèle""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Pour l'entrainement du modèle nous utilisons notre set d'entrainement et pour la validation le set de validation. Nous choisissons d'entrainer notre modèle sur 50 epochs, bien qu'avec l'earlystopping défini précédemment, celui-ci sera arrêté avant la fin des 50 epochs.

    Enfin, nous sauvegarderons le modèle dans un fichier. A noter que si le modèle a déjà été entrainé, nous réimporterons celui-ci et son historique d'entrainement, principalement pour gagner du temps si nous avions besoin de rejouer le notebook.
    """
    )
    return


@app.cell
def _(
    Path,
    callbacks_list_m3,
    create_model,
    dataset_train,
    dataset_validation,
    device_name,
    json,
    load_model,
    model3_log_path,
    model3_save_path,
    np,
    tf,
    time,
    training_time3_path,
):
    # Vérifie si le modèle a déjà été entraîné (et sauvegardé)
    # Si c'est le cas, charge le modèle à partir de la sauvegarde
    # Utilisé principalement pour économiser du temps lors de la réexécution du notebook
    if os.path.exists(model3_save_path):
        # Ouvrir le modèle à partir du fichier
        model3 = load_model(model3_save_path)

        # Ouvrir le fichier de log d'entraînement
        history3 = np.load(model3_log_path + ".npy", allow_pickle=True).item()

        with open(training_time3_path, "r") as _f:
            time_training_method_3 = json.load(_f)["training_time_m3"]
        print("Modèle chargé depuis les fichiers.")
    else:
        # Initialiser le temps au début de la fonction
        _start_time = time.time()

        # Entraînement du modèle sur l'ensemble des données
        with tf.device(device_name):
            model3 = create_model()

            _history = model3.fit(
                dataset_train,
                epochs=50,
                batch_size=64,
                callbacks=callbacks_list_m3,
                validation_data=dataset_validation,
                verbose=1,
            )

        # Calculer la durée de la fonction
        time_training_method_3 = round(time.time() - _start_time, 0)
        history3 = _history.history

        # Sauvegarde
        model3.save(model3_save_path)
        np.save(model3_log_path, history3)
        with open(training_time3_path, "w") as _f:
            json.dump({"training_time_m3": time_training_method_3}, _f)
        print(f"Modèle entraîné en {time_training_method_3} secondes.")
    return history3, model3, time_training_method_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Scoring et analyse""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calculons à présent les scores et affichons les.""")
    return


@app.cell
def _(dataset_train, dataset_validation, mo, model3):
    with mo.persistent_cache("evaluate_last_model3"):
        train_loss_m3, train_accuracy_m3 = model3.evaluate(dataset_train, verbose=True)

        val_loss_m3, val_accuracy_m3 = model3.evaluate(dataset_validation, verbose=True)

    print(f"Training Accuracy: {train_accuracy_m3:.4f}")
    print()
    print(f"Validation Accuracy: {val_accuracy_m3:.4f}")
    return


@app.cell
def _(dataset_test, dataset_validation, mo, model3, model3_save_path):
    with mo.persistent_cache("evaluate_best_model3"):
        # Charger les poids du meilleur modèle
        model3.load_weights(model3_save_path)

        # Évaluation sur l'ensemble de validation
        val_loss_final_m3, val_accuracy_final_m3 = model3.evaluate(dataset_validation, verbose=False)

        # Évaluation sur l'ensemble de test
        test_loss_m3, test_accuracy_m3 = model3.evaluate(dataset_test, verbose=False)

    print(f"Validation Accuracy: {val_accuracy_final_m3:.4f}")
    print(f"Validation Loss: {val_loss_final_m3:.4f}")
    print(f"Test Accuracy: {test_accuracy_m3:.4f}")
    print(f"Test Loss: {test_loss_m3:.4f}")
    return (
        test_accuracy_m3,
        test_loss_m3,
        val_accuracy_final_m3,
        val_loss_final_m3,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualisons les scores du modèle durant son entrainement.""")
    return


@app.cell
def _(history3, plot_history, plt, show_history):
    # Plot the history of training
    show_history(history3)
    plot_history(history3, path="models/model3_history.png")
    plt.close()
    return


@app.cell
def _(
    classification_report,
    confusion_matrix,
    dataset_test,
    mo,
    model3,
    np,
    pd,
    plt,
    sns,
    tf,
):
    with mo.persistent_cache("report_model3"):
        # Extraire les images et labels du dataset
        _X_test, _y_test_true = [], []
        for images, labels in dataset_test:
            _X_test.append(images)
            _y_test_true.append(labels)

        _X_test = tf.concat(_X_test, axis=0)
        _y_test_true = tf.argmax(tf.concat(_y_test_true, axis=0), axis=1).numpy()

        # Prédictions
        _y_test_pred = np.argmax(model3.predict(_X_test), axis=1)

        # Créer une matrice de confusion avec noms explicites
        _conf_mat = confusion_matrix(_y_test_true, _y_test_pred)
        _df_conf_mat = pd.DataFrame(
            _conf_mat,
            index=dataset_test.class_names,
            columns=[_i for _i in "0123456"],
        )

        # Heatmap de la matrice de confusion
        plt.figure(figsize=(6, 4))
        _ax = sns.heatmap(_df_conf_mat, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Prédictions")
        plt.ylabel("Vérités")
        plt.title("Matrice de Confusion")
        plt.tight_layout()

        # Rapport de classification par classe
        _report_dict = classification_report(
            _y_test_true,
            _y_test_pred,
            target_names=dataset_test.class_names,
            output_dict=True,
        )
        df_report_m3 = pd.DataFrame(_report_dict).transpose()

    # mo.vstack([
    #     mo.md("**Evaluation model 3: Dataset**"),
    #     mo.hstack([_ax]),
    #     mo.ui.table(df_report_m3),
    # ])
    print("**Evaluation model 3: Dataset**")

    print(df_report_m3)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Sauvegarde des scores""")
    return


@app.cell
def _(
    scores_df,
    test_accuracy_m3,
    test_loss_m3,
    time_prepare_method_3,
    time_training_method_3,
    val_accuracy_final_m3,
    val_loss_final_m3,
):
    _duration = time_prepare_method_3 + time_training_method_3

    _new_row = {
        "methode": "dataset sans augmentation",
        "training_times": _duration,
        "loss_validations": val_loss_final_m3,
        "loss_tests": test_loss_m3,
        "accuracy_validation": val_accuracy_final_m3,
        "accuracy_tests": test_accuracy_m3,
    }

    scores_df.loc[2] = _new_row
    scores_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Quatrième approche: Dataset avec data augmentation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Cette fois, nous allons reprendre l'approche précédente en ajoutant de la data augmentation intégrée au modèle."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Préparation des images""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    La préparation des images est effectuée comme précédemment, aussi, nous réutiliserons les variables suivantes :

    - dataset_train
    - dataset_validation
    - dataset_test
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Définition du modèle""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Pour la définition du modèle nous ne réutiliserons pas la fonction définie initialement, car ce modèle sera doté en premier lieu de la partie data augmentation.

    Ainsi, nous modifierons ce que nous avons fait précédemment comme suit :

    - La première couche gérera la data augmentation, avec :
      - un retournement aléatoire
      - une rotation aléatoire
      - un zoom aléatoire
    - Ensuite nous redimensionnerons l'image
    - Puis les couches suivantes seront les mêmes que pour les autres approches.

    Nous utiliserons également les même paramètre de fonction de perte, d'optimiseur, de checkpoint et d'earlystopping.
    """
    )
    return


@app.cell
def _(EarlyStopping, ModelCheckpoint):
    # Chemin pour sauvegarder les meilleurs poids du modèle
    model4_save_path = "models/model4_best_weights.keras"
    model4_log_path = "models/model4_training_log"
    training_time4_path = "models/model4_train_time.json"

    # Callback pour sauvegarder les meilleurs poids du modèle
    _checkpoint = ModelCheckpoint(
        model4_save_path,
        monitor="val_loss",  # Surveiller la perte de validation
        verbose=1,  # Afficher les messages de sauvegarde
        save_best_only=True,  # Sauvegarder uniquement les meilleurs poids
        mode="min",  # Mode de minimisation de la perte
    )

    # Callback pour arrêter l'entraînement si la performance ne s'améliore pas
    _early_stopping = EarlyStopping(
        monitor="val_loss",  # Surveiller la perte de validation
        mode="min",  # Mode de minimisation de la perte
        verbose=1,  # Afficher les messages d'arrêt précoce
        patience=5,  # Nombre d'époques sans amélioration avant l'arrêt
    )

    # Liste des callbacks à utiliser pendant l'entraînement
    callbacks_list_m4 = [_checkpoint, _early_stopping]
    return (
        callbacks_list_m4,
        model4_log_path,
        model4_save_path,
        training_time4_path,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Entrainement du modèle""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Pour l'entrainement du modèle nous utilisons notre set d'entrainement et pour la validation le set de validation. Nous choisissons d'entrainer notre modèle sur 50 epochs, bien qu'avec l'earlystopping défini précédemment, celui-ci sera arrêté avant la fin des 50 epochs.

    Enfin, nous sauvegarderons le modèle dans un fichier. A noter que si le modèle a déjà été entrainé, nous réimporterons celui-ci et son historique d'entrainement, principalement pour gagner du temps si nous avions besoin de rejouer le notebook.
    """
    )
    return


@app.cell
def _(
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Path,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    Rescaling,
    Sequential,
    VGG16,
    callbacks_list_m4,
    dataset_train,
    dataset_validation,
    device_name,
    json,
    load_model,
    model4_log_path,
    model4_save_path,
    np,
    tf,
    time,
    training_time4_path,
):
    # Vérifie si le modèle a déjà été entraîné (et sauvegardé)
    # Si c'est le cas, charge le modèle à partir de la sauvegarde
    # Utilisé principalement pour économiser du temps lors de la réexécution du notebook
    if os.path.exists(model4_save_path):
        # Charger le modèle existant
        model4 = load_model(model4_save_path)
        history4 = np.load(model4_log_path + ".npy", allow_pickle=True).item()

        with open(training_time4_path, "r") as _f:
            time_training_method_4 = json.load(_f)["training_time_m4"]

        print("Modèle chargé depuis les fichiers.")

    else:
        with tf.device(device_name):
            # Définir la pipeline d'augmentation de données
            data_augmentation = Sequential([
                RandomFlip("horizontal", input_shape=(224, 224, 3)),
                RandomRotation(0.1),
                RandomZoom(0.1),
            ])

            # Charger VGG16 sans la couche top
            base_model = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
            for layer in base_model.layers:
                layer.trainable = False

            # Construire le modèle complet
            model4 = Sequential([
                data_augmentation,
                Rescaling(1.0 / 127.5, offset=-1),
                base_model,
                GlobalAveragePooling2D(),
                Dense(256, activation="relu"),
                Dropout(0.5),
                Dense(7, activation="softmax"),
            ])

            model4.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
            print(model4.summary())

            # Entraînement du modèle
            _start_time = time.time()
            _history = model4.fit(
                dataset_train,
                epochs=50,
                batch_size=64,
                callbacks=callbacks_list_m4,
                validation_data=dataset_validation,
                verbose=1,
            )
            time_training_method_4 = round(time.time() - _start_time, 0)
            history4 = _history.history

            # Sauvegarder le modèle et les données d'entraînement
            model4.save(model4_save_path)
            np.save(model4_log_path, history4)
            with open(training_time4_path, "w") as _f:
                json.dump({"training_time_m4": time_training_method_4}, _f)

            print(f"Modèle entraîné en {time_training_method_4} secondes.")
    return history4, model4, time_training_method_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Scoring et analyse""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calculons à présent les scores et affichons les.""")
    return


@app.cell
def _(dataset_train, dataset_validation, mo, model4):
    with mo.persistent_cache("evaluate_last_model4"):
        train_loss_m4, train_accuracy_m4 = model4.evaluate(dataset_train, verbose=True)

        val_loss_m4, val_accuracy_m4 = model4.evaluate(dataset_validation, verbose=True)

    print(f"Training Accuracy: {train_accuracy_m4:.4f}")
    print()
    print(f"Validation Accuracy: {val_accuracy_m4:.4f}")
    return


@app.cell
def _(dataset_test, dataset_validation, mo, model4, model4_save_path):
    with mo.persistent_cache("evaluate_best_model4"):
        # Charger les poids du meilleur modèle
        model4.load_weights(model4_save_path)

        # Évaluation sur l'ensemble de validation
        val_loss_final_m4, val_accuracy_final_m4 = model4.evaluate(dataset_validation, verbose=False)

        # Évaluation sur l'ensemble de test
        test_loss_m4, test_accuracy_m4 = model4.evaluate(dataset_test, verbose=False)

    print(f"Validation Accuracy: {val_accuracy_final_m4:.4f}")
    print(f"Validation Loss: {val_loss_final_m4:.4f}")
    print(f"Test Accuracy: {test_accuracy_m4:.4f}")
    print(f"Test Loss: {test_loss_m4:.4f}")
    return (
        test_accuracy_m4,
        test_loss_m4,
        val_accuracy_final_m4,
        val_loss_final_m4,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualisons les scores du modèle durant son entrainement.""")
    return


@app.cell
def _(history4, plot_history, plt, show_history):
    # Plot the history of training
    show_history(history4)
    plot_history(history4, path="models/model4_history.png")
    plt.close()
    return


@app.cell
def _(
    classification_report,
    confusion_matrix,
    dataset_test,
    mo,
    model4,
    np,
    pd,
    plt,
    sns,
    tf,
):
    with mo.persistent_cache("report_model4"):
        # Extraire les images et labels du dataset
        _X_test, _y_test_true = [], []
        for _images, _labels in dataset_test:
            _X_test.append(_images)
            _y_test_true.append(_labels)

        _X_test = tf.concat(_X_test, axis=0)
        _y_test_true = tf.argmax(tf.concat(_y_test_true, axis=0), axis=1).numpy()

        # Prédictions
        _y_test_pred = np.argmax(model4.predict(_X_test), axis=1)

        # Créer une matrice de confusion avec noms explicites
        _conf_mat = confusion_matrix(_y_test_true, _y_test_pred)
        _df_conf_mat = pd.DataFrame(
            _conf_mat,
            index=dataset_test.class_names,
            columns=[_i for _i in "0123456"],
        )

        # Heatmap de la matrice de confusion
        plt.figure(figsize=(6, 4))
        _ax = sns.heatmap(_df_conf_mat, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Prédictions")
        plt.ylabel("Vérités")
        plt.title("Matrice de Confusion")
        plt.tight_layout()

        # Rapport de classification par classe
        _report_dict = classification_report(
            _y_test_true,
            _y_test_pred,
            target_names=dataset_test.class_names,
            output_dict=True,
        )
        df_report_m4 = pd.DataFrame(_report_dict).transpose()

    # mo.vstack([
    #     mo.md("**Evaluation model 4: Dataset augmentation**"),
    #     mo.hstack([_ax]),
    #     mo.ui.table(df_report_m4),
    # ])
    (print("**Evaluation model 4: Dataset augmentation**"),)

    print(df_report_m4)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Sauvegarde des scores""")
    return


@app.cell
def _(
    scores_df,
    test_accuracy_m4,
    test_loss_m4,
    time_prepare_method_3,
    time_training_method_4,
    val_accuracy_final_m4,
    val_loss_final_m4,
):
    _duration = time_prepare_method_3 + time_training_method_4

    _new_row = {
        "methode": "dataset avec augmentation",
        "training_times": _duration,
        "loss_validations": val_loss_final_m4,
        "loss_tests": test_loss_m4,
        "accuracy_validation": val_accuracy_final_m4,
        "accuracy_tests": test_accuracy_m4,
    }

    scores_df.loc[3] = _new_row
    scores_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Analyse des résultats""")
    return


@app.cell
def _(plt, scores_df, sns):
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)

    # Ajouter une grille de sous-graphiques avec 2 lignes et 2 colonnes
    gs = fig.add_gridspec(nrows=2, ncols=2)

    # Sous-graphe 1 : Entropie croisée validation/test
    fig_ax0 = fig.add_subplot(gs[0, 0])
    sns.barplot(ax=fig_ax0, data=scores_df, x="methode", y="loss_validations", label="Validation")
    sns.barplot(ax=fig_ax0, data=scores_df, x="methode", y="loss_tests", label="Test", alpha=0.5)
    fig_ax0.set_title("Comparaison entropie croisée validation/test", fontweight="bold")
    fig_ax0.legend(loc="lower left")
    fig_ax0.set_ylabel("Entropie croisée")
    fig_ax0.set_xlabel("Méthode")
    fig_ax0.tick_params(axis="x", rotation=45)

    # Sous-graphe 2 : Exactitude (accuracy) validation/test
    fig_ax1 = fig.add_subplot(gs[0, 1])
    sns.barplot(ax=fig_ax1, data=scores_df, x="methode", y="accuracy_validation", label="Validation")
    sns.barplot(ax=fig_ax1, data=scores_df, x="methode", y="accuracy_tests", label="Test", alpha=0.5)
    fig_ax1.set_title("Comparaison exactitude validation/test", fontweight="bold")
    fig_ax1.legend(loc="lower left")
    fig_ax1.set_ylabel("Exactitude")
    fig_ax1.set_xlabel("Méthode")
    fig_ax1.tick_params(axis="x", rotation=45)

    # Sous-graphe 3 : Temps d'entraînement
    fig_ax2 = fig.add_subplot(gs[1, :])
    sns.barplot(ax=fig_ax2, data=scores_df, x="methode", y="training_times")
    fig_ax2.set_title("Temps d'entraînement des modèles", fontweight="bold")
    fig_ax2.set_ylabel("Temps d'entraînement (s)")
    fig_ax2.set_xlabel("Méthode")
    fig_ax2.tick_params(axis="x", rotation=45)

    # Afficher la figure complète
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Les graphiques illustrent les performances et les temps d'entraînement de quatre méthodes d'entraînement de modèles : fonctionnelle, data generator, dataset sans augmentation, et dataset avec augmentation.

        L'entropie croisée, plus basse pour une meilleure performance, montre des différences entre les phases de validation et de test, avec un risque de surapprentissage pour la méthode "dataset avec augmentation".

        L'exactitude, élevée et similaire entre validation et test pour toutes les méthodes, est légèrement inférieure pour le test avec augmentation de données.

        Les temps d'entraînement varient, avec la méthode "fonctionnelle" étant la plus rapide et "dataset avec augmentation" la plus longue.

        Les méthodes "fonctionnelle" et "data generator" offrent un bon compromis entre performance et temps d'entraînement.

        L'augmentation des données améliore l'exactitude mais augmente le risque de surapprentissage et le temps d'entraînement.    """
    )
    return


if __name__ == "__main__":
    app.run()
