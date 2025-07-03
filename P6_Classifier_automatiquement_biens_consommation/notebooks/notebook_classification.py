import marimo

__generated_with = "0.14.9"
app = marimo.App()


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
    import importlib
    import os
    import time

    from pathlib import Path

    import cv2
    import tensorflow as tf

    from PIL import Image, ImageFilter, ImageOps
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
    from tf_keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
    from tf_keras.utils import to_categorical
    from plot_keras_history import show_history, plot_history

    import json
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
def _():
    # Create lists to store scores
    training_times = []
    loss_validations = []
    loss_tests = []
    accuracy_validations = []
    accuracy_tests = []
    return (
        accuracy_tests,
        accuracy_validations,
        loss_tests,
        loss_validations,
        training_times,
    )


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
    return X_test_preprocessed, X_train_preprocessed, X_val_preprocessed


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
    training_time1_path = "model1_train_time.json"

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

    - **Non-trainable para** représentent les paramètres qui proviennent de couches pré-entraînées et qui ne seront pas modifiés pendant l'entraîneme (t les poids des couches de convolution de VGG).
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
def _(
    X_train_preprocessed,
    X_val_preprocessed,
    callbacks_list_m1,
    create_model,
    json,
    load_model,
    model1_log_path,
    model1_save_path,
    np,
    os,
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

        # Exécuter le modèle sur le CPU ou GPU si dispo en remplaçant par gpu
        device_name = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"

        # Entraînement du modèle sur l'ensemble d'entraînement
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
        # Évaluation sur l'ensemble d'entraînement
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
    with mo.persistent_cache("evaluate_whole_model1"):
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
    return


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

        _report_dict = classification_report(_y_test_true, _y_test_pred, target_names=_categories, output_dict=True)
        df_report_m1 = pd.DataFrame(_report_dict).transpose()

    # Affichage d'un rapport de classification complet
    mo.vstack([
        mo.md("# Evaluation model 1: Traitement simple des images"),
        mo.hstack([mo.pyplot(plt)]),
        mo.ui.table(df_report_m1),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Sauvegarde des scores""")
    return


@app.cell
def _(
    accuracy_test,
    accuracy_tests,
    accuracy_validation,
    accuracy_validations,
    loss_test,
    loss_tests,
    loss_validation,
    loss_validations,
    time_prepare_method_1,
    time_training_method_1,
    training_times,
):
    training_times.append(time_prepare_method_1 + time_training_method_1)
    loss_validations.append(loss_validation)
    loss_tests.append(loss_test)
    accuracy_validations.append(accuracy_validation)
    accuracy_tests.append(accuracy_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Seconde approche""")
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
    data_1, data_test = train_test_split(df, test_size=0.15, random_state=42)
    return data_1, data_test


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
    data_1,
    data_test,
    image_data_generator_test,
    image_data_generator_train,
    time,
):
    _start_time = time.time()
    train_flow = image_data_generator_train.flow_from_dataframe(
        data_1,
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
        data_1,
        directory="",
        x_col="image",
        y_col="category",
        weight_col=None,
        target_size=(224, 224),
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=8,
        subset="validation",
    )
    test_flow = image_data_generator_test.flow_from_dataframe(
        data_test,
        directory="",
        x_col="image",
        y_col="category",
        weight_col=None,
        target_size=(224, 224),
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
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
def _(EarlyStopping, ModelCheckpoint, create_model, tf):
    with tf.device("/cpu:0"):
        _model2 = create_model()
    model2_save_path = "models/model2_best_weights.keras"
    model2_log_path = "models/model2_training_log"
    _checkpoint = ModelCheckpoint(model2_save_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    _early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)
    callbacks_list_1 = [_checkpoint, _early_stopping]
    return callbacks_list_1, model2_log_path, model2_save_path


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
    callbacks_list_1,
    create_model,
    load_model,
    model2_log_path,
    model2_save_path,
    np,
    os,
    test_flow,
    tf,
    time,
    train_flow,
):
    if os.path.exists(model2_save_path):
        model2_ = load_model(model2_save_path)
        history2 = np.load(model2_log_path + ".npy", allow_pickle="True").item()
    else:
        _start_time = time.time()
        with tf.device("/cpu:0"):
            model2 = create_model()
            history2 = model2.fit(
                train_flow, epochs=50, batch_size=64, callbacks=callbacks_list_1, validation_data=test_flow, verbose=1
            )
            np.save(model2_log_path, history2.history)
        time_training_method_2 = round(time.time() - _start_time, 0)
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
def _(model2, train_flow, validation_flow):
    loss_1, accuracy_1 = model2.evaluate(train_flow, verbose=True)
    print("Training Accuracy: {:.4f}".format(accuracy_1))
    print()
    loss_1, accuracy_1 = model2.evaluate(validation_flow, verbose=True)
    print("Validation Accuracy:  {:.4f}".format(accuracy_1))
    return


@app.cell
def _(
    X_test_preprocessed_2,
    X_val_preprocessed_2,
    mo,
    model1,
    model2,
    model2_save_path,
    y_test,
    y_val,
):
    with mo.persistent_cache("evaluate_whole_model2"):
        # Charger les poids du meilleur modèle
        model2.load_weights(model2_save_path)

        # Évaluation sur l'ensemble de validation
        val_loss_final_m2, val_accuracy_final_m2 = model2.evaluate(X_val_preprocessed_2, y_val, verbose=False)

        # Évaluation sur l'ensemble de test
        test_loss_m2, test_accuracy_m2 = model1.evaluate(X_test_preprocessed_2, y_test, verbose=False)

    print(f"Validation Accuracy       :  {val_accuracy_final_m2:.4f}")
    print(f"Validation Loss           :  {val_loss_final_m2:.4f}")
    print(f"Test Accuracy             :  {test_accuracy_m2:.4f}")
    print(f"Test Loss                 :  {test_loss_m2:.4f}")
    return


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


app._unparsable_cell(
    r"""

    _y_test_true = np.argmax(y_test, axis=1)
    _y_test_pred = np.argmax(model2.predict(X_test_1), axis=1)
    _conf_mat = confusion_matrix(_y_test_true, _y_test_pred)
    _df_conf_mat = pd.DataFrame(_conf_mat, index=[label for label in  _categories, columns=[_i for _i in \"0123456\"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(_df_conf_mat, annot=True, cmap=\"Blues\")
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Sauvegarde des scores""")
    return


@app.cell
def _(
    accuracy_test_1,
    accuracy_tests,
    accuracy_validation_1,
    accuracy_validations,
    loss_test_1,
    loss_tests,
    loss_validation_1,
    loss_validations,
    time_prepare_method_2,
    time_training_method_2,
    training_times,
):
    training_times.append(time_prepare_method_2 + time_training_method_2)
    loss_validations.append(loss_validation_1)
    loss_tests.append(loss_test_1)
    accuracy_validations.append(accuracy_validation_1)
    accuracy_tests.append(accuracy_test_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Troisième approche""")
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
def _(df, train_test_split):
    data_2, data_test_1 = train_test_split(df, test_size=0.15, random_state=42)
    return data_2, data_test_1


@app.cell
def _(data_2, data_test_1, os, shutil):
    path_train = "./data/images_train/"
    path_test = "./data/images_test/"
    if not os.path.exists(path_train):
        os.makedirs(path_train)
        for idx, ser in data_2.iterrows():
            if not os.path.exists(path_train + ser["category"]):
                os.makedirs(path_train + ser["category"])
            image_name = ser["image"].split("/")[-1]
            shutil.copy(ser["image"], path_train + ser["category"] + "/" + image_name)
    if not os.path.exists(path_test):
        os.makedirs(path_test)
        for idx, ser in data_test_1.iterrows():
            if not os.path.exists(path_test + ser["category"]):
                os.makedirs(path_test + ser["category"])
            image_name = ser["image"].split("/")[-1]
            shutil.copy(ser["image"], path_test + ser["category"] + "/" + image_name)
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
        seed=8,
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
        seed=8,
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
        seed=8,
        validation_split=0,
        subset=None,
    )
    time_prepare_method_3 = np.round(time.time() - _beginning_time, 0)
    return dataset_train, dataset_validation, time_prepare_method_3


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
def _(EarlyStopping, ModelCheckpoint, create_model, tf):
    with tf.device("/gpu:0"):
        model3 = create_model()
    model3_save_path = "./models/model3_best_weights.keras"
    model3_log_path = "./models/model3_training_log"
    _checkpoint = ModelCheckpoint(model3_save_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    _early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)
    callbacks_list_2 = [_checkpoint, _early_stopping]
    return callbacks_list_2, model3_log_path, model3_save_path


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
    callbacks_list_2,
    dataset_train,
    dataset_validation,
    load_model,
    model3_log_path,
    model3_save_path,
    np,
    os,
    tf,
    time,
):
    if os.path.exists(model3_save_path):
        model3_1 = load_model(model3_save_path)
        history3 = np.load(model3_log_path + ".npy", allow_pickle="True").item()
    else:
        _beginning_time = time.time()
        with tf.device("/gpu:0"):
            history3 = model3_1.fit(
                dataset_train,
                epochs=50,
                batch_size=64,
                callbacks=callbacks_list_2,
                validation_data=dataset_validation,
                verbose=1,
            )
            np.save(model3_log_path, history3.history)
        time_training_method_3 = np.round(time.time() - _beginning_time, 0)
    return history3, model3_1, time_training_method_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Scoring et analyse""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calculons à présent les scores et affichons les.""")
    return


@app.cell
def _(dataset_train, dataset_validation, model3_1):
    loss_2, accuracy_2 = model3_1.evaluate(dataset_train, verbose=True)
    print("Training Accuracy: {:.4f}".format(accuracy_2))
    print()
    loss_2, accuracy_2 = model3_1.evaluate(dataset_validation, verbose=True)
    print("Validation Accuracy:  {:.4f}".format(accuracy_2))
    return accuracy_2, loss_2


@app.cell
def _(
    X_test_1,
    X_val_1,
    accuracy_2,
    loss_2,
    model3_1,
    model3_save_path,
    y_test,
    y_val,
):
    model3_1.load_weights(model3_save_path)
    loss_validation_2, accuracy_validation_2 = model3_1.evaluate(X_val_1, y_val, verbose=False)
    print("Validation Accuracy       :  {:.4f}".format(accuracy_2))
    print("Validation Loss           :  {:.4f}".format(loss_2))
    loss_test_2, accuracy_test_2 = model3_1.evaluate(X_test_1, y_test, verbose=False)
    print("Test Accuracy             :  {:.4f}".format(accuracy_2))
    print("Test Loss                 :  {:.4f}".format(loss_2))
    return (
        accuracy_test_2,
        accuracy_validation_2,
        loss_test_2,
        loss_validation_2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualisons les scores du modèle durant son entrainement.""")
    return


@app.cell
def _(history3, plot_history, plt, show_history):
    # Plot the history of training
    show_history(history3)
    plot_history(history3, path="./models/model3_history.png")
    plt.close()
    return


@app.cell
def _(
    X_test_1,
    categories_list_1,
    confusion_matrix,
    model3_1,
    np,
    pd,
    plt,
    sns,
    y_test,
):
    _y_test_true = np.argmax(y_test, axis=1)
    _y_test_pred = np.argmax(model3_1.predict(X_test_1), axis=1)
    _conf_mat = confusion_matrix(_y_test_true, _y_test_pred)
    _df_conf_mat = pd.DataFrame(_conf_mat, index=[label for label in categories_list_1], columns=[_i for _i in "0123456"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(_df_conf_mat, annot=True, cmap="Blues")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Sauvegarde des scores""")
    return


@app.cell
def _(
    accuracy_test_2,
    accuracy_tests,
    accuracy_validation_2,
    accuracy_validations,
    loss_test_2,
    loss_tests,
    loss_validation_2,
    loss_validations,
    time_prepare_method_3,
    time_training_method_3,
    training_times,
):
    training_times.append(time_prepare_method_3 + time_training_method_3)
    loss_validations.append(loss_validation_2)
    loss_tests.append(loss_test_2)
    accuracy_validations.append(accuracy_validation_2)
    accuracy_tests.append(accuracy_test_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Quatrième approche""")
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

    - La première couche gérera la dat augmentation, avec :
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
def _(
    Dense,
    Dropout,
    EarlyStopping,
    GlobalAveragePooling2D,
    ModelCheckpoint,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    Rescaling,
    Sequential,
    VGG16,
    tf,
):
    with tf.device("/gpu:0"):
        data_augmentation = Sequential([
            RandomFlip("horizontal", input_shape=(224, 224, 3)),
            RandomRotation(0.1),
            RandomZoom(0.1),
        ])
        model4 = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        for layer in model4.layers:
            layer.trainable = False
        model4 = Sequential([
            data_augmentation,
            Rescaling(1.0 / 127.5, offset=-1),
            model4,
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(7, activation="softmax"),
        ])
        model4.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        print(model4.summary())
    model4_save_path = "./models/model4_best_weights.keras"
    model4_log_path = "./models/model4_training_log"
    _checkpoint = ModelCheckpoint(model4_save_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    _early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)
    callbacks_list_3 = [_checkpoint, _early_stopping]
    return callbacks_list_3, model4_log_path, model4_save_path


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
    callbacks_list_3,
    dataset_train,
    dataset_validation,
    load_model,
    model4_log_path,
    model4_save_path,
    np,
    os,
    tf,
    time,
):
    if os.path.exists(model4_save_path):
        model4_1 = load_model(model4_save_path)
        history4 = np.load(model4_log_path + ".npy", allow_pickle="True").item()
    else:
        _beginning_time = time.time()
        with tf.device("/gpu:0"):
            history4 = model4_1.fit(
                dataset_train,
                epochs=50,
                batch_size=64,
                callbacks=callbacks_list_3,
                validation_data=dataset_validation,
                verbose=1,
            )
            np.save(model4_log_path, history4.history)
        time_training_method_4 = np.round(time.time() - _beginning_time, 0)
    return history4, model4_1, time_training_method_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Scoring et analyse""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calculons à présent les scores et affichons les.""")
    return


@app.cell
def _(dataset_train, dataset_validation, model4_1):
    loss_3, accuracy_3 = model4_1.evaluate(dataset_train, verbose=True)
    print("Training Accuracy: {:.4f}".format(accuracy_3))
    print()
    loss_3, accuracy_3 = model4_1.evaluate(dataset_validation, verbose=True)
    print("Validation Accuracy:  {:.4f}".format(accuracy_3))
    return accuracy_3, loss_3


@app.cell
def _(
    X_test_1,
    X_val_1,
    accuracy_3,
    loss_3,
    model4_1,
    model4_save_path,
    y_test,
    y_val,
):
    model4_1.load_weights(model4_save_path)
    loss_validation_3, accuracy_validation_3 = model4_1.evaluate(X_val_1, y_val, verbose=False)
    print("Validation Accuracy       :  {:.4f}".format(accuracy_3))
    print("Validation Loss           :  {:.4f}".format(loss_3))
    loss_test_3, accuracy_test_3 = model4_1.evaluate(X_test_1, y_test, verbose=False)
    print("Test Accuracy             :  {:.4f}".format(accuracy_3))
    print("Test Loss                 :  {:.4f}".format(loss_3))
    return (
        accuracy_test_3,
        accuracy_validation_3,
        loss_test_3,
        loss_validation_3,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualisons les scores du modèle durant son entrainement.""")
    return


@app.cell
def _(history4, plot_history, plt, show_history):
    # Plot the history of training
    show_history(history4)
    plot_history(history4, path="./models/model4_history.png")
    plt.close()
    return


@app.cell
def _(
    X_test_1,
    categories_list_1,
    confusion_matrix,
    model4_1,
    np,
    pd,
    plt,
    sns,
    y_test,
):
    _y_test_true = np.argmax(y_test, axis=1)
    _y_test_pred = np.argmax(model4_1.predict(X_test_1), axis=1)
    _conf_mat = confusion_matrix(_y_test_true, _y_test_pred)
    _df_conf_mat = pd.DataFrame(_conf_mat, index=[label for label in categories_list_1], columns=[_i for _i in "0123456"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(_df_conf_mat, annot=True, cmap="Blues")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Sauvegarde des scores""")
    return


@app.cell
def _(
    accuracy_test_3,
    accuracy_tests,
    accuracy_validation_3,
    accuracy_validations,
    loss_test_3,
    loss_tests,
    loss_validation_3,
    loss_validations,
    time_prepare_method_3,
    time_training_method_4,
    training_times,
):
    training_times.append(time_prepare_method_3 + time_training_method_4)
    loss_validations.append(loss_validation_3)
    loss_tests.append(loss_test_3)
    accuracy_validations.append(accuracy_validation_3)
    accuracy_tests.append(accuracy_test_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Analyse des résultats""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Convertissons tout d'abord nos listes de scores dans un DataFrame pour en faciliter l'affichage""")
    return


@app.cell
def _(
    accuracy_tests,
    accuracy_validations,
    loss_tests,
    loss_validations,
    pd,
    training_times,
):
    scores_modeles = [loss_validations, loss_tests, accuracy_validations, accuracy_tests, training_times]

    scores = pd.DataFrame(
        scores_modeles,
        columns=["Méthode 1", "Méthode 2", "Méthode 3", "Méthode 4"],
        index=["loss_validations", "loss_tests", "accuracy_validations", "accuracy_tests", "training_times"],
    )
    scores = scores.T
    return (scores,)


@app.cell
def _(plt, scores, sns):
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=2)

    fig_ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(data=scores, x=scores.index, y="loss_validations", label="Entropie croisée set validation")
    sns.barplot(data=scores, x=scores.index, y="loss_tests", label="Entropie croisée set test", alpha=0.5)
    plt.title("Comparaison entropie croisée validation/test", fontweight="bold")
    plt.legend(loc="lower left")
    plt.ylabel("Entropie croisée")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Méthode")

    fig_ax1 = fig.add_subplot(gs[0, 1])
    sns.barplot(data=scores, x=scores.index, y="accuracy_validations", label="Exactitude (accuracy) set validation")
    sns.barplot(data=scores, x=scores.index, y="accuracy_tests", label="Exactitude (accuracy) set test", alpha=0.5)
    plt.title("Comparaison exactitude validation/test", fontweight="bold")
    plt.legend(loc="lower left")
    plt.ylabel("Exactitude")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Méthode")

    fig_ax1 = fig.add_subplot(gs[1, :])
    sns.barplot(data=scores, x=scores.index, y="training_times", label="Temps d'entrainement")
    plt.title("Temps d'entrainement des modèles", fontweight="bold")
    plt.ylabel("Temps d'entrainement (s)")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Méthode")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On constate que les modèles ont une exactitude similaire, cependant certains modèles se démarquent particulièrement avec les scores d'entropie croisée.

    Tout d'abord on remarque une énorme différence entre les scores d'entropie croisée de validation et de test pour les modèles 2 et 3, ce qui dénote de l'overfitting.

    Le modèle numéro 1 n'est pas très bon, avec une entropie croisée de quasiment le double du modèle n°4.

    Quand au modèle 4, celui montre un léger overfitting, cependant au vu de ses résultats il semble être le modèle le plus approprié ici, bien qu'il ait un temps d'entrainement près de 50% supérieur au modèle n°1.
    """
    )
    return


if __name__ == "__main__":
    app.run()
