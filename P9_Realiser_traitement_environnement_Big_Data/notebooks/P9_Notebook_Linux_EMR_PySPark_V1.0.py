import marimo

__generated_with = "0.17.8"
app = marimo.App(
    app_title="P9 -  Déployez un modèle dans le cloud",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Déployez un modèle dans le cloud
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Préambule

    ## 1.1 Problématique

    La très jeune start-up de l'AgriTech, nommée "**Fruits**!",
    cherche à proposer des solutions innovantes pour la récolte des fruits.

    La volonté de l’entreprise est de préserver la biodiversité des fruits
    en permettant des traitements spécifiques pour chaque espèce de fruits
    en développant des robots cueilleurs intelligents.

    La start-up souhaite dans un premier temps se faire connaître en mettant
    à disposition du grand public une application mobile qui permettrait aux
    utilisateurs de prendre en photo un fruit et d'obtenir des informations sur ce fruit.

    Pour la start-up, cette application permettrait de sensibiliser le grand public
    à la biodiversité des fruits et de mettre en place une première version du moteur
    de classification des images de fruits.

    De plus, le développement de l’application mobile permettra de construire
    une première version de l'architecture **Big Data** nécessaire.

    ## 1.2 Objectifs dans ce projet

    1. Développer une première chaîne de traitement des données qui
       comprendra le **preprocessing** et une étape de **réduction de dimension**.
    2. Tenir compte du fait que <u>le volume de données va augmenter
       très rapidement</u> après la livraison de ce projet, ce qui implique de:

    - Déployer le traitement des données dans un environnement **Big Data**
    - Développer les scripts en **pyspark** pour effectuer du **calcul distribué**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.3 Déroulement des étapes du projet

    Le projet va être réalisé en 2 temps, dans deux environnements différents.
    Nous allons dans un premier temps développer et exécuter notre code en local,
    en travaillant sur un nombre limité d'images à traiter.

    Une fois les choix techniques validés, nous déploierons notre solution
    dans un environnement Big Data en mode distribué.

    <u>Pour cette raison, ce projet sera divisé en 3 parties</u>:

    1. Liste des choix techniques généraux retenus
    2. Déploiement de la solution en local
    3. Déploiement de la solution dans le cloud
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Choix techniques généraux retenus
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    ## 2.1 Calcul distribué

    L'énoncé du projet nous impose de développer des scripts en **pyspark** 
    afin de <u>prendre en compte l'augmentation très rapide du volume 
    de donné après la livraison du projet</u>.

    Pour comprendre rapidement et simplement ce qu'est **pyspark** 
    et son principe de fonctionnement, nous vous conseillons de lire 
    cet article : [PySpark : Tout savoir sur la librairie Python](https://datascientest.com/pyspark)

    <u>Le début de l'article nous dit ceci </u>:
    « _Lorsque l'on parle de traitement de bases de données sur python, 
    on pense immédiatement à la librairie pandas. Cependant, lorsqu'on a 
    affaire à des bases de données trop massives, les calculs deviennent trop lents.
    Heureusement, il existe une autre librairie python, assez proche 
    de pandas, qui permet de traiter des très grandes quantités de données : PySpark.
    Apache Spark est un framework open-source développé par l'AMPLab 
    de UC Berkeley permettant de traiter des bases de données massives 
    en utilisant le calcul distribué, technique qui consiste à exploiter 
    plusieurs unités de calcul réparties en clusters au profit d'un seul 
    projet afin de diviser le temps d'exécution d'une requête.
    Spark a été développé en Scala et est au meilleur de ses capacités 
    dans son langage natif. Cependant, la librairie PySpark propose de 
    l'utiliser avec le langage Python, en gardant des performances 
    similaires à des implémentations en Scala.
    Pyspark est donc une bonne alternative à la librairie pandas lorsqu'on 
    cherche à traiter des jeux de données trop volumineux qui entraînent 
    des calculs trop chronophages._ »

    Comme nous le constatons, **pySpark** est un moyen de communiquer 
    avec **Spark** via le langage **Python**.
    **Spark**, quant à lui, est un outil qui permet de gérer et de coordonner 
    l'exécution de tâches sur des données à travers un groupe d'ordinateurs. 
    <u>Spark (ou Apache Spark) est un framework open source de calcul distribué 
    in-memory pour le traitement et l'analyse de données massives</u>.

    Un autre [article très intéressant et beaucoup plus complet pour 
    comprendre le **fonctionnement de Spark**](https://www.veonum.com/apache-spark-pour-les-nuls/), ainsi que le rôle 
    des **Spark Session** que nous utiliserons dans ce projet.

    <u>Voici également un extrait</u>:

    _Les applications Spark se composent d'un pilote (« driver process ») 
    et de plusieurs exécuteurs (« executor processes »). Il peut être configuré 
    pour être lui-même l'exécuteur (local mode) ou en utiliser autant que 
    nécessaire pour traiter l'application, Spark prenant en charge la mise 
    à l'échelle automatique par une configuration d'un nombre minimum 
    et maximum d'exécuteurs._

    {mo.image("public/spark-schema.png")}

    Le driver (parfois appelé « Spark Session ») distribue et planifie 
    les tâches entre les différents exécuteurs qui les exécutent et permettent 
    un traitement réparti. Il est le responsable de l'exécution du code 
    sur les différentes machines.

    Chaque exécuteur est un processus Java Virtual Machine (JVM) distinct 
    dont il est possible de configurer le nombre de CPU et la quantité de 
    mémoire qui lui est alloué. 
    Une seule tâche peut traiter un fractionnement de données à la fois.

    Dans les deux environnements (Local et Cloud) nous utiliserons donc **Spark** 
    et nous l'exploiterons à travers des scripts python grâce à **PySpark**.

    Dans la <u>version locale</u> de notre script nous **simulerons 
    le calcul distribué** afin de valider que notre solution fonctionne.
    Dans la <u>version cloud</u> nous **réaliserons les opérations sur un cluster de machine**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 Transfert Learning

    L'énoncé du projet nous demande également de
    réaliser une première chaîne de traitement
    des données qui comprendra le preprocessing et
    une étape de réduction de dimension.

    Il est également précisé qu'il n'est pas nécessaire
    d'entraîner un modèle pour le moment.

    Nous décidons de partir sur une solution de **transfert learning**.

    Simplement, le **transfert learning** consiste
    à utiliser la connaissance déjà acquise
    par un modèle entraîné (ici **MobileNetV2**) pour
    l'adapter à notre problématique.

    Nous allons fournir au modèle nos images, et nous allons
    <u>récupérer l'avant dernière couche</u> du modèle.
    En effet la dernière couche de modèle est une couche softmax
    qui permet la classification des images ce que nous ne
    souhaitons pas dans ce projet.

    L'avant dernière couche correspond à un **vecteur
    réduit** de dimension (1,1,1280).

    Cela permettra de réaliser une première version du moteur
    pour la classification des images des fruits.

    **MobileNetV2** a été retenu pour sa <u>rapidité d'exécution</u>,
    particulièrement adaptée pour le traitement d'un gros volume
    de données ainsi que la <u>faible dimensionnalité du vecteur
    de caractéristique en sortie</u> (1,1,1280)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Déploiement de la solution en local

    ## 3.1 Environnement de travail

    Pour des raisons de simplicité, nous développons dans un environnement
    Linux Unbuntu (exécuté depuis une machine Windows dans une machine virtuelle)

    - Pour installer une machine virtuelle : https://www.malekal.com/meilleurs-logiciels-de-machine-virtuelle-gratuits-ou-payants/

    ## 3.2 Installation de Spark

    [La première étape consiste à installer Spark ](https://computingforgeeks.com/how-to-install-apache-spark-on-ubuntu-debian/)

    Utilisation de pyspark 3.5.1
    java 17 doit etre installé sur le système

    ```bash
    # Installer Java 17
    sudo zypper install java-17-openjdk java-17-openjdk-devel

    # Vérifier que Java 17 est installé
    sudo update-alternatives --config java
    ```

    Vous devriez voir une liste comme :
    ```

    There are 3 choices for the alternative java (providing /usr/bin/java).

      Selection    Path                                    Priority   Status
    ------------------------------------------------------------
      0            /usr/lib64/jvm/jre-21-openjdk/bin/java   3105      auto mode
      1            /usr/lib64/jvm/jre-11-openjdk/bin/java   2105      manual mode
    * 2            /usr/lib64/jvm/jre-17-openjdk/bin/java   2705      manual mode
      3            /usr/lib64/jvm/jre-21-openjdk/bin/java   3105      manual mode
    ```

    ## 3.3 Installation des packages

    <u>On installe ensuite à l'aide de la commande **uv add**
    les packages qui nous seront nécessaires</u> :
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.4 Import des librairies
    """)
    return


@app.cell
def _():
    # Configuration environnement
    import os
    import warnings

    # TensorFlow - Mode silencieux
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Java pour Spark
    os.environ["JAVA_HOME"] = "/usr/lib64/jvm/java-11-openjdk-11"
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    # Masquer les warnings de retracing TensorFlow
    warnings.filterwarnings("ignore", message=".*tf.function retracing.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    return


@app.cell
def _():
    import tensorflow as tf

    # Configuration TensorFlow
    tf.get_logger().setLevel("ERROR")
    tf.config.set_visible_devices([], "GPU")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ✅ **Environnement configuré**
    - TensorFlow en mode CPU silencieux
    - Java 11 configuré pour Spark
    - Warnings minimisés
    """)
    return


@app.cell
def _(mo):
    import pandas as pd
    from PIL import Image
    import numpy as np
    import io
    from typing import Iterator

    from pathlib import Path

    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras import Model
    from pyspark.sql.functions import col, pandas_udf, element_at, split
    from pyspark.sql import SparkSession

    mo.md("✅ Configuration chargée")
    return (
        Image,
        Iterator,
        MobileNetV2,
        Path,
        SparkSession,
        col,
        element_at,
        img_to_array,
        io,
        np,
        pandas_udf,
        pd,
        preprocess_input,
        split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.5 Définition des PATH pour charger les images  et enregistrer les résultats

    Dans cette version locale nous partons du principe que les données
    sont stockées dans le même répertoire que le notebook.
    Nous n'utilisons qu'un extrait de **300 images** à traiter dans cette
    première version en local.
    L'extrait des images à charger est stockée dans le dossier **Test1**.
    Nous enregistrerons le résultat de notre traitement
    dans le dossier "**Results_Local**"
    """)
    return


@app.cell
def _(Path):
    PATH = Path.cwd()
    PATH_Data = PATH / "data" / "Test1"
    PATH_Result = PATH / "data" / "Results"

    print(f"""PATH: {PATH}
    PATH_Data: {PATH_Data}
    PATH_Result: {PATH_Result}""")
    return PATH_Data, PATH_Result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.6 Création de la SparkSession

    L’application Spark est contrôlée grâce à un processus de pilotage (driver process) appelé **SparkSession**.
    <u>Une instance de **SparkSession** est la façon dont Spark exécute les fonctions définies par l’utilisateur
    dans l’ensemble du cluster</u>. <u>Une SparkSession correspond toujours à une application Spark</u>.

    <u>Ici nous créons une session spark en spécifiant dans l'ordre</u> :

    1.  un **nom pour l'application**, qui sera affichée dans l'interface utilisateur Web Spark "**P9**"
    2.  que l'application doit s'exécuter **localement**.
        Nous ne définissons pas le nombre de cœurs à utiliser (comme .master('local[4]) pour 4 cœurs à utiliser),
        nous utiliserons donc tous les cœurs disponibles dans notre processeur.
    3.  une option de configuration supplémentaire permettant d'utiliser le **format "parquet"**
        que nous utiliserons pour enregistrer et charger le résultat de notre travail.
    4.  vouloir **obtenir une session spark** existante ou si aucune n'existe, en créer une nouvelle
    """)
    return


@app.cell
def _(SparkSession, mo):
    spark = (
        SparkSession.builder.appName("FruitsClassification")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.parquet.writeLegacyFormat", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "256")
        .getOrCreate()
    )

    mo.md(f"✅ **Spark {spark.version} initialisé**")
    return (spark,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <u>Nous créons également la variable "**sc**" qui est un **SparkContext** issue de la variable **spark**</u> :
    """)
    return


@app.cell
def _(spark):
    # Réduire les logs Spark
    spark.sparkContext.setLogLevel("ERROR")
    return


@app.cell
def _(spark):
    sc = spark.sparkContext
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <u>Affichage des informations de Spark en cours d'execution</u> :
    """)
    return


@app.cell
def _(spark):
    spark
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.7 Traitement des données

    <u>Dans la suite de notre flux de travail,
    nous allons successivement</u> :

    1. Préparer nos données
       1. Importer les images dans un dataframe **pandas UDF**
       2. Associer aux images leur **label**
       3. Préprocesser en **redimensionnant nos images pour
          qu'elles soient compatibles avec notre modèle**
    2. Préparer notre modèle
       1. Importer le modèle **MobileNetV2**
       2. Créer un **nouveau modèle** dépourvu de la dernière couche de MobileNetV2
    3. Définir le processus de chargement des images et l'application
       de leur featurisation à travers l'utilisation de pandas UDF
    4. Exécuter les actions d'extraction de features
    5. Enregistrer le résultat de nos actions
    6. Tester le bon fonctionnement en chargeant les données enregistrées
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Copie d'un échantillon de 300 images pour développer le notebook localement.

    ```python
    import shutil
    import random
    from pathlib import Path

    DATASET_PATH = Path("data/raw/fruits/fruits-360_dataset/fruits-360/Training")
    TEST1_PATH = Path("data/Test1")
    IMAGES_PER_CLASS = 20  # 20 images par classe
    N_CLASSES = 15         # 15 classes = 300 images

    TEST1_PATH.mkdir(parents=True, exist_ok=True)

    # Récupérer toutes les classes (dossiers)
    all_classes = [d for d in DATASET_PATH.iterdir() if d.is_dir()]
    selected_classes = random.sample(all_classes, min(N_CLASSES, len(all_classes)))

    total_copied = 0

    for fruit_class in selected_classes:
        # Créer le sous-dossier pour la classe dans TEST1_PATH
        class_dest = TEST1_PATH / fruit_class.name
        class_dest.mkdir(parents=True, exist_ok=True)

        images = list(fruit_class.glob("*.jpg"))
        selected = random.sample(images, min(IMAGES_PER_CLASS, len(images)))

        for img in selected:
            shutil.copy2(img, class_dest / img.name)
            total_copied += 1

        print(f"✓ {fruit_class.name}: {len(selected)} images")

    print(f"\n✅ Total: {total_copied} images dans {TEST1_PATH}")
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.7.1 Chargement des données

    Les images sont chargées au format binaire, ce qui offre,
    plus de souplesse dans la façon de prétraiter les images.

    Avant de charger les images, nous spécifions que nous voulons charger
    uniquement les fichiers dont l'extension est **jpg**.

    Nous indiquons également de charger tous les objets possibles contenus
    dans les sous-dossiers du dossier communiqué.
    """)
    return


@app.cell
def _(PATH_Data, mo, spark):
    images_raw = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(str(PATH_Data))
    )

    mo.md(f"✅ {images_raw.count()} images chargées")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <u>Affichage des 5 premières images contenant</u> :

    - le path de l'image
    - la date et heure de sa dernière modification
    - sa longueur
    - son contenu encodé en valeur hexadécimal
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Je ne conserve que le **path** de l'image et j'ajoute
    une colonne contenant les **labels** de chaque image :
    """)
    return


@app.cell
def _(PATH_Data, element_at, mo, spark, split):
    images = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(str(PATH_Data))
        .withColumn("label", element_at(split("path", "/"), -2))
    )

    # Afficher les résultats
    images.printSchema()
    sample = images.select("path", "label").limit(5).toPandas()

    mo.md(f"✅ **{images.count()} images** avec labels extraits")
    mo.ui.table(sample, selection=None)
    return (images,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ### 3.7.2 Préparation du modèle

    Je vais utiliser la technique du **transfert learning** pour extraire les features des images.
    J'ai choisi d'utiliser le modèle **MobileNetV2** pour sa rapidité d'exécution comparée 
    à d'autres modèles comme _VGG16_ par exemple.

    Pour en savoir plus sur la conception et le fonctionnement de MobileNetV2, 
    je vous invite à lire [cet article](https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c).

    <u>Voici le schéma de son architecture globale</u> :

    {mo.image("public/mobilenetv2_architecture.png", caption="Architecture de MobileNetV2")}

    Il existe une dernière couche qui sert à classer les images 
    selon 1000 catégories que nous ne voulons pas utiliser.
    L'idée dans ce projet est de récupérer le **vecteur de caractéristiques 
    de dimensions (1,1,1280)** qui servira, plus tard, au travers d'un moteur 
    de classification à reconnaitre les différents fruits du jeu de données.

    Comme d'autres modèles similaires, **MobileNetV2**, lorsqu'on l'utilise 
    en incluant toutes ses couches, attend obligatoirement des images 
    de dimension (224,224,3). Nos images étant toutes de dimension (100,100,3), 
    nous devrons simplement les **redimensionner** avant de les confier au modèle.

    <u>Dans l'odre</u> :

    1.  Nous chargeons le modèle **MobileNetV2** avec les poids **précalculés** 
        issus d'**imagenet** et en spécifiant le format de nos images en entrée
    2.  Nous créons un nouveau modèle avec:

    - <u>en entrée</u> : l'entrée du modèle MobileNetV2
    - <u>en sortie</u> : l'avant dernière couche du modèle MobileNetV2
    """)
    return


@app.cell
def _():
    # model = MobileNetV2(weights='imagenet',
    #                     include_top=True,
    #                     input_shape=(224, 224, 3))
    return


@app.cell
def _():
    # new_model = Model(inputs=model.input,
    #                   outputs=model.layers[-2].output)
    return


@app.cell
def _(MobileNetV2, mo):
    # Créer le feature extractor
    feature_extractor = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg")

    # Désactiver l'entraînement (important pour l'inférence)
    feature_extractor.trainable = False

    mo.md(f"""
    ## 🧠 Feature Extractor MobileNetV2

    **Configuration** :
    - Architecture : MobileNetV2 (ImageNet)
    - Input : {feature_extractor.input_shape}
    - Output : {feature_extractor.output_shape} (vecteur de features)
    - Paramètres : {feature_extractor.count_params():,}
    """)
    return (feature_extractor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Affichage du résumé de notre nouveau modèle où nous constatons
    que <u>nous récupérons bien en sortie un vecteur de dimension (1, 1, 1280)</u> :
    """)
    return


@app.cell
def _(feature_extractor):
    feature_extractor.summary()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Broadcast des poids

    Les workers doivent accéder au modèle et ses poids.

    **Bonne pratique** : Charger sur le driver, broadcaster les poids aux workers.
    """)
    return


@app.cell
def _(feature_extractor, mo, spark):
    broadcast_weights = spark.sparkContext.broadcast(feature_extractor.get_weights())

    mo.md(f"""
    ✅ **Poids broadcastés**
    - Taille approximative : {sum(w.nbytes for w in feature_extractor.get_weights()) / 1024 / 1024:.2f} MB
    - Nombre de tenseurs : {len(feature_extractor.get_weights())}
    """)
    return (broadcast_weights,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <u>Mettons cela sous forme de fonction</u> :
    """)
    return


@app.cell
def _():
    # def model_fn():
    #     """
    #     Returns a MobileNetV2 model with top layer removed
    #     and broadcasted pretrained weights.
    #     """
    #     model = MobileNetV2(weights='imagenet',
    #                         include_top=True,
    #                         input_shape=(224, 224, 3))
    #     for layer in model.layers:
    #         layer.trainable = False
    #     new_model = Model(inputs=model.input,
    #                   outputs=model.layers[-2].output)
    #     new_model.set_weights(brodcast_weights.value)
    #     return new_model
    return


@app.cell
def _(Path, feature_extractor, mo):
    from tensorflow.keras.utils import plot_model

    # Créer le dossier pour les visualisations
    viz_path = Path("notebooks/public")
    viz_path.mkdir(exist_ok=True)

    # Générer différentes visualisations

    # 1. Architecture complète
    full_arch_path = viz_path / "mobilenet_full_architecture.png"
    plot_model(
        feature_extractor,
        to_file=str(full_arch_path),
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        rankdir="TB",  # Vertical
        expand_nested=True,
        dpi=150,
    )

    mo.md(f"""
    ## 🏗️ Architecture MobileNetV2

    Visualisation de l'architecture du modèle utilisé pour l'extraction de features.
    """)
    return (full_arch_path,)


@app.cell
def _(full_arch_path, mo):
    # Afficher l'image dans Marimo
    mo.image(str(full_arch_path), width=600)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.7.3 Définition des fonctions de preprocessing et featurisation

    **Architecture** :
    1. `preprocess()` : Redimensionner et normaliser une image
    2. `featurize_series()` : Extraire features d'un batch d'images
    3. `featurize_udf()` : Pandas UDF pour distribution Spark
    """)
    return


@app.cell
def _(
    Image,
    MobileNetV2,
    broadcast_weights,
    img_to_array,
    io,
    mo,
    np,
    pd,
    preprocess_input,
):
    # Fonction de prétraitement d'une image
    def preprocess(content):
        """
        Prétraite les bytes d'une image pour la prédiction.

        Args:
            content: bytes de l'image

        Returns:
            array numpy (224, 224, 3) normalisé
        """
        # Ouvre l'image à partir des données binaires et la redimensionne à 224x224 pixels
        img = Image.open(io.BytesIO(content)).resize([224, 224])
        # Convertit l'image PIL en tableau numpy
        arr = img_to_array(img)
        # Applique le prétraitement spécifique à MobileNetV2 (normalisation)
        return preprocess_input(arr)


    # Fonction d'extraction de caractéristiques sur une série d'images
    def featurize_series(model, content_series):
        """
        Extrait les features d'une pd.Series d'images.

        Args:
            model: Modèle TensorFlow
            content_series: pd.Series de bytes d'images

        Returns:
            pd.Series de vecteurs de features (1280,)
        """
        # Applique le prétraitement à chaque image de la série
        input_batch = np.stack(content_series.map(preprocess))
        # Fait la prédiction (extraction de features) sur le batch d'images
        preds = model.predict(input_batch, verbose=0)
        # Aplatir les features en vecteurs 1D
        output = [p.flatten() for p in preds]
        return pd.Series(output)


    # Construction du modèle MobileNetV2 sans la couche de classification
    def model_fn():
        """
        Recrée le modèle MobileNetV2 avec les poids broadcastés.
        Appelé une fois par worker.

        Returns:
            Modèle MobileNetV2 configuré
        """
        # Charge le modèle de base avec les poids ImageNet, sans la couche finale
        model = MobileNetV2(
            weights=None,  # Pas de téléchargement sur les workers
            include_top=False,
            input_shape=(224, 224, 3),
            pooling="avg",
        )
        model.trainable = False
        model.set_weights(broadcast_weights.value)  # Charger les poids broadcastés
        # Retourne un modèle avec l'entrée et la sortie du modèle de base
        return model


    mo.md("✅ Fonctions de preprocessing définies")
    return featurize_series, model_fn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    L'utilité des Users Define Fonctions

    Les UDF (fonctions définies par l'utilisateur) permettent d’appliquer une logique personnalisée à des colonnes de DataFrames PySpark lorsque les fonctions natives ne suffisent pas.
    """).callout("info")
    return


@app.cell
def _(Iterator, featurize_series, mo, model_fn, pandas_udf, pd):
    @pandas_udf("array<float>")
    def featurize_udf(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        """
        Pandas UDF Scalar Iterator pour featuriser des images.

        Le modèle est chargé une fois par worker et réutilisé pour tous les batches,
        ce qui amortit le coût de chargement.

        Args:
            content_series_iter: Iterator sur des batches de pd.Series d'images

        Yields:
            pd.Series de vecteurs de features
        """
        # Charger le modèle une fois par worker (pas par batch)
        model = model_fn()

        # Traiter chaque batch
        for content_series in content_series_iter:
            yield featurize_series(model, content_series)


    mo.md("✅ Pandas UDF définie")
    return (featurize_udf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.7.4 Exécution des actions d'extraction de features

    Les Pandas UDF, sur de grands enregistrements (par exemple, de très grandes images),
    peuvent rencontrer des erreurs de type Out Of Memory (OOM).
    Si vous rencontrez de telles erreurs dans la cellule ci-dessous,
    essayez de réduire la taille du lot Arrow via 'maxRecordsPerBatch'

    Je n'utiliserai pas cette commande dans ce projet
    et je laisse donc la commande en commentaire.

    **Configuration Arrow** : Limite la taille des batches pour éviter les OOM.
    - Valeur par défaut : 10 000 enregistrements
    - Ajustée à : 1 024 pour de grandes images
    """)
    return


@app.cell
def _():
    # # Configuration Arrow (IMPORTANT pour éviter OOM)
    # spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")
    # spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    # mo.md("✅ Configuration Arrow activée (batches de 1024 images max)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nous pouvons maintenant exécuter la featurisation sur l'ensemble de notre DataFrame Spark.
    <u>REMARQUE</u> : Cela peut prendre beaucoup de temps, tout dépend du volume de données à traiter.

    Notre jeu de données de **Test** contient **22819 images**.
    Cependant, dans l'exécution en mode **local**,
    nous <u>traiterons un ensemble réduit de **300 images**</u>.
    """)
    return


@app.cell(hide_code=True)
def _(images, mo):
    mo.md(f"""
    ### Extraction des features

    Dataset : **{images.count()} images** à traiter

    **Stratégie** :
    - Repartition en 20 partitions pour parallélisme
    - Extraction via Pandas UDF
    - Sauvegarde en format Parquet
    """)
    return


@app.cell
def _(col, featurize_udf, images):
    # Extraction des features
    features_df = images.repartition(20).select(col("path"), col("label"), featurize_udf("content").alias("features"))
    return (features_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <u>Rappel du PATH où seront inscrits les fichiers au format "**parquet**"
    contenant nos résultats, à savoir, un DataFrame contenant 3 colonnes</u> :

    1.  Path des images
    2.  Label de l'image
    3.  Vecteur de caractéristiques de l'image
    """)
    return


@app.cell(hide_code=True)
def _(PATH_Result, mo):
    mo.md(f"""
    ✅ **Pipeline de features configuré**
    - Output : `{PATH_Result}`
    - Colonnes : path, label, features (1280 floats)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <u>Enregistrement des données traitées au format "**parquet**"</u> :
    """)
    return


@app.cell
def _(PATH_Result, features_df, mo):
    # Sauvegarder en Parquet
    features_df.write.mode("overwrite").parquet(str(PATH_Result))

    mo.md(f"✅ **Features sauvegardées** dans `{PATH_Result}`")
    return


@app.cell
def _(mo):
    mo.md(r"""
    À ce stade, nous avons réalisé les étapes suivantes :

    - Chargé le modèle MobileNetV2 sans la couche de classification (include_top=False, pooling='avg')

    - Prétraité les images en les redimensionnant à (224, 224) et en appliquant la normalisation attendue par le modèle

    - Converti chaque image en un vecteur de caractéristiques de dimension (1280,)

    - Utilisé une Pandas UDF pour paralléliser l’extraction des features à l’aide de Spark

    - Réparti les données en 20 partitions pour exploiter le parallélisme

    - Limité l’échantillon à 330 images pour faciliter les tests en mode local

    - Enregistré les résultats au format parquet à l’emplacement spécifié par PATH_Result

    - Le DataFrame final contient 3 colonnes :
        - path : chemin d'accès de l'image

        - label : étiquette (classe) de l'image

        - features : vecteur de caractéristiques (extrait par le modèle)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.8 Chargement des données enregistrées et validation du résultat

    <u>On charge les données fraichement enregistrées dans un **DataFrame Pandas**</u> :
    """)
    return


@app.cell
def _(PATH_Result, pd):
    # Charger et valider
    df_results = pd.read_parquet(PATH_Result, engine="pyarrow")
    return (df_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <u>On affiche les 5 premières lignes du DataFrame</u> :
    """)
    return


@app.cell
def _(df_results):
    df_results.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <u>On valide que la dimension du vecteur de caractéristiques des images est bien de dimension 1280</u> :
    """)
    return


@app.cell
def _(df_results, mo):
    mo.md(f"""
    Dimensions du vecteur des caractéristiques {df_results.loc[0, 'features'].shape[0]}
    """)
    return


@app.cell
def _(df_results):
    # Vérification que toutes les lignes ont bien 1280 dimensions
    assert all(df_results["features"].apply(lambda x: len(x) == 1280)), "Certaines lignes ont une mauvaise dimension"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nous avons validé que :

    - Le fichier Parquet est lisible

    - Les vecteurs extraits ont bien la taille de 1280 flottants

    - Le format de données est prêt pour les prochaines étapes : réduction de dimension, visualisation ou entraînement d’un modèle de classification

    - Nous pouvons maintenant passer à la section suivante : réduction de dimension par PCA avec PySpark.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.9 Réduction de dimension PCA en PySpark

    Les vecteurs de 1280 dimensions extraits via MobileNetV2, bien qu’informatifs, posent des défis en termes de :

    - Performance (traitements lourds à l’échelle),
    - Bruit (dimensions redondantes ou peu pertinentes),
    - Visualisation (difficile en haute dimension).

    Solution : Application d’un PCA (50 composantes) via PySpark pour :

    - Conserver l’essentiel de la variance,
    - Accélérer les calculs,
    - Permettre une projection visuelle (2D/3D).
    """)
    return


@app.cell
def _(PATH_Result, mo, spark):
    from pyspark.ml.feature import PCA
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.sql.functions import udf

    # 1. Recharger le DataFrame
    df_spark = spark.read.parquet(str(PATH_Result))

    mo.md(f"📥 **{df_spark.count()} images** chargées depuis Parquet")
    return PCA, VectorUDT, Vectors, df_spark, udf


@app.cell
def _(VectorUDT, Vectors, col, df_spark, mo, udf):
    # 2. Convertir en vecteurs MLlib
    vectorize_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
    df_with_vectors = df_spark.withColumn("features_vec", vectorize_udf(col("features")))

    mo.md("✅ **Features converties** en vecteurs MLlib (compatibles PCA)")
    return (df_with_vectors,)


@app.cell
def _(PCA, df_with_vectors, mo, np):
    # 3. Appliquer PCA
    n_components = 150

    pca = PCA(k=n_components, inputCol="features_vec", outputCol="features_pca")
    pca_model = pca.fit(df_with_vectors)
    df_pca = pca_model.transform(df_with_vectors)

    # Variance expliquée
    explained_variance = pca_model.explainedVariance.toArray()
    cumulative_variance = np.cumsum(explained_variance)

    mo.md(f"""
    ## 📊 PCA - Résultats

    **Configuration** :
    - Dimensions originales : **1280**
    - Dimensions PCA : **{n_components}**
    - Réduction : **{(1 - n_components / 1280) * 100:.1f}%**

    **Variance expliquée** :
    - Par les 50 composantes : **{cumulative_variance[-1] * 100:.2f}%**
    - Par les 10 premières : **{cumulative_variance[9] * 100:.2f}%**
    - Par les 5 premières : **{cumulative_variance[4] * 100:.2f}%**
    """)
    return cumulative_variance, df_pca, explained_variance, n_components


@app.cell
def _(df_pca, mo):
    # 4. Visualisation : Échantillon du résultat
    sample_pca = df_pca.select("path", "label", "features_pca").limit(300).toPandas()

    # Convertir les vecteurs en listes pour l'affichage
    sample_pca["features_pca_preview"] = sample_pca["features_pca"].apply(
        lambda v: str(v.toArray()[:5]) + "..."  # Afficher les 5 premières valeurs
    )

    display_df = sample_pca[["path", "label", "features_pca_preview"]]

    mo.md("### 📋 Échantillon des features après PCA")
    mo.ui.table(display_df)
    return (sample_pca,)


@app.cell
def _(plt, sample_pca):
    # Extraire les deux premières composantes
    _x = sample_pca["features_pca"].apply(lambda _x: _x[0])
    _y = sample_pca["features_pca"].apply(lambda _x: _x[1])

    # Affichage en scatter plot avec la couleur selon le label
    plt.figure(figsize=(8, 6))
    for label in sample_pca["label"].unique():
        plt.scatter(_x[sample_pca["label"] == label], _y[sample_pca["label"] == label], label=label, alpha=0.6)

    plt.title("Projection PCA des images (2 premières dimensions)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(cumulative_variance, explained_variance, mo, n_components):
    # 5. Graphique de la variance expliquée
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graphique 1 : Variance par composante
    ax1.bar(range(1, n_components + 1), explained_variance, alpha=0.7, color="steelblue")
    ax1.set_xlabel("Composante PCA", fontsize=12)
    ax1.set_ylabel("Variance expliquée", fontsize=12)
    ax1.set_title("Variance expliquée par composante", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Graphique 2 : Variance cumulée
    ax2.plot(
        range(1, n_components + 1), cumulative_variance * 100, marker="o", linewidth=2, markersize=4, color="darkgreen"
    )
    ax2.axhline(y=95, color="r", linestyle="--", alpha=0.7, label="95% variance")
    ax2.set_xlabel("Nombre de composantes", fontsize=12)
    ax2.set_ylabel("Variance cumulée (%)", fontsize=12)
    ax2.set_title("Variance cumulée", fontsize=14, fontweight="bold")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    mo.md("### 📈 Analyse de la variance")
    mo.mpl.interactive(fig)
    return (plt,)


@app.cell
def _(cumulative_variance, explained_variance, mo, n_components, np):
    # Calculer les composantes nécessaires avec gestion des cas non atteints
    def find_n_components(cumulative_variance, threshold):
        """Trouve le nombre de composantes nécessaires ou retourne None si non atteint"""
        if np.any(cumulative_variance >= threshold):
            return np.argmax(cumulative_variance >= threshold) + 1
        else:
            return None


    n_for_95 = find_n_components(cumulative_variance, 0.95)
    n_for_99 = find_n_components(cumulative_variance, 0.99)

    # Variance actuelle avec n_components
    current_variance = cumulative_variance[-1]

    mo.md(f"""
    ### 📊 Statistiques PCA (k={n_components})

    **Top 10 composantes** (variance expliquée) :

    | Composante | Variance | Cumulée |
    |-----------|----------|---------|
    {
        chr(10).join([
            f"| PC{i + 1} | {explained_variance[i] * 100:.2f}% | {cumulative_variance[i] * 100:.2f}% |"
            for i in range(min(10, n_components))
        ])
    }

    **Analyse de la variance** :

    - Variance totale conservée avec {n_components} composantes : **{current_variance * 100:.2f}%**
    - Variance perdue : **{(1 - current_variance) * 100:.2f}%**

    **Recommandations** :

    - Pour conserver **95% de la variance** : {
        f"**{n_for_95} composantes**" if n_for_95 else f"**Plus de {n_components} composantes nécessaires**"
    }
    - Pour conserver **99% de la variance** : {
        f"**{n_for_99} composantes**"
        if n_for_99
        else f"**Plus de {n_components} composantes nécessaires** (estimé ~250-300)"
    }

    **Conclusion** : Avec **k={n_components}**, on atteint **{current_variance * 100:.2f}%** de variance, ce qui est {
        "✅ excellent (> 95%)" if current_variance >= 0.95 else "⚠️ acceptable mais < 95%"
    }
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nous venons de valider le processus sur un jeu de données allégé en local
    où nous avons simulé un cluster de machines en répartissant la charge de travail
    sur différents cœurs de processeur au sein d'une même machine.

    Nous allons maintenant généraliser le processus en déployant notre solution
    sur un réel cluster de machines et nous travaillerons désormais sur la totalité
    des 22819 images de notre dossier "Test".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Déploiement de la solution sur le cloud

    Maintenant que nous avons vérifié que notre solution fonctionne,
    il est temps de la <u>déployer à plus grande échelle sur un vrai cluster de machines</u>.

    **Attention**, _je travaille sous Linux avec une version Ubuntu,
    les commandes décrites ci-dessous sont donc réalisées
    exclusivement dans cet environnement._

    <u>Plusieurs contraintes se posent</u> :

    1.  Quel prestataire de Cloud choisir ?
    2.  Quelles solutions de ce prestataire adopter ?
    3.  Où stocker nos données ?
    4.  Comment configurer nos outils dans ce nouvel environnement ?

    ## 4.1 Choix du prestataire cloud : AWS

    Le prestataire le plus connu et qui offre à ce jour l'offre
    la plus large dans le cloud computing est **Amazon Web Services** (AWS).
    Certaines de leurs offres sont parfaitement adaptées à notre problématique
    et c'est la raison pour laquelle j'utiliserai leurs services.

    L'objectif premier est de pouvoir, grâce à AWS, <u>louer de la puissance de calcul à la demande</u>.
    L'idée étant de pouvoir, quel que soit la charge de travail,
    obtenir suffisamment de puissance de calcul pour pouvoir traiter nos images,
    même si le volume de données venait à fortement augmenter.

    De plus, la capacité d'utiliser cette puissance de calcul à la demande
    permet de diminuer drastiquement les coûts si l'on compare les coûts d'une location
    de serveur complet sur une durée fixe (1 mois, 1 année par exemple).

    ## 4.2 Choix de la solution technique : EMR

    <u>Plusieurs solutions s'offre à nous</u> :

    1. Solution **IAAS** (Infrastructure AS A Service)

    - Dans cette configuration **AWS** met à notre disposition des serveurs vierges
      sur lequel nous avons un accès en administrateur, ils sont nommés **instance EC2**.
      Pour faire simple, nous pouvons avec cette solution reproduire pratiquement
      à l'identique la solution mis en œuvre en local sur notre machine.
      <u>On installe nous-même l'intégralité des outils puis on exécute notre script</u> :
    - Installation de **Spark**, **Java** etc.
    - Installation de **Python** (via Anaconda par exemple)
    - Installation de **Jupyter Notebook**
    - Installation des **librairies complémentaires**
    - Il faudra bien évidement veiller à **implémenter les librairies
      nécessaires à toutes les machines (workers) du cluster**
    - <u>Avantages</u> :
      - Liberté totale de mise en œuvre de la solution
      - Facilité de mise en œuvre à partir d'un modèle qui s'exécute en local sur une machine Linux
    - <u>Inconvénients</u> :
      - Cronophage
        - Nécessité d'installer et de configurer toute la solution
      - Possible problèmes techniques à l'installation des outils (des problématiques qui
        n'existaient pas en local sur notre machine peuvent apparaitre sur le serveur EC2)
      - Solution non pérenne dans le temps, il faudra veiller à la mise à jour des outils
        et éventuellement devoir réinstaller Spark, Java etc.

    2. Solution **PAAS** (Plateforme As A Service)

    - **AWS** fournit énormément de services différents, dans l'un de ceux-là
      il existe une offre qui permet de louer des **instances EC2**
      avec des applications préinstallées et configurées : il s'agit du **service EMR**.
    - **Spark** y sera déjà installé
    - Possibilité de demander l'installation de **Tensorflow** ainsi que **JupyterHub**
    - Possibilité d'indiquer des **packages complémentaires** à installer
      à l'initialisation du serveur **sur l'ensemble des machines du cluster**.
    - <u>Avantages</u> :
      - Facilité de mise en œuvre
        - Il suffit de très peu de configuration pour obtenir
          un environnement parfaitement fonctionnel
      - Rapidité de mise en œuvre
        - Une fois la première configuration réalisée, il est très facile
          et très rapide de recréer des clusters à l'identique qui seront
          disponibles presque instantanément (le temps d'instancier les
          serveurs soit environ 15/20 minutes)
      - Solutions matérielless et logicielles optimisées par les ingénieurs d'AWS
        - On sait que les versions installées vont fonctionner
          et que l'architecture proposée est optimisée
      - Stabilité de la solution
      - Solution évolutive
        Il est facile d’obtenir à chaque nouvelle instanciation une version à jour
        de chaque package, en étant garanti de leur compatibilité avec le reste de l’environnement.
    - Plus sécurisé
      - Les éventuels patchs de sécurité seront automatiquement mis à jour
        à chaque nouvelle instanciation du cluster EMR.
    - <u>Inconvénients</u> :
      - Peut-être un certain manque de liberté sur la version des packages disponibles ?
        Même si je n'ai pas constaté ce problème.

    Je retiens la solution **PAAS** en choisissant d'utiliser
    le service **EMR** d'Amazon Web Services.
    Je la trouve plus adaptée à notre problématique et permet
    une mise en œuvre qui soit à la fois plus rapide et
    plus efficace que la solution IAAS.

    ## 4.3 Choix de la solution de stockage des données : Amazon S3

    <u>Amazon propose une solution très efficace pour la gestion du stockage des données</u> : **Amazon S3**.
    S3 pour Amazon Simple Storage Service.

    Il pourrait être tentant de stocker nos données sur l'espace alloué par le serveur **EC2**,
    mais si nous ne prenons aucune mesure pour les sauvegarder ensuite sur un autre support,
    <u>les données seront perdues</u> lorsque le serveur sera résilié (on résilie le serveur lorsqu'on
    ne s'en sert pas pour des raisons de coût).
    De fait, si l'on décide d'utiliser l'espace disque du serveur EC2 il faudra imaginer
    une solution pour sauvegarder les données avant la résiliation du serveur.
    De plus, nous serions exposés à certaines problématiques si nos données venaient à
    **saturer** l'espace disponible de nos serveurs (ralentissements, disfonctionnements).

    <u>Utiliser **Amazon S3** permet de s'affranchir de toutes ces problématiques</u>.
    L'espace disque disponible est **illimité**, et il est **indépendant de nos serveurs EC2**.
    L'accès aux données est **très rapide** car nous restons dans l'environnement d'AWS
    et nous prenons soin de <u>choisir la même région pour nos serveurs **EC2** et **S3**</u>.

    De plus, comme nous le verrons <u>il est possible d'accéder aux données sur **S3**
    de la même manière que l'on **accède aux données sur un disque local**</u>.
    Nous utiliserons simplement un **PATH au format s3://...** .

    ## 4.4 Configuration de l'environnement de travail

    La première étape est d'installer et de configurer [**AWS Cli**](https://aws.amazon.com/fr/cli/),
    il s'agit de l'**interface en ligne de commande d'AWS**.
    Elle nous permet d'**interagir avec les différents services d'AWS**, comme **S3** par exemple.

    Pour pouvoir utiliser **AWS Cli**, il faut le configurer en créant préalablement
    un utilisateur à qui on donnera les autorisations dont nous aurons besoin.
    Dans ce projet il faut que l'utilisateur ait à minima un contrôle total sur le service S3.

    <u>La gestion des utilisateurs et de leurs droits s'effectue via le service **AMI**</u> d'AWS.

    Une fois l'utilisateur créé et ses autorisations configurées nous créons une **paire de clés**
    qui nous permettra de nous **connecter sans à avoir à devoir saisir systématiquement notre login/mot de passe**.

    Il faut également configurer l'**accès SSH** à nos futurs serveurs EC2.
    Ici aussi, via un système de clés qui nous dispense de devoir nous authentifier "à la main" à chaque connexion.

    Toutes ses étapes de configuration sont parfaitement décrites
    dans le cours du projet: [Découvrez le cloud avec Amazon Web Services / Faites vos premiers pas sur AWS](https://openclassrooms.com/fr/courses/4810836-decouvrez-le-cloud-avec-amazon-web-services/7821712-faites-vos-premiers-pas-sur-aws)

    ## 4.5 Upload de nos données sur S3

    Nos outils sont configurés.
    Il faut maintenant uploader nos données de travail sur Amazon S3.

    Ici aussi les étapes sont décrites avec précision
    dans le cours [Découvrez le cloud avec Amazon Web Services / Stockez et accédez à des fichiers sur Amazon S3](https://openclassrooms.com/fr/courses/4810836-decouvrez-le-cloud-avec-amazon-web-services/7822690-stockez-et-accedez-a-des-fichiers-sur-amazon-s3)

    Je décide de n'uploader que les données contenues dans le dossier **Test** du [jeu de données du projet](https://www.kaggle.com/moltean/fruits/download)

    La première étape consiste à **créer un bucket sur S3**
    dans lequel nous uploaderons les données du projet:

    - ```aws s3 mb s3://p9-fruits-data```

    On vérifie que le bucket à bien été créé

    - ```aws s3 ls```
    - Si le nom du bucket s'affiche alors c'est qu'il a été correctement créé.

    On copie ensuite le contenu du dossier "**Test**"
    dans un répertoire "**Test**" sur notre bucket "**p9-fruits-data**":

    1. On se place à l'intérieur du répertoire **Test**
    2. ```aws s3 sync . s3://p9-fruits-data/Test```

    La commande **sync** est utile pour synchroniser deux répertoires.

    <u>Nos données du projet sont maintenant disponibles sur Amazon S3</u>.

    ## 4.6 Configuration du serveur EMR

    Une fois encore, le cours [Découvrez le cloud avec Amazon Web Services / Découvrez les services d'Amazon EC2](https://openclassrooms.com/fr/courses/4810836-decouvrez-le-cloud-avec-amazon-web-services/7822091-demarrez-votre-premiere-instance-ec2)  détaille l'essentiel des étapes pour lancer un cluster avec **EMR**.

    <u>Je détaillerai ici les étapes particulières qui nous permettent
    de configurer le serveur selon nos besoins</u> :

    1. Cliquez sur Créer un cluster

       {mo.image("public/EMR_creer.png",caption="Créer un cluster")}

    3. Cliquez sur Accéder aux options avancées

       {mo.image("public/EMR_options_avancees.png",caption="Créer un cluster")}

    ### 4.6.1 Étape 1 : Logiciels et étapes

    #### 4.6.1.1 Configuration des logiciels

    <u>Sélectionnez les packages dont nous aurons besoin comme dans la capture d'écran</u> :

    1. Nous sélectionnons la dernière version d'**EMR**, soit la version **6.3.0** au moment où je rédige ce document
    2. Nous cochons bien évidement **Hadoop** et **Spark** qui seront préinstallés dans leur version la plus récente
    3. Nous aurons également besoin de **TensorFlow** pour importer notre modèle et réaliser le **transfert learning**
    4. Nous travaillerons enfin avec un **notebook Marimo** via l'application **Marimo**

    - Comme nous le verrons dans un instant nous allons <u>paramétrer l'application afin que les notebooks</u>,
      comme le reste de nos données de travail, <u>soient enregistrés directement sur S3</u>.

      {mo.image("public/EMR_configuration_logiciels.png",caption="Créer un cluster")}

    #### 4.6.1.2 Modifier les paramètres du logiciel

    <u>Paramétrez la persistance des notebooks créés et ouvert via JupyterHub</u> :

    - On peut à cette étape effectuer des demandes de paramétrage particulières sur nos applications.
      L'objectif est, comme pour le reste de nos données de travail,
      d'éviter toutes les problématiques évoquées précédemment.
      C'est l'objectif à cette étape, <u>nous allons enregistrer
      et ouvrir les notebooks</u> non pas sur l'espace disque de l'instance EC2 (comme
      ce serait le cas dans la configuration par défaut de JupyterHub) mais
      <u>directement sur **Amazon S3**</u>.
    - <u>deux solutions sont possibles pour réaliser cela</u> :

    1.  Créer un **fichier de configuration JSON** que l'on **upload sur S3** et on indique ensuite le chemin d’accès au fichier JSON
    2.  Rentrez directement la configuration au format JSON

    J'ai personnellement créé un fichier JSON lors de la création de ma première instance EMR,
    puis lorsqu'on décide de cloner notre serveur pour en recréer un facilement à l'identique,
    la configuration du fichier JSON se retrouve directement copié comme dans la capture ci-dessous.

    <u>Voici le contenu de mon fichier JSON</u> :
    ```
    [{"classification":"jupyter-s3-conf","properties":{"s3.persistence.bucket":"p9-data","s3.persistence.enabled":"true"}}]
    ```

    Appuyez ensuite sur "**Suivant**"

    {mo.image("public/EMR_parametres_logiciel.png",caption="Modifier les paramètres du logiciel")}

    ### 4.6.2 Étape 2 : Matériel

    A cette étape, laissez les choix par défaut.
    <u>L'important ici est la sélection de nos instances</u> :

    1. je choisi les instances de type **M5** qui sont des **instances de type équilibrés**
    2. je choisi le type **xlarge** qui est l'instance la **moins onéreuse disponible**
       [Plus d'informations sur les instances M5 Amazon EC2](https://aws.amazon.com/fr/ec2/instance-types/m5/)
    3. Je sélectionne **1 instance Maître** (le driver) et **2 instances Principales** (les workeurs)
       soit **un total de 3 instance EC2**.

       {mo.image("public/EMR_materiel.png",caption="Choix du materiel")}

    ### 4.6.3 Étape 3 : Paramètres de cluster généraux

    #### 4.6.3.1 Options générales

    <u>La première chose à faire est de donner un nom au cluster</u> :
    _J'ai également décoché "Protection de la résiliation" pour des raisons pratiques._

    {mo.image("public/EMR_nom_cluster.png",caption="Nom du Cluster")}

    #### 4.6.3.2 Actions d'amorçage

    Nous allons à cette étape **choisir les packages manquants à installer** et qui
    nous serons utiles dans l'exécution de notre notebook.
    <u>L'avantage de réaliser cette étape maintenant est que les packages
    installés le seront sur l'ensemble des machines du cluster</u>.

    Nous créons un fichier nommé "**bootstrap-emr.sh**" que nous <u>uploadons
    sur S3</u>(je l’installe dans le dossier script de mon **bucket "p9-fruits-data"**) et nous l'ajoutons
    comme indiqué dans la capture d'écran ci-dessous:

    {mo.image("public/EMR_amorcage.png",caption="Actions d'amorcage")}

    Voici le contenu du fichier **bootstrap-emr.sh**
    Comme on peut le constater il s'agit simplement de commande "**pip install**"
    pour **installer les bibliothèques manquantes** comme réalisé en local.
    Une fois encore, <u>il est nécessaire de réaliser ces actions à cette étape</u>
    pour que <u>les packages soient installés sur l'ensemble des machines du cluster</u>
    et non pas uniquement sur le driver, comme cela serait le cas si nous exécutions
    ces commandes directement dans le notebook JupyterHub ou dans la console EMR (connecté au driver).

    {mo.image("public/EMR_bootstrap.png",caption="Contenu du fichier bootstrap")}

    **setuptools** et **pip** sont mis à jour pour éviter une problématique
    avec l'installation du package **pyarrow**.
    **Pandas** a eu droit à une mise à jour majeur (1.3.0) il y a moins d'une semaine
    au moment de la rédaction de ce notebook, et la nouvelle version de **Pandas**
    nécessite une version plus récente de **Numpy** que la version installée par
    défaut (1.16.5) à l'initialisation des instances **EC2**. <u>Il ne semble pas
    possible d'imposer une autre version de Numpy que celle installé par
    défaut</u> même si on force l'installation d'une version récente de **Numpy**
    (en tout cas, ni simplement ni intuitivement).
    La mise à jour étant très récente <u>la version de **Numpy** n'est pas encore
    mise à jour sur **EC2**</u> mais on peut imaginer que ce sera le cas très rapidement
    et il ne sera plus nécessaire d'imposer une version spécifique de **Pandas**.
    En attendant, je demande <u>l'installation de l'avant dernière version de **Pandas (1.2.5)**</u>

    On clique ensuite sur **_Suivant_**

    ### 4.6.4 Étape 4 : Sécurité

    #### 4.6.4.1 Options de sécurité

    A cette étape nous sélectionnons la **paire de clés EC2** créé précédemment.
    Elle nous permettra de se connecter en **ssh** à nos **instances EC2**
    sans avoir à entrer nos login/mot de passe.
    On laisse les autres paramètres par défaut.
    Et enfin, on clique sur "**_Créer un cluster_**"

    {mo.image("public/EMR_securite.png",caption="EMR Sécurité")}

    ## 4.7 Instanciation du serveur

    Il ne nous reste plus qu'à attendre que le serveur soit prêt.
    Cette étape peut prendre entre **15 et 20 minutes**.

    <u>Plusieurs étapes s'enchaîne, on peut suivre l'avancé du statut du **cluster EMR**</u> :

    {mo.image("public/EMR_instanciation_01.png",caption="Instanciation étape 1")}

    {mo.image("public/EMR_instanciation_02.png",caption="Instanciation étape 2")}

    {mo.image("public/EMR_instanciation_03.png",caption="Instanciation étape 3")}


    <u>Lorsque le statut affiche en vert: "**En attente**" cela signifie que l'instanciation
    s'est bien déroulée et que notre serveur est prêt à être utilisé</u>.

    ## 4.8 Création du tunnel SSH à l'instance EC2 (Maître)

    ### 4.8.1 Création des autorisations sur les connexions entrantes

    <u>Nous souhaitons maintenant pouvoir accéder à nos applications</u> :

    - **Marimo** pour l'exécution de notre notebook
    - **Serveur d'historique Spark** pour le suivi de l'exécution
      des tâches de notre script lorsqu'il sera lancé

    Cependant, <u>ces applications ne sont accessibles que depuis le réseau local du driver</u>,
    et pour y accéder nous devons **créer un tunnel SSH vers le driver**.

    Par défaut, ce driver se situe derrière un firewall qui bloque l'accès en SSH.
    <u>Pour ouvrir le port 22 qui correspond au port sur lequel écoute le serveur SSH,
    il faut modifier le **groupe de sécurité EC2 du driver**</u>.

    _Il faudra que l'on se connecte en SSH au driver de notre cluster.
    Par défaut, ce driver se situe derrière un firewall qui bloque l'accès en SSH.
    Pour ouvrir le port 22 qui correspond au port sur lequel écoute le serveur SSH,
    il faut modifier le groupe de sécurité EC2 du driver. Sur la page de la console
    consacrée à EC2, dans l'onglet "Réseau et sécurité", cliquez sur "Groupes de sécurité".
    Vous allez devoir modifier le groupe de sécurité d’ElasticMapReduce-Master.
    Dans l'onglet "Entrant", ajoutez une règle SSH dont la source est "N'importe où"
    (ou "Mon IP" si vous disposez d'une adresse IP fixe)._

    {mo.image("public/EMR_config_ssh_01.png",caption="Configuration autorisation ports entrants pour ssh")}

    <u>Une fois cette étape réalisée vous devriez avoir une configuration semblable à la mienne</u> :

    {mo.image("public/EMR_config_ssh_02.png",caption="Configuration ssh terminée")}

    ### 4.8.2 Création du tunnel ssh vers le Driver

    On peut maintenant établir le **tunnel SSH** vers le **Driver**.
    Pour cela on récupère les informations de connexion fournis par Amazon
    depuis la page du service EMR / Cluster / onglet Récapitulatif en
    cliquant sur "**Activer la connexion Web**"

    {mo.image("public/EMR_tunnel_ssh_01.png",caption="Activer la connexion Web")}

    <u>On récupère ensuite la commande fournis par Amazon pour **établir le tunnel SSH**</u> :

    {mo.image("public/EMR_tunnel_ssh_02.png",caption="Récupérer la commande pour établir le tunnel ssh")}

    <u>Dans mon cas, la commande ne fonctionne pas tel</u> quel et j'ai du **l'adapter à ma configuration**.
    La **clé ssh** se situe dans un dossier "**.ssh**" elle-même située dans
    mon **répertoire personnel** dont le symbole est, sous Linux, identifié par un tilde "**~**".

    <u>Finalement, j'utilise la commande suivante dans un terminal pour établir
    mon tunnel ssh (seul l'URL change d'une instance à une autre)</u> :
    "**ssh -i ~/.ssh/p8-ec2.pem -D 5555 hadoop@ec2-35-180-91-39.eu-west-3.compute.amazonaws.com**"

    <u>On inscrit "**yes**" pour valider la connexion et si
    la connexion est établit on obtient le résultat suivant</u> :

    {mo.image("public/EMR_connexion_ssh_01.png",caption="Création du tunnel SSH")}

    Nous avons **correctement établi le tunnel ssh avec le driver** sur le port "5555".

    ### 4.8.3 Configuration de FoxyProxy

    Une dernière étape est nécessaire pour accéder à nos applications,
    en demandant à notre navigateur d'emprunter le tunnel ssh.
    J'utilise pour cela **FoxyProxy**.

    Sinon, ouvrez la configuration de **FoxyProxy** et <u>cliquez sur **Ajouter**</u> en haut à gauche
    puis renseigner les éléments comme dans la capture ci-dessous :

    {mo.image("public/EMR_foxyproxy_config_01.png",caption="Configuration FoxyProxy Etape 1")}

    <u>On obtient le résultat ci-dessous</u> :

    {mo.image("public/EMR_foxyproxy_config_02.png.png",caption="Configuration FoxyProxy Etape 2")}

    ### 4.8.4 Accès aux applications du serveur EMR via le tunnel ssh

    <u>Avant d'établir notre **tunnel ssh** nous avions ça</u> :

    {mo.image("public/EMR_tunnel_ssh_avant.png",caption="avant tunnel ssh")}

    <u>On active le **tunnel ssh** comme vu précédemment puis on demande
    à notre navigateur de l'utiliser avec **FoxyProxy**</u> :

    {mo.image("public/EMR_foxyproxy_activation.png",caption="FoxyProxy activation")}

    <u>On peut maintenant s'apercevoir que plusieurs applications nous sont accessibles</u> :

    {mo.image("public/EMR_tunnel_ssh_apres.png",caption="avant tunnel ssh")}

    ## 4.9 Connexion au notebook JupyterHub

    Pour se connecter à **JupyterHub** en vue d'exécuter notre **notebook**,
    il faut commencer par <u>cliquer sur l'application **JupyterHub**</u> apparu
    depuis que nous avons configuré le **tunnel ssh** et **foxyproxy** sur
    notre navigateur (actualisez la page si ce n’est pas le cas).

    {mo.image("public/EMR_jupyterhub_connexion_01.png",caption="Démarrage de JupyterHub")}

    On passe les éventuels avertissements de sécurité puis
    nous arrivons sur une page de connexion.

    <u>On se connecte avec les informations par défaut</u> :

    - <u>login</u>: **jovyan**
    - <u>password</u>: **jupyter**

    {mo.image("public/EMR_jupyterhub_connexion_02.png",caption="Connexion à JupyterHub")}

    Nous arrivons ensuite dans un dossier vierge de notebook.
    Il suffit d'en créer un en cliquant sur "**New**" en haut à droite.

    {mo.image("public/EMR_jupyterhub_creer_notebooks.png",caption="Liste et création des notebook")}

    Il est également possible d'en <u>uploader un directement dans notre **bucket S3**</u>.

    Grace à la <u>**persistance** paramétrée à l'instanciation du cluster
    nous sommes actuellement dans l'arborescence de notre **bucket S3**</u>

    {mo.image("public/EMR_jupyterhub_S3.png",caption="Notebook stockés sur S3")}

    Je décide d'**importer un notebook déjà rédigé en local directement
    sur S3** et je l'ouvre depuis **l'interface JupyterHub**.

    ## 4.10 Exécution du code

    Je décide d'exécuter cette partie du code depuis **JupyterHub hébergé sur notre cluster EMR**.
    Pour ne pas alourdir inutilement les explications du **notebook**, je ne réexpliquerai pas les étapes communes
    que nous avons déjà vues dans la première partie où l'on a exécuté le code localement sur notre machine virtuelle Ubuntu.

    <u>Avant de commencer</u>, il faut s'assurer d'utiliser le **kernel pyspark**.

    **En utilisant ce kernel, une session spark est créé à l'exécution de la première cellule**.
    Il n'est donc **plus nécessaire d'exécuter le code "spark = (SparkSession ..."** comme lors
    de l'exécution de notre notebook en local sur notre VM Ubuntu.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.10.1 Démarrage de la session Spark
    """)
    return


@app.cell
def _():
    # L'exécution de cette cellule démarre l'application Spark
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <u>Affichage des informations sur la session en cours et liens vers Spark UI</u> :
    """)
    return


@app.cell
def _():
    # %%info
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.10.2 Installation des packages

    Les packages nécessaires ont été installé via l'étape de **bootstrap** à l'instanciation du serveur.

    ### 4.10.3 Import des librairies
    """)
    return


@app.cell
def _():
    # import pandas as pd
    # import numpy as np
    # import io
    # import os
    # import tensorflow as tf
    # from PIL import Image
    # from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    # from tensorflow.keras.preprocessing.image import img_to_array
    # from tensorflow.keras import Model
    # from pyspark.sql.functions import col, pandas_udf, PandasUDFType, element_at, split
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.10.4 Définition des PATH pour charger les images et enregistrer les résultats

    Nous accédons directement à nos **données sur S3** comme si elles étaient **stockées localement**.
    """)
    return


@app.cell
def _():
    # PATH = 's3://p9-data'
    # PATH_Data = PATH+'/Test'
    # PATH_Result = PATH+'/Results'
    # print('PATH:        '+\
    #       PATH+'\nPATH_Data:   '+\
    #       PATH_Data+'\nPATH_Result: '+PATH_Result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.10.5 Traitement des données
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.10.5.1 Chargement des données
    """)
    return


@app.cell
def _():
    # images = spark.read.format("binaryFile") \
    #   .option("pathGlobFilter", "*.jpg") \
    #   .option("recursiveFileLookup", "true") \
    #   .load(PATH_Data)
    return


@app.cell
def _():
    # images.show(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <u>Je ne conserve que le **path** de l'image et j'ajoute
    une colonne contenant les **labels** de chaque image</u> :
    """)
    return


@app.cell
def _():
    # images = images.withColumn('label', element_at(split(images['path'], '/'),-2))
    # print(images.printSchema())
    # print(images.select('path','label').show(5,False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.10.5.2 Préparation du modèle
    """)
    return


@app.cell
def _():
    # model = MobileNetV2(weights='imagenet',
    #                     include_top=True,
    #                     input_shape=(224, 224, 3))
    return


@app.cell
def _():
    # new_model = Model(inputs=model.input,
    #                   outputs=model.layers[-2].output)
    return


@app.cell
def _():
    # brodcast_weights = sc.broadcast(new_model.get_weights())
    return


@app.cell
def _():
    # new_model.summary()
    return


@app.cell
def _():
    # def model_fn():
    #     """
    #     Returns a MobileNetV2 model with top layer removed
    #     and broadcasted pretrained weights.
    #     """
    #     model = MobileNetV2(weights='imagenet',
    #                         include_top=True,
    #                         input_shape=(224, 224, 3))
    #     for layer in model.layers:
    #         layer.trainable = False
    #     new_model = Model(inputs=model.input,
    #                   outputs=model.layers[-2].output)
    #     new_model.set_weights(brodcast_weights.value)
    #     return new_model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.10.5.3 Définition du processus de chargement des images <br/> et application de leur featurisation à travers l'utilisation de pandas UDF
    """)
    return


@app.cell
def _():
    # def preprocess(content):
    #     """
    #     Preprocesses raw image bytes for prediction.
    #     """
    #     img = Image.open(io.BytesIO(content)).resize([224, 224])
    #     arr = img_to_array(img)
    #     return preprocess_input(arr)

    # def featurize_series(model, content_series):
    #     """
    #     Featurize a pd.Series of raw images using the input model.
    #     :return: a pd.Series of image features
    #     """
    #     input = np.stack(content_series.map(preprocess))
    #     preds = model.predict(input)
    #     # For some layers, output features will be multi-dimensional tensors.
    #     # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    #     output = [p.flatten() for p in preds]
    #     return pd.Series(output)

    # @pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
    # def featurize_udf(content_series_iter):
    #     '''
    #     This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    #     The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).

    #     :param content_series_iter: This argument is an iterator over batches of data, where each batch
    #                               is a pandas Series of image data.
    #     '''
    #     # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    #     # for multiple data batches.  This amortizes the overhead of loading big models.
    #     model = model_fn()
    #     for content_series in content_series_iter:
    #         yield featurize_series(model, content_series)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.10.5.4 Exécutions des actions d'extractions de features
    """)
    return


@app.cell
def _():
    # spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")
    return


@app.cell
def _():
    # features_df = images.repartition(24).select(col("path"),
    #                                             col("label"),
    #                                             featurize_udf("content").alias("features")
    #                                            )
    return


@app.cell
def _():
    # print(PATH_Result)
    return


@app.cell
def _():
    # features_df.write.mode("overwrite").parquet(PATH_Result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.10.6 Chargement des données enregistrées et validation du résultat
    """)
    return


@app.cell
def _():
    # df = pd.read_parquet(PATH_Result, engine='pyarrow')
    return


@app.cell
def _():
    # df.head()
    return


@app.cell
def _():
    # df.loc[0,'features'].shape
    return


@app.cell
def _():
    # df.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <u>On peut également constater la présence des fichiers
    au format "**parquet**" sur le **serveur S3**</u> :

    {mo.image("public/S3_Results.png",caption="Affichage des résultats sur S3")}

    ## 4.11 Suivi de l'avancement des tâches avec le Serveur d'Historique Spark

    Il est possible de voir l'avancement des tâches en cours
    avec le **serveur d'historique Spark**.

    {mo.image("public/EMR_serveur_historique_spark_acces.png",caption="Accès au serveur d'historique spark")}

    **Il est également possible de revenir et d'étudier les tâches
    qui ont été réalisé, afin de debugger, optimiser les futurs
    tâches à réaliser.**

    <u>Lorsque la commande "**features_df.write.mode("overwrite").parquet(PATH_Result)**"
    était en cours, nous pouvions observer son état d'avancement</u> :

    {mo.image("public/EMR_jupyterhub_avancement.png",caption="Progression execution script")}

    <u>Le **serveur d'historique Spark** nous permet une vision beaucoup plus précise
    de l'exécution des différentes tâche sur les différentes machines du cluster</u> :

    {mo.image("public/EMR_SHSpark_01.png",caption="Suivi des tâches spark")}

    On peut également constater que notre cluster de calcul a mis
    un tout petit peu **moins de 8 minutes** pour traiter les **22 688 images**.

    {mo.image("public/EMR_SHSpark_02.png",caption="Temps de traitement")}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.12 Résiliation de l'instance EMR

    Notre travail est maintenant terminé.
    Le cluster de machines EMR est **facturé à la demande**,
    et nous continuons d'être facturé même lorsque
    les machines sont au repos.
    Pour **optimiser la facturation**, il nous faut
    maintenant **résilier le cluster**.

    <u>Je réalise cette commande depuis l'interface AWS</u> :

    1. Commencez par **désactiver le tunnel ssh dans FoxyProxy** pour éviter des problèmes de **timeout**.
       {mo.image("public/EMR_foxyproxy_desactivation.png",caption="Désactivation de FoxyProxy")}

    3. Cliquez sur "**Résilier**"
       mo.image("public/EMR_resiliation_01.png",caption="Cliquez sur Résilier")}


    5. Confirmez la résiliation
        {mo.image("public/EMR_resiliation_02.png",caption="Confirmez la résiliation")}

    7. La résiliation prend environ **1 minute**
       {mo.image("public/EMR_resiliation_03.png",caption="Résiliation en cours")}

    9. La résiliation est effectuée
       {mo.image("public/EMR_resiliation_04.png",caption="Résiliation terminée")}

    ## 4.13 Cloner le serveur EMR (si besoin)

    Si nous devons de nouveau exécuter notre notebook dans les mêmes conditions,
    il nous suffit de **cloner notre cluster** et ainsi en obtenir une copie fonctionnelle
    sous 15/20 minutes, le temps de son instanciation.

    <u>Pour cela deux solutions</u> :

    1. <u>Depuis l'interface AWS</u> :
    1. Cliquez sur "**Cloner**"
       {mo.image("public/EMR_cloner_01.png",caption="Cloner un cluster")}

    1. Dans notre cas nous ne souhaitons pas inclure d'étapes
        {mo.image("public/EMR_cloner_02.png",caption="Ne pas inclure d'étapes")}

    1. La configuration du cluster est recréée à l’identique.
       On peut revenir sur les différentes étapes si on souhaite apporter des modifications
       Quand tout est prêt, cliquez sur "**Créer un cluster**"
       {mo.image("public/EMR_cloner_03.png",caption="Vérification/Modification/Créer un cluster")}

    1. <u>En ligne de commande</u> (avec AWS CLI d'installé et de configuré et en s'assurant
       de s'attribuer les droits nécessaires sur le compte AMI utilisé)
    1. Cliquez sur "**Exporter AWS CLI**"
       {mo.image("public/EMR_cloner_cli_01.png",caption="Exporter AWS CLI")}

    1. Copier/Coller la commande **depuis un terminal**
       {mo.image("public/EMR_cloner_cli_02.png",caption="Copier Coller Commande")}

    ## 4.14 Arborescence du serveur S3 à la fin du projet

    <u>Pour information, voici **l'arborescence complète de mon bucket S3 p8-data** à la fin du projet</u> :
    _Par soucis de lisibilité, je ne liste pas les 131 sous dossiers du répertoire "Test"_

    1. Results/\_SUCCESS
    1. Results/part-00000-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00001-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00002-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00003-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00004-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00005-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00006-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00007-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00008-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00009-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00010-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00011-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00012-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00013-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00014-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00015-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00016-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00017-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00018-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00019-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00020-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00021-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00022-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Results/part-00023-2cc36f38-19ef-4d8a-a0d1-5ddb309b3894-c000.snappy.parquet
    1. Test/
    1. bootstrap-emr.sh
    1. jupyter-s3-conf.json
    1. jupyter/jovyan/.s3keep
    1. jupyter/jovyan/P8_01_Notebook.ipynb
    1. jupyter/jovyan/\_metadata
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/.aws-editors-workspace-metadata/
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/.aws-editors-workspace-metadata/file-perm.sqlite
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/.aws-editors-workspace-metadata/nbconvert/
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/.aws-editors-workspace-metadata/nbconvert/templates/
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/.aws-editors-workspace-metadata/nbconvert/templates/html/
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/.aws-editors-workspace-metadata/nbconvert/templates/latex/
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/.aws-editors-workspace-metadata/nbsignatures.db
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/.aws-editors-workspace-metadata/notebook_secret
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/.ipynb_checkpoints/
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/.ipynb_checkpoints/Untitled-checkpoint.ipynb
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/.ipynb_checkpoints/Untitled1-checkpoint.ipynb
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/.ipynb_checkpoints/test3-checkpoint.ipynb
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/Untitled.ipynb
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/Untitled1.ipynb
    1. jupyter/jovyan/e-5OTY4VKPDT21945FF6DN15E35/test3.ipynb
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. Conclusion

    Nous avons réalisé ce projet **en deux temps** en tenant
    compte des contraintes qui nous ont été imposées.

    Nous avons **dans un premier temps développé notre solution en local**
    sur une machine virtuelle dans un environnement Linux Ubuntu.

    La <u>première phase</u> a consisté à **installer l'environnement de travail Spark**.
    **Spark** a un paramètre qui nous permet de travaillé en local et nous permet
    ainsi de **simuler du calcul partagé** en considérant
    **chaque cœur d'un processeur comme un worker indépendant**.
    Nous avons travaillé sur un plus **petit jeu de donnée**, l'idée était
    simplement de **valider le bon fonctionnement de la solution**.

    Nous avons fait le choix de réaliser du **transfert learning**
    à partir du model **MobileNetV2**.
    Ce modèle a été retenu pour sa **légèreté** et sa **rapidité d'exécution**
    ainsi que pour la **faible dimension de son vecteur en sortie**.

    Les résultats ont été enregistrés sur disque en plusieurs
    partitions au format "**parquet**".

    <u>**La solution a parfaitement fonctionné en mode local**</u>.

    La <u>deuxième phase</u> a consisté à créer un **réel cluster de calculs**.
    L'objectif était de pouvoir **anticiper une future augmentation de la charge de travail**.

    Le meilleur choix retenu a été l'utilisation du prestataire de services **Amazon Web Services**
    qui nous permet de **louer à la demande de la puissance de calculs**,
    pour un **coût tout à fait acceptable**.
    Ce service se nomme **EC2** et se classe parmi les offres **Infrastructure As A Service** (IAAS).

    Nous sommes allez plus loin en utilisant un service de plus
    haut niveau (**Plateforme As A Service** PAAS)
    en utilisant le service **EMR** qui nous permet d'un seul coup
    d'**instancier plusieurs serveur (un cluster)** sur lesquels
    nous avons pu demander l'installation et la configuration de plusieurs
    programmes et librairies nécessaires à notre projet comme **Spark**,
    **Hadoop**, **JupyterHub** ainsi que la librairie **TensorFlow**.

    En plus d'être plus **rapide et efficace à mettre en place**, nous avons
    la **certitude du bon fonctionnement de la solution**, celle-ci ayant été
    préalablement validé par les ingénieurs d'Amazon.

    Nous avons également pu installer, sans difficulté, **les packages
    nécessaires sur l'ensembles des machines du cluster**.

    Enfin, avec très peu de modification, et plus simplement encore,
    nous avons pu **exécuter notre notebook comme nous l'avions fait localement**.
    Nous avons cette fois-ci exécuté le traitement sur **l'ensemble des images de notre dossier "Test"**.

    Nous avons opté pour le service **Amazon S3** pour **stocker les données de notre projet**.
    S3 offre, pour un faible coût, toutes les conditions dont nous avons besoin pour stocker
    et exploiter de manière efficace nos données.
    L'espace alloué est potentiellement **illimité**, mais les coûts seront fonction de l'espace utilisé.

    Il nous sera **facile de faire face à une monté de la charge de travail** en **redimensionnant**
    simplement notre cluster de machines (horizontalement et/ou verticalement au besoin),
    les coûts augmenteront en conséquence mais resteront nettement inférieurs aux coûts engendrés
    par l'achat de matériels ou par la location de serveurs dédiés.
    """)
    return


if __name__ == "__main__":
    app.run()
