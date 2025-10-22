import marimo

__generated_with = "0.17.0"
app = marimo.App(
    width="columns",
    app_title="P8 Notebook veille",
    auto_download=["html"],
)


@app.cell(column=0)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Projet 8 -  Veille technologique (nouveau modele de NLP) en
    Comparaison avec ceux du Projet 6 pour le Traitement de Texte
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Introduction""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Importation des librairies""")
    return


@app.cell
def _():
    # Chargement des librairies
    import time
    import warnings

    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Plotly Libraries
    import plotly.express as px
    import seaborn as sns

    sns.set_theme(style="white", palette="Set2")

    import plotly.io as pio

    pio.get_chrome()

    import ast

    import shap

    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.metrics import classification_report as SKclassification_report
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split

    from yellowbrick.classifier import classification_report, confusion_matrix

    # Warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")
    return (
        LogisticRegression,
        Path,
        SKclassification_report,
        SentenceTransformer,
        accuracy_score,
        ast,
        confusion_matrix,
        f1_score,
        np,
        pd,
        plt,
        px,
        shap,
        sns,
        time,
        train_test_split,
    )


@app.cell
def _(ast, np, pd, train_test_split):
    # Fonction pour reconcaténer les mots en phrase
    def reconcatener_mots(liste_mots):
        # Supprimer les doublons consécutifs
        liste_unique = []
        for mot in liste_mots:
            if not liste_unique or mot != liste_unique[-1]:
                liste_unique.append(mot)
        # Rejoindre les mots en une phrase
        return " ".join(liste_unique)

    # ------------------------------
    # 1. Chargement des données
    # ------------------------------
    data = pd.read_csv("data/raw/clean_description.csv", usecols=["preprocessed_descr", "main_category"])

    # Appliquer la fonction à la colonne preprocessed_descr
    data["preprocessed_descr"] = data["preprocessed_descr"].apply(lambda x: reconcatener_mots(ast.literal_eval(x)))

    # 3. Préparation des données
    texts = data["preprocessed_descr"].astype(str).tolist()
    labels = data["main_category"].tolist()

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, stratify=data["main_category"], random_state=42
    )

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    return X_test, X_train, data, y_test, y_train


@app.cell(hide_code=True)
def _(data):
    # Taille : nombre de lignes/colonnes
    nRow, nVar = data.shape
    print(f"Le jeu de données contient {nRow} lignes et {nVar} variables.")
    return


@app.cell(hide_code=True)
def _(pd, plt, sns, y_train):
    # Compter les valeurs de chaque catégorie
    counts = pd.Series(y_train).value_counts()

    # Créer un graphique en secteurs avec Seaborn
    plt.figure(figsize=(6, 6))
    colors = sns.color_palette("pastel")[0 : len(counts)]
    plt.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct=lambda p: f"{p * sum(counts) / 100:.0f}",
        startangle=140,
    )

    plt.title("Nombre de produit par catégorie")
    plt.tight_layout()
    plt.show()
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""## Fonctions""")
    return


@app.cell
def _(Path):
    # Chemin pour sauvegarder les temps et les embeddings
    EMBEDDINGS_DIR = Path("data/embeddings")
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    TIMES_PATH = EMBEDDINGS_DIR / "embedding_times.csv"
    return EMBEDDINGS_DIR, TIMES_PATH


@app.cell
def _(EMBEDDINGS_DIR, SentenceTransformer, TIMES_PATH, pd, time):
    def generate_or_load_embeddings(model_name, model_path, texts, labels, split="train", batch_size=32, device="cpu"):
        embeddings_path = EMBEDDINGS_DIR / f"{model_name}_{split}.parquet"

        if embeddings_path.exists():
            print(f"📂 Chargement des embeddings sauvegardés pour {model_name} ({split})")
            embeddings_df = pd.read_parquet(embeddings_path)
            embeddings = embeddings_df.drop(columns=["label"], errors="ignore").values
            labels = embeddings_df["label"].values
            if TIMES_PATH.exists():
                df_times = pd.read_csv(TIMES_PATH)
                time_row = df_times[(df_times["model"] == model_name) & (df_times["split"] == split)]
                if not time_row.empty:
                    elapsed_time = time_row["time_sec"].values[0]
                    print(f"⏳ Temps d'encodage précédent pour {model_name} ({split}) : {elapsed_time:.2f} sec")
                else:
                    elapsed_time = None
            else:
                elapsed_time = None
        else:
            print(f"🛠️ Génération des embeddings pour {model_name} ({split})")
            model = SentenceTransformer(model_path, device=device)
            start = time.time()
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
            elapsed_time = time.time() - start
            print(f"⏱️ Temps d'encodage pour {model_name} ({split}) : {elapsed_time:.2f} sec")

            # Sauvegarde des embeddings avec les labels
            embeddings_df = pd.DataFrame(embeddings)
            embeddings_df["label"] = labels
            embeddings_df.to_parquet(embeddings_path, index=False)

            # Sauvegarde du temps
            new_row = pd.DataFrame([{"model": model_name, "split": split, "time_sec": elapsed_time}])
            if TIMES_PATH.exists():
                df_times = pd.read_csv(TIMES_PATH)
                df_times = pd.concat([df_times, new_row], ignore_index=True)
            else:
                df_times = new_row
            df_times.to_csv(TIMES_PATH, index=False)

        return embeddings, labels, elapsed_time
    return (generate_or_load_embeddings,)


@app.cell
def _(
    LogisticRegression,
    Path,
    SKclassification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
    generate_or_load_embeddings,
    pd,
    plt,
    sns,
):
    def train_and_evaluate(model_name, model_path, X_train, X_test, y_train, y_test, batch_size=32, device="cpu"):
        RESULTS_DIR = Path(f"notebooks/results/{model_name}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Générer/charger les embeddings pour train et test
        embeddings_train, labels_train, _ = generate_or_load_embeddings(
            model_name,
            model_path,
            X_train,
            y_train,
            split="train",
            batch_size=batch_size,
            device=device,
        )
        embeddings_test, labels_test, _ = generate_or_load_embeddings(
            model_name,
            model_path,
            X_test,
            y_test,
            split="test",
            batch_size=batch_size,
            device=device,
        )

        # Entraîner le classifieur
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(embeddings_train, labels_train)

        # Prédictions et évaluation avec scikit-learn
        labels_pred = clf.predict(embeddings_test)
        class_names = clf.classes_

        # Calculer accuracy et F1-score
        accuracy = accuracy_score(labels_test, labels_pred)
        f1 = f1_score(labels_test, labels_pred, average="weighted")

        # Enregistrer les métriques dans un fichier CSV
        csv_path = RESULTS_DIR / f"{model_name}_metrics.csv"
        metrics_data = {"model_name": [model_name], "accuracy": [accuracy], "f1_score": [f1]}
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv(csv_path, index=False)

        print(f"\n--- Résultats pour {model_name} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        report_dict = SKclassification_report(
            labels_test, labels_pred, output_dict=True, target_names=class_names, zero_division=0
        )

        # Visualiser la matrice de confusion avec Yellowbrick
        visualizer = confusion_matrix(
            clf,
            embeddings_train,
            labels_train,
            embeddings_test,
            labels_test,
            classes=class_names,
            percent=True,
            zero_division=0,
        )
        visualizer.finalize()
        confusion_matrix_path = str(RESULTS_DIR / f"{model_name}_confusion_matrix.png")
        visualizer.fig.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close(visualizer.fig)  # Close the figure to avoid display in notebook if not wanted


        # Convert report to DataFrame for easier plotting
        report_df = pd.DataFrame(report_dict).transpose()

        # Extract precision, recall, f1-score for each class (exclude accuracy, macro avg, weighted avg)
        metrics = ["precision", "recall", "f1-score",]
        class_metrics = report_df.loc[class_names, metrics]

        # Plot heatmap with support as annotations
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create heatmap for precision, recall, f1-score (without support)
        metrics_for_heatmap = class_metrics[['precision', 'recall', 'f1-score']]
        sns.heatmap(metrics_for_heatmap.astype(float), annot=True, cmap='Blues', fmt='.2f', 
                    cbar_kws={'label': 'Score'}, ax=ax)


        plt.title(f'Classification Report - {model_name}')
        plt.ylabel('Classes')
        plt.xlabel('Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the figure
        plt.savefig(RESULTS_DIR / f"{model_name}_classification_report.png", dpi=300, bbox_inches='tight')
        plt.close()
        return clf, embeddings_train, embeddings_test
    return (train_and_evaluate,)


@app.cell
def _(Path, np, plt, shap, y_test):
    def visualize_shap_sentence_level(
        model_name, model_path, clf_sklearn, embeddings_train, embeddings_test, device="cpu"
    ):
        """
        Visualise les importances SHAP au NIVEAU PHRASE (embedding-level)
        pour un modèle SentenceTransformer + classifieur sklearn.

        Args:
            model_name (str): Nom du modèle (pour sauvegarde)
            model_path (str): Chemin du modèle SentenceTransformer
            clf_sklearn: Classifieur sklearn entraîné (LogisticRegression, etc.)
            embeddings_train (list): embeddings entrainement
            embeddings_test (list): embeddings tests
            device (str): "cpu" ou "cuda"
        """
        print(f"\n{'=' * 70}")
        print(f"🔍 SHAP SENTENCE-LEVEL ANALYSIS - {model_name}")
        print(f"{'=' * 70}")

        RESULTS_DIR = Path(f"notebooks/results/{model_name}/shap_sentence_level")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # === 2. Encoder les phrases ===
        print(f"✅ Enmbeddings test : {embeddings_test.shape[0]} phrases × {embeddings_test.shape[1]} dimensions")

        # === 3. Préparer SHAP ===
        print(f"⚙️  Création de l'explainer SHAP...")
        class_names = clf_sklearn.classes_

        # --- SHAP EXPLANATION ---
        explainer = shap.Explainer(clf_sklearn, embeddings_train, output_names=class_names)
        shap_values = explainer(embeddings_test)

        # === 4. Visualisation globale ===
        print("📊 Génération des visualisations SHAP globales...")

        # -- Summary plot (type bar)
        plt.figure()
        shap.summary_plot(shap_values, embeddings_test, class_names=class_names,show=False)
        plt.title(f"Global Feature Importance - {model_name}", fontsize=14, weight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"shap_global_bar_{model_name}.png", bbox_inches="tight", dpi=200)
        plt.close()

        print(f"💾 Sauvegardé dans {RESULTS_DIR}")

        # === 5. VISUALISATION LOCALE (Waterfall) pour classe 0 ===
        print(f"\n🔎 Génération d'un plot local pour une phrase de la classe 0...")

        # Obtenir les prédictions
        y_pred = clf_sklearn.predict(embeddings_test)

        # Trouver les indices des phrases prédites comme classe 0
        class_name = "home furnishing"
        class_0_indices = np.where(y_pred == class_name)[0]

        if len(class_0_indices) == 0:
            print(f"⚠️  Aucune phrase prédite comme classe {class_name} trouvée!")
        else:
            # Sélectionner le premier exemple de classe 0
            idx_class_0 = class_0_indices[0]

            print(f"   📌 Index de l'exemple sélectionné : {idx_class_0}")
            print(f"   📌 Prédiction : Classe {y_pred[idx_class_0]}")

            if y_test is not None:
                print(f"   📌 Label réel : Classe {y_test[idx_class_0]}")

            # Waterfall plot pour cet exemple (classe 0)
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[idx_class_0, :, 0], show=False)
            plt.title(f"Explication locale - Exemple classe {class_name} (index {idx_class_0})", 
                      fontsize=14, weight="bold")
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"shap_local_waterfall_class0_{model_name}.png", 
                        bbox_inches="tight", dpi=200)
            plt.close()

            print(f"   💾 Waterfall plot sauvegardé dans {RESULTS_DIR}")

        return shap_values
    return (visualize_shap_sentence_level,)


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""## Liste de modèles à tester""")
    return


@app.cell
def _():
    models = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",  # très léger
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",  # plus lourd mais performant
        "bert-base-uncased": "sentence-transformers-testing/st-bert-base-uncased",  # ancienne méthode
    }
    return (models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Boucle sur les modèles""")
    return


@app.cell
def _(X_train, generate_or_load_embeddings, models, y_train):
    # 1. Calculer les temps d'embeddings pour chaque modèle sur le jeu de train
    embedding_times = {}
    for _model_name, _model_path in models.items():
        print(f"\nCalcul des embeddings pour {_model_name} sur le jeu de train...")
        _, _, elapsed_time = generate_or_load_embeddings(
            _model_name,
            _model_path,
            X_train,
            y_train,
            split="train",
        )
        embedding_times[_model_name] = elapsed_time

    print("\nTemps d'embeddings pour chaque modèle (train) :")
    for _model_name, _time_sec in embedding_times.items():
        print(f"{_model_name}: {_time_sec:.2f} secondes")
    return


@app.cell
def _(
    X_test,
    X_train,
    models,
    train_and_evaluate,
    visualize_shap_sentence_level,
    y_test,
    y_train,
):
    # Pour chaque modèle
    for _model_name, _model_path in models.items():
        print(f"\n=== Modèle : {_model_name} ===")

        # 1. Entraîner et évaluer le classifieur
        clf, embeddings_train, embeddings_test = train_and_evaluate(
            _model_name,
            _model_path,
            X_train,
            X_test,
            y_train,
            y_test,
        )


        # 2. Visualiser les feature importances avec SHAP
        # visualize_shap3(_model_name, _model_path, clf, X_test, y_test)
        # visualize_shap_word_level(_model_name, _model_path, clf, X_test, y_test)
        results_shap = visualize_shap_sentence_level(
            model_name=_model_name,
            model_path=_model_path,
            clf_sklearn=clf,
            embeddings_train=embeddings_train,
            embeddings_test=embeddings_test,
            device="cpu",
        )
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""## Graphiques""")
    return


@app.cell
def _(TIMES_PATH, pd):
    time_df = pd.read_csv(TIMES_PATH, nrows=3, usecols=["model","time_sec"])
    time_df
    return (time_df,)


@app.cell
def _(Path, models, pd):
    # Define your models dictionary and base results directory
    # models = {"model1": "path1", "model2": "path2"}
    RESULTS_DIR = Path("notebooks/results")

    # Read all CSV files into a list of DataFrames using list comprehension
    dfs = [
        pd.read_csv(RESULTS_DIR / model_name / f"{model_name}_metrics.csv").assign(model_name=model_name)
        for model_name in models
    ]

    # Concatenate all DataFrames at once
    merged_metrics = pd.concat(dfs, ignore_index=True)
    merged_metrics
    return (merged_metrics,)


@app.cell
def _(merged_metrics, pd, time_df):
    df_results = pd.merge(time_df, merged_metrics, left_on="model", right_on="model_name")
    df_results = df_results.drop(columns=["model_name"])
    df_results
    return (df_results,)


@app.cell
def _(df_results, px):
    _fig_acc = px.scatter(
        df_results,
        x="time_sec",
        y="f1_score",
        text="model",
        size="accuracy",  # taille des points proportionnelle à l'accuracy
        color="model",
        title="Compromis temps/performance : Temps de calcul vs F1-score<br><sup>Taille des points proportionnelle à l'Accuracy</sup>",
        labels={"time_sec": "Temps embeddings (sec)", "f1_score": "F1 Score"},
    )

    # _fig_acc.update_traces(textposition="top center")
    _fig_acc.update_yaxes(range=[0.8, 1])  # Limite l'axe des F1 entre 0.5 et 1
    _fig_acc.update_xaxes(range=[0, 200])

    _fig_acc.show()
    _fig_acc.write_image("notebooks/results/fig_acc.png")
    return


@app.cell
def _(df_results, px):
    fig_bar_time = px.bar(
        df_results,
        x="model",
        y="time_sec",
        text="time_sec",
        title="Temps de calcul des embeddings par modèle",
        labels={"time_sec": "Temps (sec)", "model": "Modèle"},
    )
    fig_bar_time.update_traces(texttemplate="%{text:.2f} sec", textposition="outside")
    fig_bar_time.show()
    fig_bar_time.write_image("notebooks/results/fig_bar_time.png")
    return


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(r"""## Comparaisons""")
    return


@app.cell
def _(mo, models):
    # Create a list of labeled images
    labeled_images = [
        mo.vstack([
            mo.md(f"**{_model_name}**"),
            mo.image(
                f"notebooks/results/{_model_name}/{_model_name}_confusion_matrix.png",
                width=600,  # set the same width
            ),
        ])
        for _model_name in models
    ]

    # Arrange labeled images in a grid (e.g., 2 columns)
    rows = [mo.hstack(labeled_images[i : i + 2]) for i in range(0, len(labeled_images), 2)]
    mo.vstack(rows)
    return


@app.cell
def _(mo, models):
    # Create a list of labeled images
    _labeled_images = [
        mo.vstack([
            mo.md(f"**{_model_name}**"),
            mo.image(
                f"notebooks/results/{_model_name}/{_model_name}_classification_report.png",
                width=600,  # set the same width
            ),
        ])
        for _model_name in models
    ]

    # Arrange labeled images in a grid (e.g., 2 columns)
    _rows = [mo.hstack(_labeled_images[i : i + 2]) for i in range(0, len(_labeled_images), 2)]
    mo.vstack(_rows)
    return


@app.cell
def _(mo, models):
    # Create a list of labeled images
    _labeled_images = [
        mo.vstack([
            mo.md(f"**{_model_name}**"),
            mo.image(
                f"notebooks/results/{_model_name}/shap_sentence_level/shap_global_bar_{_model_name}.png",
                width=600,  # set the same width
            ),
        ])
        for _model_name in models
    ]

    # Arrange labeled images in a grid (e.g., 2 columns)
    _rows = [mo.hstack(_labeled_images[i : i + 2]) for i in range(0, len(_labeled_images), 2)]
    mo.vstack(_rows)
    return


@app.cell
def _(mo, models):
    # Create a list of labeled images
    _labeled_images = [
        mo.vstack([
            mo.md(f"**{_model_name}**"),
            mo.image(
                f"notebooks/results/{_model_name}/shap_sentence_level/shap_local_waterfall_class0_{_model_name}.png",
                width=600,  # set the same width
            ),
        ])
        for _model_name in models
    ]

    # Arrange labeled images in a grid (e.g., 2 columns)
    _rows = [mo.hstack(_labeled_images[i : i + 2]) for i in range(0, len(_labeled_images), 2)]
    mo.vstack(_rows)
    return


if __name__ == "__main__":
    app.run()
