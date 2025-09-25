import marimo

__generated_with = "0.16.2"
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
        r"""# Projet 8 -  Test Veille technoligique (nouveau modele) en Comparaison avec ceux du Projet 6 pour le Traitement de Texte"""
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
    import numpy as np
    import pandas as pd

    from pathlib import Path

    # Plotly Libraries
    import plotly.express as px
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", palette="Set2")

    import plotly.io as pio

    pio.get_chrome()

    from sentence_transformers import SentenceTransformer

    from datasets import Dataset, DatasetDict

    import pycaret.classification as clf

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report

    # Warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    # Nombre aléatoire pour la reproductibilité des résultats
    seed = 42
    return (
        Dataset,
        DatasetDict,
        Path,
        SentenceTransformer,
        clf,
        np,
        pd,
        plt,
        px,
        sns,
        time,
        train_test_split,
    )


@app.cell
def _(Dataset, DatasetDict, pd, train_test_split):
    # ------------------------------
    # 1. Chargement des données
    # ------------------------------
    data = pd.read_csv("data/raw/clean_description.csv")

    # 2. Échantillonnage stratifié initial
    sample_size = 1000
    sampled_data, _ = train_test_split(data, train_size=sample_size, stratify=data["main_category"], random_state=42)

    # 3. Préparation des données pour le Dataset
    texts = sampled_data["preprocessed_descr"].astype(str).tolist()
    labels = sampled_data["main_category"].tolist()  # Garde les labels sous forme de catégories

    # 4. Création du Dataset Hugging Face
    dataset = Dataset.from_dict({"text": texts, "label": labels})

    # 5. Récupération des indices et des labels pour la stratification
    indices = list(range(len(dataset)))
    y = [label for label in dataset["label"]]

    # 6. Division train/test stratifiée avec sklearn
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)

    # 7. Création des sous-ensembles
    train_dataset = dataset.select(train_idx)
    test_dataset = dataset.select(test_idx)

    # 8. Construction du DatasetDict final
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    return data, dataset


@app.cell(hide_code=True)
def _(data):
    # Taille : nombre de lignes/colonnes
    nRow, nVar = data.shape
    print(f"Le jeu de données contient {nRow} lignes et {nVar} variables.")
    return


@app.cell(hide_code=True)
def _(dataset, pd, plt, sns):
    # Compter les valeurs de chaque catégorie
    counts = pd.Series(dataset["train"]["label"]).value_counts()

    # Créer un graphique en secteurs avec Seaborn
    plt.figure(figsize=(6, 6))
    colors = sns.color_palette("pastel")[0 : len(counts)]
    plt.pie(
        counts.values,
        labels=counts.index,
        # colors=colors,
        autopct=lambda p: "{:.0f}".format(p * sum(counts) / 100),
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
def _(Path, SentenceTransformer, pd, time):
    # ------------------------------
    # Fonction pour générer ou charger les embeddings
    # ------------------------------
    def generate_embeddings(model, dataset, batch_size=32, device="cpu"):
        embeddings_dir = Path("data/embeddings")
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        model_name, model_path = model

        # chemins de sauvegarde
        train_path = embeddings_dir / f"{model_name}_train.parquet"
        test_path = embeddings_dir / f"{model_name}_test.parquet"
        times_path = embeddings_dir / "embedding_times.csv"

        if train_path.exists() and test_path.exists():
            print(f"📂 Chargement des embeddings sauvegardés pour {model_name}")
            df_train = pd.read_parquet(train_path)
            df_test = pd.read_parquet(test_path)

            # Recherche du temps associé dans le fichier CSV
            if times_path.exists():
                df_times = pd.read_csv(times_path)
                time_row = df_times[df_times["model"] == model_name]
                if not time_row.empty:
                    elapsed_time = time_row["time_sec"].values[0]
                    print(f"⏳ Temps d'encodage précédent pour {model_name} : {elapsed_time:.2f} sec")
                else:
                    elapsed_time = None
                    print(f"⚠️ Temps non trouvé pour {model_name} dans le fichier CSV")
            else:
                elapsed_time = None
                print(f"⚠️ Fichier de temps introuvable pour {model_name}")
        else:
            print(f"🛠️ Génération des embeddings pour {model_name}")
            embedder = SentenceTransformer(model_path, device=device)
            start = time.time()

            # calcul des embeddings
            X_train = embedder.encode(
                dataset["train"]["text"],
                batch_size=batch_size,
                show_progress_bar=True,
            )
            X_test = embedder.encode(
                dataset["test"]["text"],
                batch_size=batch_size,
                show_progress_bar=True,
            )
            elapsed_time = time.time() - start
            print(f"⏱️ Temps d'encodage pour {model_name} : {elapsed_time:.2f} sec")

            # reconstruction DataFrames
            y_train = dataset["train"]["label"]
            y_test = dataset["test"]["label"]
            df_train = pd.DataFrame(X_train)
            df_train["label"] = y_train
            df_test = pd.DataFrame(X_test)
            df_test["label"] = y_test

            # Sauvegarde embeddings
            df_train.to_parquet(train_path, index=False)
            df_test.to_parquet(test_path, index=False)

            # Sauvegarde du temps dans un fichier CSV cumulatif
            new_row = pd.DataFrame([{"model": model_name, "time_sec": elapsed_time}])
            if times_path.exists():
                df_times = pd.read_csv(times_path)
                df_times = pd.concat([df_times, new_row], ignore_index=True)
            else:
                df_times = new_row
            df_times.to_csv(times_path, index=False)

        return df_train, df_test, elapsed_time
    return (generate_embeddings,)


@app.cell
def _(Path, clf):
    # ------------------------------
    # Fonction pour tester un embedding avec PyCaret
    # ------------------------------
    def evaluate_with_pycaret(df_train, df_test, model_name, seed=42):
        models_dir = Path("models")
        results_dir = Path("notebooks/results")
        model_results_dir = results_dir / model_name

        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        model_results_dir.mkdir(parents=True, exist_ok=True)

        save_path = models_dir / model_name.replace("/", "_")

        # Vérifie si un modèle existe déjà
        if (save_path.with_suffix(".pkl")).exists():
            print(f"📂 Chargement du modèle déjà entraîné : {model_name}")
            lr = clf.load_model(str(save_path))
        else:
            print(f"🛠️ Entraînement du modèle : {model_name}")
            clf.setup(data=df_train, target="label", session_id=seed, verbose=False)
            lr = clf.create_model("lr", fold=5)
            clf.save_model(lr, str(save_path))

        # Refaire un setup rapide avec le test pour activer le contexte PyCaret
        clf.setup(data=df_test, target="label", session_id=seed, verbose=False)

        # Prédictions + calcul métriques
        preds = clf.predict_model(lr, data=df_test)

        metrics = clf.pull().iloc[-1].to_dict()

        # Ajout d'infos complémentaires
        metrics["model_name"] = lr.__class__.__name__
        metrics["embedding_model"] = model_name
        metrics["saved_path"] = str(save_path)

        # Générez le chemin complet pour le fichier de sortie
        output_path = model_results_dir

        class_names = sorted(df_train["label"].unique().tolist())

        # Générez et enregistrez le graphique
        clf.plot_model(
            lr,
            plot="confusion_matrix",
            plot_kwargs={"percent": True, "classes": class_names},
            save=str(output_path),
        )

        print(f"📊 Confusion matrix sauvegardée")

        clf.plot_model(
            lr,
            plot="error",
            plot_kwargs={
                "percent": True,
                "classes": class_names,
            },
            save=str(output_path),
        )

        print(f"📊 Error plot sauvegardé")

        clf.plot_model(
            lr,
            plot="class_report",
            plot_kwargs={
                "percent": True,
                "classes": class_names,
            },
            save=str(output_path),
        )

        print(f"📊 Classification plot sauvegardé")

        return metrics
    return (evaluate_with_pycaret,)


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""## Liste de modèles à tester""")
    return


@app.cell
def _():
    models = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",  # très léger
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",  # plus lourd mais performant
        "bert-base-uncased": "google-bert/bert-base-uncased",  # ancienne méthode
    }
    return (models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Boucle sur les modèles""")
    return


@app.cell
def _(Path, dataset, evaluate_with_pycaret, generate_embeddings, models, pd):
    results = []
    for model in models.items():
        print(f"\n🚀 Test du modèle d'embedding : {model[0]}")

        # Génération ou chargement des embeddings + temps
        df_train, df_test, elapsed_time = generate_embeddings(model, dataset)

        # Évaluation avec PyCaret
        _metrics = evaluate_with_pycaret(df_train, df_test, model[0])

        # Ajout du temps de génération
        _metrics["embedding_time_sec"] = elapsed_time

        results.append(_metrics)

    # Conversion en DataFrame + sauvegarde
    df_results = pd.DataFrame(results)

    results_dir = Path("notebooks/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    df_results.to_csv(results_dir / "embeddings_pycaret_benchmark.csv", index=False)
    print("📂 Résultats sauvegardés dans data/results/embeddings_pycaret_benchmark.csv")
    return (df_results,)


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""## Graphiques""")
    return


@app.cell
def _(df_results):
    df_results
    return


@app.cell
def _(df_results, px):
    _fig_acc = px.scatter(
        df_results,
        x="embedding_time_sec",
        y="F1",
        text="embedding_model",
        size="Accuracy",  # taille des points proportionnelle à l'accuracy
        color="embedding_model",
        title="Compromis temps/performance : Temps de calcul vs F1-score<br><sup>Taille des points proportionnelle à l'Accuracy</sup>",
        labels={"embedding_time_sec": "Temps embeddings (sec)", "F1": "F1 Score"},
    )

    # _fig_acc.update_traces(textposition="top center")
    _fig_acc.update_yaxes(range=[0.8, 1])  # Limite l'axe des F1 entre 0.5 et 1
    _fig_acc.update_xaxes(range=[0, 600])

    _fig_acc.show()
    _fig_acc.write_image("notebooks/results/fig_acc.png")
    return


@app.cell
def _(df_results, px):
    fig_bar_time = px.bar(
        df_results,
        x="embedding_model",
        y="embedding_time_sec",
        text="embedding_time_sec",
        title="Temps de calcul des embeddings par modèle",
        labels={"embedding_time_sec": "Temps (sec)", "embedding_model": "Modèle"},
    )
    fig_bar_time.update_traces(texttemplate="%{text:.2f} sec", textposition="outside")
    fig_bar_time.show()
    fig_bar_time.write_image("notebooks/results/fig_bar_time.png")
    return


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(r"""## Analyse avec pycaret""")
    return


@app.cell
def _(mo, models):
    # Create a list of labeled images
    labeled_images = [
        mo.vstack([
            mo.md(f"**{_model_name}**"),
            mo.image(
                f"notebooks/results/{_model_name}/Confusion Matrix.png",
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
                f"notebooks/results/{_model_name}/Prediction Error.png",
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
                f"notebooks/results/{_model_name}/Class Report.png",
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
def _(np, plt, sns):
    # Exemple de données (remplace par tes F1-scores réels)
    categories = ["baby care", "beauty and personal care", "computers", "home decor & festive needs", "home furnishing", "kitchen & dining", "watches"]
    _models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "bert-base-uncased"]
    f1_scores = np.array([
        [0.857, 0.714, 1.000, 0.947, 0.900, 0.947, 1.000],  # all-MiniLM-L6-v2
        [0.714, 0.750, 1.000, 0.857, 1.000, 0.889, 1.000],  # all-mpnet-base-v2
        [0.714, 0.800, 1.000, 0.889, 0.947, 0.857, 1.000]   # bert-base-uncased
    ])

    # Création de la heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(f1_scores, annot=True, xticklabels=categories, yticklabels=_models, cmap="YlGnBu", fmt=".3f", linewidths=.5)
    plt.title("Comparaison des F1-scores par catégorie et par modèle")
    plt.xlabel("Catégories")
    plt.ylabel("Modèles")
    plt.show()

    return categories, f1_scores


@app.cell
def _():
    return


@app.cell
def _(categories, f1_scores, np, plt):
    # Exemple pour comparer les F1-scores des 3 modèles
    plt.figure(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.2

    plt.bar(x - width, f1_scores[0], width, label="all-MiniLM-L6-v2")
    plt.bar(x, f1_scores[1], width, label="all-mpnet-base-v2")
    plt.bar(x + width, f1_scores[2], width, label="bert-base-uncased")

    plt.xticks(x, categories, rotation=45)
    plt.ylabel("F1-score")
    plt.title("Comparaison des F1-scores par catégorie")
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    app.run()
