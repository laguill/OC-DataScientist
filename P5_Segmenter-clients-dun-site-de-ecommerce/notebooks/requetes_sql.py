import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium", app_title="P5 Requetes Dashboard")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Créations des requêtes SQL pour créer un dashboard""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Introduction""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Dans ce notebook, je liste les requêtes SQL suite à la demande de Olist.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Bienvenue chez Olist ! Nous sommes ravis d’avoir du soutien en cette période de mise en place de notre écosystème Data.

    A ce sujet d’ailleurs, l’un de nos projets Data phares du moment est la construction et la maintenance de notre Dashboard au service des équipes Customer Experience. Nous y exposons les KPIs essentiels pour que les équipes puissent avoir de la visibilité sur les états, les villes, ou les vendeurs qui nécessitent un suivi de près de la part de notre service client.

    Nous sommes en train d’alimenter les résultats de ce Dashboard avec des requêtes qui interrogent la base de données SQL à laquelle tu as accès.

    C’est sur l’implémentation de certaines requêtes urgentes que tu peux nous donner un coup de main le temps qu’un nouveau Data Analyst rejoigne l’équipe pour prendre le relais.

    Je mets en PJ la liste de requêtes SQL que nous avons besoin d’intégrer au Dashboard.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Voici les informations que le client souhaite obtenir et afficher dans son dashboard.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Liste de requêtes SQL pour le dashboard :

    1. En excluant les commandes annulées, quelles sont les commandes récentes de moins de 3 mois que les clients ont reçues avec au moins 3 jours de retard ?

    2. Qui sont les vendeurs ayant généré un chiffre d'affaires de plus de 100 000 Real sur des commandes livrées via Olist ?

    3. Qui sont les nouveaux vendeurs (moins de 3 mois d'ancienneté) qui sont déjà très engagés avec la plateforme (ayant déjà vendu plus de 30 produits) ?

    3. Quels sont les 5 codes postaux, enregistrant plus de 30 reviews, avec le pire review score moyen sur les 12 derniers mois ?
    """
    )
    return


@app.cell
def _(mo):
    _src = (
        "/home/laguill/Documents/01-Etudes/OpenClassrooms/P5_Segmenter-clients-dun-site-de-ecommerce/notebooks/public/image.png"
    )
    mo.image(src=_src, rounded=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Chargement des données""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Je commence par charger la base de données et les librairies nécessaires.""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import seaborn as sns

    import sqlalchemy

    DATABASE_URL = "sqlite:///data/raw/olist.db"
    engine = sqlalchemy.create_engine(DATABASE_URL)
    return engine, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Je liste les tables de la base de données.""")
    return


@app.cell
def _(engine, mo, sqlite_master):
    _df = mo.sql(
        f"""
        SELECT * FROM sqlite_master WHERE type='table'
        """,
        engine=engine,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""La base de données contient 9 tables.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Question 1""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Question 1
    **En excluant les commandes annulées, quelles sont les commandes récentes de moins de 3 mois que les clients ont reçues avec au moins 3 jours de retard ?**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - ~~charger la table `orders`~~ ✅

    - ~~filtrer les commandes `canceled`~~ ✅

    - ~~conserver liste des commandes des 3 derniers mois~~ ✅

    - ~~compter nb commandes où `estimated_delivery` - `delivered_customer` > 3~~ ✅
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    J'ai besoin de la date de la commande la plus récente

    **2018-10-17 17:30:18**
    """
    )
    return


@app.cell
def _(engine, mo, orders):
    _df = mo.sql(
        f"""
        SELECT
            MAX(order_purchase_timestamp)
        FROM
            orders
        """,
        engine=engine,
    )
    return


@app.cell
def _(engine, mo, orders):
    _df = mo.sql(
        f"""
        SELECT
            DATE(MAX(order_purchase_timestamp),"-3 months")
        FROM
            orders
        """,
        engine=engine,
    )
    return


@app.cell
def _(engine, mo, orders):
    _df = mo.sql(
        f"""
        SELECT
            order_id,
            customer_id,
            order_purchase_timestamp,
            JULIANDAY (order_delivered_customer_date) - JULIANDAY (order_estimated_delivery_date) AS retard_livraison
        FROM
            orders
        WHERE
            order_status != 'canceled'
            AND order_purchase_timestamp >= (
                SELECT
                    DATE(MAX(order_purchase_timestamp), '-3 months')
                FROM
                    orders
            )
            AND retard_livraison > 3
        ORDER BY
            retard_livraison DESC;
        """,
        engine=engine,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Question 2""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Question 2
    **Qui sont les vendeurs ayant généré un chiffre d'affaires de plus de 100 000 Real sur des commandes livrées via Olist ?**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - ~~Charger la table `order_items`~~ ✅
    - ~~Selectionner colonnes `seller_id` et `price`~~ ✅
    - ~~Calculer la somme de `price_id` et renommer en chiffre d'affaire~~ ✅
    - ~~Grouper par `seller_id`~~ ✅
    - ~~Conserver les lignes dont chiffres d'affaires supérieurs à 100 000~~ ✅
    - ~~Classer les chiffres d'affaires par ordre croissant~~ ✅
    """
    )
    return


@app.cell
def _(engine, mo, order_items):
    _df = mo.sql(
        f"""
        SELECT
            seller_id,
            SUM(price) AS chiffre_affaire
        FROM
            order_items
        GROUP BY
            seller_id
        HAVING
            chiffre_affaire > 100000
        ORDER BY
            chiffre_affaire DESC
        """,
        engine=engine,
    )
    return


@app.cell
def _(engine, mo, sqlite_master):
    _df = mo.sql(
        f"""
        SELECT * FROM sqlite_master WHERE type='table'
        """,
        engine=engine,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Question 3""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Question 3
    **Qui sont les nouveaux vendeurs (moins de 3 mois d'ancienneté) qui sont déjà très engagés avec la plateforme (ayant déjà vendu plus de 30 produits) ?**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - ~~Trouver date première vente des vendeurs **MIN()**~~ ✅
    - ~~Grouper par vendeur~~ ✅
    - ~~Filtrer pour conserver première vente datant de moins de 3 mois~~ ✅
    - ~~Aggréger avec la somme du nombre de produits vendus~~ ✅
    - ~~Filtrer pour conserver les vendeurs où nb_produits_vendu > 30~~ ✅
    """
    )
    return


@app.cell
def _(engine, mo, order_items, orders):
    _df = mo.sql(
        f"""
        SELECT
            order_items.seller_id,
            orders.order_id,
            MIN(orders.order_purchase_timestamp) AS premiere_vente,
            COUNT(order_items.product_id) AS nombre_produits_vendus
        FROM
            order_items,
            orders
        WHERE
            order_items.order_id = orders.order_id
        GROUP BY
            order_items.seller_id
        HAVING
            premiere_vente >= DATE(
                (
                    SELECT
                        MAX(order_purchase_timestamp)
                    FROM
                        orders
                ),
                '-3 months'
            )
            AND COUNT(order_items.product_id) > 30
        ORDER BY
            premiere_vente DESC
        """,
        engine=engine,
    )
    return


@app.cell
def _(engine, mo, orders):
    _df = mo.sql(
        f"""
        SELECT
            *
        FROM
            orders
        """,
        engine=engine,
    )
    return


@app.cell
def _(engine, mo, order_items):
    _df = mo.sql(
        f"""
        SELECT
            *
        FROM
            order_items
        """,
        engine=engine,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Question 4""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | Question 4

    **Quels sont les 5 codes postaux, enregistrant plus de 30 reviews, avec le pire review score moyen sur les 12 derniers mois ?**
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    - ~~Zip code de customer `customer_zip_code_preview`~~✅
    - ~~collecter les review des 12 derniers mois~~✅
    - ~~Aggréger le nombre de review (COUNT)~~✅
    - ~~filtrer pour conserver nb review > 30~~ ✅
    - ~~calculer la moyenne des review~~✅
    - ~~Conserver les 5 derniers code postaux (LIMIT)~~✅
    - ~~Trier les notes par ordre croissant~~✅
    """
    )
    return


@app.cell
def _(customers, engine, mo, order_reviews, orders):
    _df = mo.sql(
        f"""
        SELECT
            customers.customer_zip_code_prefix AS code_postal,
            AVG(order_reviews.review_score) AS note_moyenne,
            COUNT(order_reviews.review_score) AS nombre_commentaires
        FROM
            customers,
            order_reviews,
            orders
        WHERE
            order_reviews.order_id = orders.order_id
            AND customers.customer_id = orders.customer_id
            AND order_reviews.review_creation_date >= DATE(
                (
                    SELECT
                        MAX(review_creation_date)
                    FROM
                        order_reviews
                ),
                "-12 months"
            )
        GROUP BY
            code_postal
        HAVING
            nombre_commentaires > 30
        ORDER BY
            note_moyenne ASC
        LIMIT
            5
        """,
        engine=engine,
    )
    return


if __name__ == "__main__":
    app.run()
