# ChurnGuard — application Streamlit

Tableau de bord interactif pour le système d'aide à la décision anti-churn
e-commerce du projet DDDM.

## Architecture

```
streamlit_app/
├── app.py                          # Page d'accueil + préparation des données
├── requirements.txt                # Dépendances spécifiques à l'application
├── assets/
│   └── styles.css                  # Design system (CSS unique)
├── components/
│   ├── ui.py                       # Primitives UI (hero, page_header, metric, …)
│   └── charts.py                   # Factory Plotly (palette unifiée)
├── pages/                          # Pages auto-découvertes par Streamlit
│   ├── 1_Donnees.py                # Sources, RFM, top produits, marchés
│   ├── 2_Qualite.py                # Audit, complétude, doublons, biais
│   ├── 3_Segments.py               # Distributions, tests, K-Means, saisonnalité
│   ├── 4_Scoring_client.py         # Scoring individuel + comparaison segment
│   ├── 5_Scoring_par_lot.py        # Scoring d'un CSV de clients
│   └── 6_Performance.py            # ROC, matrice de confusion, SHAP
├── services/
│   ├── data_service.py             # Chargement + cache des artefacts
│   ├── model_service.py            # Inférence + feature engineering
│   └── pipeline.py                 # Reconstruction du pipeline de données
└── utils/
    └── config.py                   # Chemins, palette pastel, segments, constantes
```

> Les données et les modèles vivent **à la racine du projet** (`data/raw/`,
> `data/processed/`, `models/`), pas dans `streamlit_app/`. Les chemins sont
> centralisés dans `utils/config.py` et résolus à partir de la racine.

## Installation

Depuis la racine du projet :

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app/app.py
```

L'application s'ouvre sur `http://localhost:8501`.

> Si la commande `streamlit` n'est pas reconnue (cas typique avec le Python du
> Microsoft Store), le préfixe `python -m` ci-dessus est suffisant.

## Première utilisation

Au premier lancement, certains artefacts dérivés ne sont pas encore présents
sur disque : `meta.json`, les agrégats `agg_*.csv`, et les modèles
`kmeans.pkl` + `kmeans_scaler.pkl`. La page d'accueil les détecte et propose
un bouton **« Préparer les données »**.

| Cas | Temps approximatif |
|---|---|
| `clean_transactions.csv` déjà présent (chemin rapide) | 10–20 s |
| Pipeline complet depuis les fichiers bruts (CSV Kaggle + Excel UCI) | ≈ 1 minute |

Deux options sont disponibles dans le panneau de préparation :

- **Enrichissement géographique (REST Countries API)** — ajoute région,
  sous-région et population. Décochez si pas d'accès réseau.
- **Audit complet** — force la relecture des fichiers bruts même si
  `clean_transactions.csv` existe déjà. Nécessaire pour produire les compteurs
  « doublons cross-source » de la page Qualité ; sinon, le chemin rapide
  suffit largement.

Pendant l'exécution, une barre de progression animée affiche l'étape en cours
(lecture des sources, nettoyage, RFM, K-Means, agrégats) avec le temps écoulé
en direct.

## Pipeline de prédiction

```text
Recency, Frequency, Monetary       # entrées brutes
       │
       ▼
AvgBasket    = Monetary / max(Frequency, 1)
LogFrequency = log1p(Frequency)
LogMonetary  = log1p(Monetary)
LogAvgBasket = log1p(AvgBasket)
Cluster      = KMeans.predict(scale(log1p(R, F, M)))
       │
       ▼
features = [LogFrequency, LogMonetary, LogAvgBasket, Cluster]
proba    = XGBoost.predict_proba(features)[:, 1]
```

> La récence sert au calcul du segment K-Means (qui devient une feature) mais
> n'entre **pas** directement dans le modèle final : elle définit la cible
> (`Churn = Recency > 90j`). Inclure `Recency` comme variable explicative
> provoquerait une fuite de données.

Les valeurs non numériques ou manquantes dans un CSV de scoring par lot sont
ignorées proprement et signalées à l'utilisateur (nombre de lignes écartées).

## Configuration

- `streamlit_app/utils/config.py` centralise les chemins, la palette pastel
  et les seuils métier.
- `.streamlit/config.toml` à la racine du dépôt pose les couleurs de base de
  Streamlit et désactive la collecte d'usage.
- `assets/styles.css` est l'unique source du design system. Toutes les
  classes utilitaires (`cg-*`) et les variables de thème vivent là —
  modifiez-le pour rebrander l'application.

## Bons réflexes en développement

- L'auto-rechargement de Streamlit (`runOnSave = true` dans
  `.streamlit/config.toml`) prend en compte les changements de pages
  immédiatement, mais les modifications dans `services/` ou `components/`
  nécessitent parfois de redémarrer le serveur (`Ctrl+C` puis relancer la
  commande) pour que les imports soient rejoués.
- Pour invalider les caches après une régénération manuelle des artefacts
  hors application, supprimez le contenu de `data/processed/` puis relancez
  la préparation depuis la page d'accueil.
