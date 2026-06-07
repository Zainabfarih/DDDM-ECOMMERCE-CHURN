# Prédiction du Churn Client en E-commerce

Système d'aide à la décision pour anticiper l'attrition (churn) des clients d'un
site e-commerce et orienter les actions de rétention.

**Projet en binôme** — Zainab Farih & Assia Rguibi
**Filière** GL — Groupe GL2

---

## Table des matières

- [Présentation](#présentation)
- [Architecture du projet](#architecture-du-projet)
- [Installation](#installation)
- [Lancement des notebooks](#lancement-des-notebooks)
- [Lancement du dashboard](#lancement-du-dashboard)
- [Phase 1 — Définition du problème & KPIs](#phase-1--définition-du-problème--kpis)
- [Phase 2 — Collecte & audit des données](#phase-2--collecte--audit-des-données)
- [Phase 3 — Exploration & analyse statistique](#phase-3--exploration--analyse-statistique)
- [Phase 4 — Modélisation prédictive & interprétabilité](#phase-4--modélisation-prédictive--interprétabilité)
- [Phase 5 — Dashboard interactif](#phase-5--dashboard-interactif)
- [Reproductibilité](#reproductibilité)

---

## Présentation

Le churn (perte de clients) est un enjeu majeur en e-commerce : retenir un client
coûte généralement moins cher que d'en acquérir un nouveau. L'objectif du projet
est de construire un système capable d'identifier les clients à risque de churn
à 90 jours, à partir de leur historique d'achats, afin de cibler les actions de
rétention.

> **Question décisionnelle centrale**
> Quels sont les facteurs prédictifs de l'attrition à 90 jours, et comment les
> utiliser pour identifier et retenir proactivement les clients à risque ?

Livrables :

- **Notebooks Jupyter** complets et reproductibles couvrant les phases 1 à 4.
- **Dashboard Streamlit** interactif (`streamlit_app/`) qui rejoue le pipeline,
  expose les analyses et permet le scoring d'un client ou d'un fichier complet.
- **Modèle entraîné** sérialisé dans `models/` (XGBoost, AUC ≈ 0,87 sur le
  jeu de test).

---

## Architecture du projet

```
dddm-ecommerce-churn/
├── data/
│   ├── raw/                              # Données brutes (Git)
│   │   ├── data.csv                      # Source 1 — Kaggle E-Commerce Data
│   │   └── online_retail_II.xlsx         # Source 2 — UCI Online Retail II
│   └── processed/                        # Données nettoyées
│       ├── clean_transactions.csv        # ~190 MB, regénérée par le pipeline
│       └── rfm_segmented.csv             # Table RFM par client + label Churn
│
├── notebooks/                            # Notebooks Jupyter (phases 1 à 4)
│   ├── 01_problem_definition.ipynb
│   ├── 02_data_audit.ipynb
│   ├── 03_eda_statistical_analysis.ipynb
│   └── 04_modeling_interpretability.ipynb
│
├── src/
│   └── preprocessing.py                  # Chargement, audit, nettoyage
│
├── models/                               # Artefacts entraînés
│   ├── churn_model_xgb.pkl               # XGBoost réglé (Phase 4)
│   ├── features.pkl                      # Liste ordonnée des features
│   ├── scaler.pkl                        # StandardScaler (Logistic baseline)
│   ├── kmeans.pkl                        # ⚙ regénéré par le dashboard
│   └── kmeans_scaler.pkl                 # ⚙ regénéré par le dashboard
│
├── streamlit_app/                        # Dashboard interactif (Phase 5)
│   ├── app.py                            # Point d'entrée
│   ├── pages/                            # 6 pages auto-découvertes
│   ├── components/                       # Helpers UI + Plotly factory
│   ├── services/                         # Pipeline + inférence
│   ├── utils/                            # Config, palette, segments
│   ├── assets/                           # CSS du design system
│   └── README.md                         # Doc spécifique au dashboard
│
├── images/                               # Figures produites par les notebooks
├── .streamlit/config.toml                # Thème Streamlit aligné sur le CSS
├── requirements.txt                      # Toutes les dépendances Python
└── README.md
```

---

## Installation

Prérequis : Python ≥ 3.11.

Cloner le dépôt puis installer toutes les dépendances en une seule commande :

```bash
git clone https://github.com/Zainabfarih/DDDM-ECOMMERCE-CHURN.git
cd DDDM-ECOMMERCE-CHURN
pip install -r requirements.txt
```

`requirements.txt` couvre à la fois les notebooks (numpy, pandas, scikit-learn,
xgboost, shap, matplotlib, seaborn, jupyter, openpyxl, …) et le dashboard
(streamlit, plotly, joblib).

> Sur Windows avec le Python du Microsoft Store, la commande `streamlit` n'est
> pas toujours sur le `PATH`. Préférez `python -m streamlit …` (cf. plus bas).

---

## Lancement des notebooks

```bash
jupyter lab            # ou : jupyter notebook
```

Puis ouvrir les notebooks dans l'ordre 01 → 04. Chaque notebook est autonome :

| Notebook | Produit | Temps approximatif |
|---|---|---|
| `01_problem_definition.ipynb` | KPI Tree, Business Case | < 1 min |
| `02_data_audit.ipynb` | `data/processed/clean_transactions.csv` | 1–2 min |
| `03_eda_statistical_analysis.ipynb` | `data/processed/rfm_segmented.csv` | 2–3 min |
| `04_modeling_interpretability.ipynb` | `models/churn_model_xgb.pkl`, `features.pkl`, `scaler.pkl`, figures SHAP | 3–5 min |

Les artefacts produits sont commit-trackés (modèle entraîné + RFM segmenté) afin
que le dashboard puisse fonctionner sans avoir à rejouer toute la chaîne.

---

## Lancement du dashboard

Le dashboard Streamlit est dans `streamlit_app/`. Depuis la racine du projet :

```bash
python -m streamlit run streamlit_app/app.py
```

Ouverture automatique sur `http://localhost:8501`. Si le port est occupé,
ajoutez `--server.port 8502`.

Au premier lancement, la page d'accueil détecte les artefacts manquants et
propose un bouton **« Préparer les données »** qui rejoue le pipeline.

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

> Sur Windows avec le Python du Microsoft Store, la commande `streamlit` n'est
> pas toujours sur le `PATH`. Le préfixe `python -m` ci-dessus contourne le
> problème.

### Déploiement en ligne (optionnel)

Le dashboard est compatible **Streamlit Community Cloud** :

1. Forker / pusher ce dépôt sur GitHub.
2. Sur [share.streamlit.io](https://share.streamlit.io), créer une nouvelle app
   pointant vers `streamlit_app/app.py` du dépôt.
3. Streamlit Cloud installera automatiquement `requirements.txt` et démarrera
   l'application.

---

## Phase 1 — Définition du problème & KPIs

Cadrage métier de la problématique de churn : formulation de la question
décisionnelle, définition des KPIs primaires (taux d'attrition, CA sauvegardé)
et secondaires (récence, conversion des campagnes), construction d'un KPI Tree
reliant objectifs stratégiques et métriques opérationnelles, puis Business Case
avec estimation du ROI prévisionnel.

Notebook : `notebooks/01_problem_definition.ipynb`

## Phase 2 — Collecte & audit des données

Combinaison de deux jeux de données transactionnels réels :

| # | Source | Format | Volume |
|---|--------|--------|--------|
| 1 | [Kaggle E-Commerce Data](https://www.kaggle.com/datasets/carrie1/ecommerce-data) | CSV | 541 909 lignes |
| 2 | [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | Excel | 1 067 371 lignes |

Les deux sources sont harmonisées puis concaténées. L'audit évalue complétude,
cohérence, fraîcheur et doublons (notamment les doublons inter-sources sur la
période commune 2010-2011). Après nettoyage et enrichissement géographique
(référentiel [REST Countries](https://restcountries.com/)), le jeu de données
final est consolidé dans `data/processed/clean_transactions.csv`.

Notebook : `notebooks/02_data_audit.ipynb` · Code : `src/preprocessing.py`

## Phase 3 — Exploration & analyse statistique

Construction de la table RFM (Récence, Fréquence, Montant) par client et
définition de la cible churn (inactivité > 90 jours). Analyse exploratoire
complète : distributions et asymétries, détection et traitement des valeurs
aberrantes (winsorisation), corrélations, tests statistiques (t-test, chi-deux,
Mann-Whitney) et segmentation par clustering K-Means. La table segmentée est
sauvegardée dans `data/processed/rfm_segmented.csv`.

Notebook : `notebooks/03_eda_statistical_analysis.ipynb`

## Phase 4 — Modélisation prédictive & interprétabilité

Entraînement et comparaison de trois modèles de classification (régression
logistique, Random Forest, XGBoost) pour prédire le churn à partir du
comportement d'achat. La récence, qui définit mécaniquement la cible, est
exclue des variables explicatives pour éviter toute fuite de données. Après
validation croisée et réglage des hyperparamètres, XGBoost atteint une AUC-ROC
d'environ 0,87. L'interprétabilité est assurée par les valeurs SHAP (importance
globale et explication locale). Le modèle final est sérialisé dans `models/`.

Notebook : `notebooks/04_modeling_interpretability.ipynb`

## Phase 5 — Dashboard interactif

Le dashboard Streamlit (`streamlit_app/`) consolide les phases précédentes en
une application unique :

- **Accueil** : vue d'ensemble + bouton de préparation des données avec barre
  de progression en temps réel.
- **Données** : sources combinées, table RFM, top produits et marchés.
- **Qualité** : audit complétude, doublons cross-source, biais géographique.
- **Segments & EDA** : distributions RFM, tests Mann-Whitney, K-Means,
  saisonnalité.
- **Scoring client** : score individuel à partir d'un profil RFM saisi à la
  main, avec positionnement par rapport au segment d'appartenance.
- **Scoring par lot** : import d'un CSV de clients, retour des probabilités et
  niveaux de risque, export du résultat.
- **Performance** : ROC, matrice de confusion, importance des variables,
  figures SHAP.

### Pipeline de prédiction

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

### Personnalisation

- `streamlit_app/utils/config.py` centralise les chemins, la palette pastel
  et les seuils métier.
- `.streamlit/config.toml` aligne les couleurs de base de Streamlit sur le
  design system.
- `streamlit_app/assets/styles.css` est l'unique source du thème (variables
  CSS, classes utilitaires `cg-*`). Modifiez-le pour rebrander l'application.

Tous les graphiques utilisent la palette pastel du notebook EDA, le pipeline
de prédiction est strictement identique à celui du notebook 04, et les
résultats sont persistés en `session_state` pour survivre aux interactions.

---

## Reproductibilité

- **Versions plancher** des dépendances pinned dans `requirements.txt`,
  testées sur Python 3.11 et 3.13.
- **Données brutes** versionnées dans `data/raw/` (Git LFS non requis).
- **Artefacts modèle** versionnés dans `models/` (`*_xgb.pkl`, `features.pkl`,
  `scaler.pkl`) — la même prédiction sera obtenue d'une machine à l'autre.
- **Table RFM** versionnée dans `data/processed/rfm_segmented.csv`. Le très
  gros `clean_transactions.csv` (≈ 190 MB) est volontairement gitignoré et
  reconstruit à la volée par le pipeline (`src/preprocessing.py` ou la page
  d'accueil du dashboard).
- **Seeds** fixées (`random_state=42`) dans tous les modèles et splits.
- **Aucune connexion Internet requise** sauf pour l'enrichissement
  géographique optionnel (REST Countries) — décochable dans le dashboard.
