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
- [Phase 1 — Définition du problème & KPIs](#phase-1--définition-du-problème--kpis)
- [Phase 2 — Collecte & audit des données](#phase-2--collecte--audit-des-données)
- [Phase 3 — Exploration & analyse statistique](#phase-3--exploration--analyse-statistique)
- [Phase 4 — Modélisation prédictive & interprétabilité](#phase-4--modélisation-prédictive--interprétabilité)

---

## Présentation

Le churn (perte de clients) est un enjeu majeur en e-commerce : retenir un client
coûte généralement moins cher que d'en acquérir un nouveau. L'objectif du projet est
de construire un système capable d'identifier les clients à risque de churn à
90 jours, à partir de leur historique d'achats, afin de cibler les actions de
rétention.

> **Question décisionnelle centrale**
> Quels sont les facteurs prédictifs de l'attrition à 90 jours, et comment les
> utiliser pour identifier et retenir proactivement les clients à risque ?

---

## Architecture du projet

```
dddm-ecommerce-churn/
├── data/
│   ├── raw/                          # données brutes 
│   │   ├── data.csv                  # Source 1 — Kaggle E-Commerce Data
│   │   └── online_retail_II.xlsx     # Source 2 — UCI Online Retail II
│   └── processed/                    # données nettoyées 
│       ├── clean_transactions.csv
│       └── rfm_segmented.csv
├── notebooks/
│   ├── 01_problem_definition.ipynb        # Phase 1
│   ├── 02_data_audit.ipynb                # Phase 2
│   ├── 03_eda_statistical_analysis.ipynb  # Phase 3
│   └── 04_modeling_interpretability.ipynb # Phase 4
├── src/
│   └── preprocessing.py              # chargement, audit, nettoyage, enrichissement
├── models/                           # modèle entraîné, scaler, liste de variables
├── dashboard/
│   └── app.py                        # dashboard Streamlit
├── images/                           # figures produites par les notebooks
├── reports/                          # livrables 
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn plotly streamlit scipy jupyter openpyxl
```

ou via le fichier de dépendances :

```bash
pip install -r requirements.txt
```

Lancer les notebooks :

```bash
jupyter lab
```

---

## Phase 1 — Définition du problème & KPIs

Cadrage métier de la problématique de churn : formulation de la question
décisionnelle, définition des KPIs primaires (taux d'attrition, CA sauvegardé) et
secondaires (récence, conversion des campagnes), construction d'un KPI Tree reliant
objectifs stratégiques et métriques opérationnelles, puis Business Case avec
estimation du ROI prévisionnel.

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
(référentiel [REST Countries](https://restcountries.com/)), le jeu de données final
est consolidé dans `data/processed/clean_transactions.csv`.

Notebook : `notebooks/02_data_audit.ipynb` · Code : `src/preprocessing.py`

## Phase 3 — Exploration & analyse statistique

Construction de la table RFM (Récence, Fréquence, Montant) par client et définition
de la cible churn (inactivité > 90 jours). Analyse exploratoire complète :
distributions et asymétries, détection et traitement des valeurs aberrantes
(winsorisation), corrélations, tests statistiques (t-test, chi-deux, Mann-Whitney) et
segmentation par clustering K-Means. La table segmentée est sauvegardée dans
`data/processed/rfm_segmented.csv`.

Notebook : `notebooks/03_eda_statistical_analysis.ipynb`

## Phase 4 — Modélisation prédictive & interprétabilité

Entraînement et comparaison de trois modèles de classification (régression
logistique, Random Forest, XGBoost) pour prédire le churn à partir du comportement
d'achat. La récence, qui définit mécaniquement la cible, est exclue des variables
explicatives pour éviter toute fuite de données. Après validation croisée et réglage
des hyperparamètres, XGBoost atteint une AUC-ROC d'environ 0,87. L'interprétabilité
est assurée par les valeurs SHAP (importance globale et explication locale). Le modèle
final est sérialisé dans `models/`.

Notebook : `notebooks/04_modeling_interpretability.ipynb`
