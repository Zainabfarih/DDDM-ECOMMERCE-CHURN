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
│   ├── raw/                          # données brutes (versionnées)
│   │   ├── data.csv                  # Source 1 — Kaggle E-Commerce Data
│   │   └── online_retail_II.xlsx     # Source 2 — UCI Online Retail II
│   └── processed/                    # données nettoyées (générées, non versionnées)
│       └── clean_transactions.csv
├── notebooks/
│   ├── 01_problem_definition.ipynb   # Phase 1
│   └── 02_data_audit.ipynb           # Phase 2
├── src/
│   └── preprocessing.py              # chargement, audit, nettoyage, enrichissement
├── dashboard/
│   └── app.py                        # dashboard Streamlit
├── images/                           # figures produites par les notebooks
├── reports/                          # livrables (A/B test plan, slides…)
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
