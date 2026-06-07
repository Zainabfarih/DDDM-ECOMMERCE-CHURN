# Plan d'A/B Test — Campagne de Réactivation des Clients à Risque

**Projet :** Prédiction du churn client en e-commerce
**Recommandation testée :** R1 — Campagne de réactivation personnalisée (email + offre)

---

## 1. Contexte et objectif

Le modèle de churn (Phase 4) identifie un segment de **3 486 clients à risque**
(taux de churn 76 %). La recommandation R1 propose de leur adresser une campagne de
réactivation personnalisée. L'A/B test vise à **mesurer l'effet causal** de cette
campagne sur le taux de reconversion, avant tout déploiement à grande échelle.

## 2. Hypothèses

- **H0 (nulle)** : la campagne n'a aucun effet — `p_traitement = p_contrôle`.
- **H1 (alternative)** : la campagne augmente le taux de reconversion —
  `p_traitement > p_contrôle`.

## 3. Conception expérimentale

| Paramètre | Valeur |
|-----------|--------|
| Population | Clients classés « à risque » par le modèle |
| Groupe contrôle | Aucune campagne |
| Groupe traitement | Email + offre personnalisée |
| Répartition | 50 / 50, affectation aléatoire |
| Métrique primaire | Taux de reconversion à 30 jours (≥ 1 achat) |
| Métriques secondaires | Panier moyen, nombre de commandes, taux de désinscription |

## 4. Dimensionnement statistique

| Paramètre | Valeur |
|-----------|--------|
| Taux de référence (contrôle) | 5 % |
| Effet minimal détectable (MDE) | +2 points (→ 7 %) |
| Seuil de signification α | 0,05 (bilatéral) |
| Puissance 1 − β | 0,80 |
| **Taille par groupe** | **≈ 2 213 clients** |
| **Taille totale** | **≈ 4 426 clients** |
| **Durée** | 30 jours (≈ cycle d'achat moyen) |

Calcul fondé sur la formule de comparaison de deux proportions
(test bilatéral, `z_α/2 = 1,96`, `z_β = 0,84`).

## 5. Protocole d'exécution

1. Geler la liste des clients à risque à la date de lancement.
2. Affecter aléatoirement chaque client au groupe contrôle ou traitement.
3. Diffuser la campagne au seul groupe traitement.
4. Observer les achats pendant 30 jours sans autre sollicitation.
5. Comparer les taux de reconversion (test du chi-deux) au terme de la période.

## 6. Critère de décision

- **p-value < 0,05** → H0 rejetée : la campagne a un effet significatif → déploiement.
- **p-value ≥ 0,05** → effet non démontré : ne pas généraliser, réviser l'offre.

## 7. Métriques de suivi post-décision

| Métrique | Fréquence | Objectif |
|----------|-----------|----------|
| Taux de churn à 90 jours | Mensuelle | Baisse |
| Taux de reconversion des campagnes | Par campagne | ≥ 7 % |
| CA récupéré sur clients à risque | Mensuelle | Croissant |
| Précision du modèle (drift) | Trimestrielle | AUC ≥ 0,80 |
| Taux de désinscription | Par campagne | < 2 % |

## 8. Risques et limites

- **Saisonnalité** : la période de test peut coïncider avec un pic/creux saisonnier ;
  prévoir un test sur une fenêtre représentative.
- **Effet de nouveauté** : un gain initial peut s'estomper ; suivre la rétention au-delà
  de 30 jours.
- **Contamination** : éviter que le groupe contrôle reçoive d'autres sollicitations
  marketing pendant le test.
