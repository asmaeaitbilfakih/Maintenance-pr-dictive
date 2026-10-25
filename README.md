# 🔧 Maintenance prédictive pour les équipements industriels

> Prédire une panne **avant** qu'elle arrive, à partir des capteurs (vibrations, température, pression) et de l'historique de maintenance.

**Auteures :** Asmae Ait Bilfakih et Asma Azizi
**Contexte :** industrie marocaine (cimenterie, agroalimentaire, automobile, aéronautique)
**Statut :** 🚧 en cours

---

## 1. C'est quoi ce projet ?

Dans une usine, une machine qui tombe en panne **sans prévenir** coûte cher : ligne à l'arrêt, production perdue, pièces importées avec de longs délais.

Il existe trois façons de faire de la maintenance :

| Type | Principe | Problème |
|---|---|---|
| Corrective | On répare après la panne | Arrêts imprévus, coûts élevés |
| Préventive | On change les pièces à date fixe | On remplace parfois des pièces encore bonnes |
| **Prédictive** | On surveille la machine et on intervient au bon moment | C'est l'objectif de ce projet |

Notre but : construire des modèles de Machine Learning qui répondent à trois questions.

1. **Cette machine va-t-elle tomber en panne bientôt ?** (classification)
2. **Se comporte-t-elle de façon anormale ?** (détection d'anomalies)
3. **Dans combien de temps risque-t-elle de tomber en panne ?** (modèles de survie)

---

## 2. Cas d'usage choisi

> À confirmer : **[ventilateurs et broyeurs en cimenterie / compresseurs en agroalimentaire / autre]**

Nous estimerons aussi le **gain métier** : combien d'heures d'arrêt peut-on éviter, et à quel coût ?

---

## 3. Données

| Jeu de données | Utilisation |
|---|---|
| **AI4I 2020** (10 000 lignes, ~3,4 % de pannes) | Classification, prise en main |
| **Données synthétiques** (générées par nous) | Dégradation progressive, historiques de maintenance, modèles de survie |
| *(optionnel)* NASA C-MAPSS, CWRU, NASA IMS | Durée de vie résiduelle, signaux de vibration |

---

## 4. Méthodes utilisées

- **Classification :** régression logistique (référence), Random Forest, XGBoost / LightGBM
- **Détection d'anomalies :** Isolation Forest, One-Class SVM, autoencodeur
- **Modèles de survie :** Kaplan-Meier, Cox, Weibull AFT, Random Survival Forest
- **Interprétabilité :** SHAP

---

## 5. Organisation du travail

### 👥 Répartition des rôles

| Étape | Asmae | Asma Azizi |
|---|---|---|
| **0. Cadrage** (secteur, objectifs, structure du repo) | ✅ ensemble | ✅ ensemble |
| **1. Données** | Générateur de données synthétiques (Weibull, vibrations, maintenances) | Chargement, nettoyage et EDA d'AI4I |
| **2. Features** | Features vibrations (RMS, kurtosis, crest factor, FFT) | Features temporelles (moyennes glissantes, tendances, temps depuis la dernière maintenance) |
| **3. Modèles** | Détection d'anomalies (Isolation Forest, One-Class SVM, autoencodeur) | Classification (LogReg, RF, XGBoost) puis modèles de survie |
| **4. Évaluation** | Métriques anomalies | Métriques classification et survie |
| **5. Synthèse** | ✅ comparaison des approches, SHAP, étude de coût | ✅ comparaison des approches, SHAP, étude de coût |
| **6. Livrables** | Rapport (partie anomalies, contexte) + dashboard | Rapport (partie classification et survie) + README final |

### 📅 Planning indicatif

| Semaine | Objectif |
|---|---|
| 1 | Cadrage, setup GitHub, EDA, premier jeu synthétique |
| 2 | Feature engineering, premiers modèles de base |
| 3 | Anomalies et survie, réglage des modèles |
| 4 | Comparaison, SHAP, coût métier |
| 5 | Rapport, dashboard, nettoyage du dépôt |

---

## 6. Structure du dépôt

```
Maintenance-pr-dictive/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            # données brutes (AI4I, ...)
│   ├── synthetic/      # données générées
│   └── processed/      # données prêtes pour les modèles
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_anomalies.ipynb
│   ├── 05_survie.ipynb
│   └── 06_comparaison.ipynb
├── src/
│   ├── generate_data.py
│   ├── features.py
│   └── models.py
├── reports/            # rapport final et figures
└── app/                # dashboard / API (optionnel)
```

---

## 7. Installation

```bash
git clone https://github.com/asmaeaitbilfakih/Maintenance-pr-dictive.git
cd Maintenance-pr-dictive
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Bibliothèques principales : `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `tensorflow` ou `torch`, `lifelines`, `scikit-survival`, `shap`, `matplotlib`, `streamlit`.

---

## 8. Règles de collaboration (Git)

- Branche principale `main` : **on ne travaille pas directement dessus**.
- Une branche par tâche : `feature/donnees-synthetiques`, `feature/classification`, etc.
- Petits commits avec des messages clairs (ex. `Ajout des features FFT`).
- Pull Request et relecture par l'autre avant de fusionner dans `main`.
- Chaque notebook commence par une courte description de son objectif.

---

## 9. Points d'attention techniques

- **Fuite de données :** dans AI4I, exclure `TWF`, `HDF`, `PWF`, `OSF`, `RNF` des variables explicatives.
- **Découpage train/test :** par machine ou par période, jamais au hasard sur des séries temporelles.
- **Classes déséquilibrées :** ne pas utiliser l'accuracy. Utiliser PR-AUC, recall, F1 et un seuil basé sur le coût.
- **Censure :** garder les machines encore en marche en fin d'observation dans les modèles de survie.

---

## 10. Résultats

> À compléter au fur et à mesure.

| Modèle | PR-AUC | Recall | F1 | Remarques |
|---|---|---|---|---|
| Régression logistique | | | | |
| Random Forest | | | | |
| XGBoost | | | | |
| Autoencodeur | | | | |

---

## 11. Perspectives

- Tester sur des données réelles d'un partenaire industriel
- Déployer une API de scoring de risque
- Alertes automatiques et planification des interventions

---

## 📚 Références

- AI4I 2020 Predictive Maintenance Dataset (UCI)
- NASA C-MAPSS, NASA IMS Bearing Dataset, CWRU Bearing Dataset
- Documentation `lifelines`, `scikit-survival`, `scikit-learn`

## 📄 Master

AI
