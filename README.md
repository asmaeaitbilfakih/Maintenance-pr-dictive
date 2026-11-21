# 🔧 Maintenance prédictive pour les équipements industriels
 
> Prédire une panne **avant** qu'elle arrive, à partir des capteurs (vibrations, température, pression) et de l'historique de maintenance.
 
**Auteures :** Asmae Ait Bilfakih et Asma Azizi
**Contexte :** industrie marocaine, avec un cas d'usage en **cimenterie**
**Statut :** 🚧 analyses terminées (classification, anomalies, survie, coût, SHAP, comparaison) ; rapport final à venir
 
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
Puis une quatrième : **laquelle de ces approches coûte le moins cher ?** (étude de coût en dirhams).
 
---
 
## 2. Cas d'usage choisi
 
**Cimenterie** : ventilateurs, broyeurs et compresseurs d'air, trois équipements tournants dont l'arrêt immobilise la ligne de production. Le générateur de données synthétiques simule ces trois types de machines.
 
Le **gain métier** est estimé en dirhams par heure de fonctionnement (voir §10). Les coûts utilisés sont des **hypothèses illustratives**, à remplacer par des chiffres justifiés.
 
---
 
## 3. Données
 
| Jeu de données | Utilisation |
|---|---|
| **AI4I 2020** (10 000 lignes, ~3,4 % de pannes), fichier `data/ai4i2020.csv` | Classification, prise en main |
| **Données synthétiques** (générées par nous) | Dégradation progressive, signaux de vibration, historiques de maintenance, anomalies, survie, coût |
| *(optionnel)* NASA C-MAPSS, CWRU, NASA IMS | Durée de vie résiduelle, signaux de vibration |
 
### Données synthétiques
 
`src/generate_data.py` simule 60 machines (3 types) qui vivent des **cycles successifs** : la durée de vie d'un cycle suit une loi de Weibull (usure), les capteurs dérivent avec l'âge, puis le cycle se termine par une panne ou reste **censuré** (machine encore en marche à la fin de l'observation).
 
| Fichier généré (`data/synthetic/`) | Contenu |
|---|---|
| `sensors.csv` | mesures toutes les 6 h : vibrations, température, pression, courant, RUL, étiquette « panne dans 48 h » |
| `signals.npy` + `signals_index.csv` | 4 048 fenêtres de signal brut de vibration (1 024 points à 2 048 Hz) |
| `survival.csv` | une ligne par cycle : durée, panne ou censure, type de machine, charge |
| `maintenance_log.csv`, `machines.csv` | historique des interventions, liste des machines |
 
⚠️ Ces données sont **simulées** : elles servent à valider la méthode, pas à mesurer la performance sur une vraie usine.
 
---
 
## 4. Méthodes utilisées
 
- **Classification :** régression logistique (référence), Random Forest, XGBoost
- **Détection d'anomalies :** Isolation Forest, One-Class SVM, autoencodeur (entraînés sans étiquette de panne)
- **Modèles de survie :** Kaplan-Meier, Weibull, Cox, Weibull AFT
- **Features :** vibrations (RMS, kurtosis, crest factor, FFT) et temporelles (moyennes et pentes glissantes, écart à la baseline du cycle)
- **Étude de coût :** comparaison de politiques de maintenance en MAD par heure
- **Interprétabilité :** SHAP
---
 
## 5. Organisation du travail
 
### 👥 Répartition des rôles (prévue)
 
| Étape | Asmae | Asma Azizi |
|---|---|---|
| **0. Cadrage** (secteur, objectifs, structure du repo) | ✅ ensemble | ✅ ensemble |
| **1. Données** | Générateur de données synthétiques (Weibull, vibrations, maintenances) | Chargement, nettoyage et EDA d'AI4I |
| **2. Features** | Features vibrations (RMS, kurtosis, crest factor, FFT) et temporelles | Illustration et vérification des features |
| **3. Modèles** | Détection d'anomalies, modèles de survie | Classification (AI4I et données synthétiques) |
| **4. Évaluation** | Métriques anomalies et survie, étude de coût | Métriques classification, seuil de coût du classifieur |
| **5. Synthèse** | Comparaison des approches | SHAP, comparaison des approches |
| **6. Livrables** | Rapport (contexte, anomalies, survie, coût) + dashboard | Rapport (classification, SHAP, comparaison) + README final |
 
### ✅ État d'avancement
 
| Étape | Notebook / fichier | État |
|---|---|---|
| Prétraitement AI4I | `01_pretraitement` | ✅ terminé |
| Features (génération + illustration) | `src/features.py`, `02_features` | ✅ terminé |
| Classification AI4I | `03_classification` | ✅ terminé |
| Classification sur données synthétiques | `03b_classification_synthetique` | ✅ terminé |
| Détection d'anomalies | `04_anomalies` | ✅ terminé |
| Modèles de survie | `05_survie` | ✅ terminé |
| Comparaison des approches | `06_comparaison` | ✅ terminé |
| Étude de coût (anomalies, survie) | `07_etude_cout` | ✅ terminé |
| SHAP | `08_shap` | ✅ terminé |
| Seuil de coût du classifieur | `09_seuil_cout_classifieur` | ✅ terminé |
| Rapport final | `reports/` | ⏳ à faire |
| Dashboard, tests | `app/`, `tests/` | ⏳ optionnel |
 
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
│   ├── ai4i2020.csv    # données AI4I
│   ├── synthetic/      # données générées (non versionnées, voir §7)
│   └── processed/      # données prêtes pour les modèles
├── notebooks/
│   ├── 01_pretraitement.ipynb              # AI4I : nettoyage, features, train/test
│   ├── 02_features.ipynb                   # pourquoi ces features aident à prédire une panne
│   ├── 03_classification.ipynb             # AI4I : LogReg, Random Forest, XGBoost
│   ├── 03b_classification_synthetique.ipynb # mêmes modèles sur les données synthétiques
│   ├── 04_anomalies.ipynb                  # Isolation Forest, One-Class SVM, autoencodeur
│   ├── 05_survie.ipynb                     # Kaplan-Meier, Weibull, Cox, AFT, âge de remplacement
│   ├── 06_comparaison.ipynb                # comparaison des approches
│   ├── 07_etude_cout.ipynb                 # corrective vs âge fixe vs alarme anomalies (en MAD)
│   ├── 08_shap.ipynb                       # interprétabilité du classifieur
│   └── 09_seuil_cout_classifieur.ipynb     # seuil de décision choisi par le coût
├── src/
│   ├── generate_data.py                    # données synthétiques
│   └── features.py                         # features vibrations (FFT...) et temporelles
├── models/             # modèles entraînés (classification)
└── reports/
    ├── figures/        # graphiques
    └── tables/         # tableaux de résultats (CSV)
```
 
*À venir : rapport final, `app/` (dashboard Streamlit, optionnel), `tests/` (optionnel).*
 
---
 
## 7. Installation et reproduction
 
```bash
git clone https://github.com/asmaeaitbilfakih/Maintenance-pr-dictive.git
cd Maintenance-pr-dictive
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```
 
### Régénérer les données synthétiques
 
Les fichiers de `data/synthetic/` et les features ne sont **pas versionnés** (ils sont volumineux et se recréent à l'identique grâce à la graine `--seed 42`). Avant d'ouvrir les notebooks `02` à `09` (sauf `03` et `01`, qui utilisent AI4I) :
 
```bash
python src/generate_data.py      # crée data/synthetic/
python src/features.py           # crée data/processed/features_*.csv
```
 
Options utiles de `generate_data.py` : `--n-machines`, `--horizon-h`, `--preventive-h 1200` (ajoute une maintenance préventive fixe), `--signal-every 0` (sans signal brut).
 
Ensuite, exécuter les notebooks dans l'ordre. Ils détectent eux-mêmes la racine du projet : ils peuvent être lancés depuis `notebooks/`.
 
Bibliothèques principales : `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lifelines`, `shap`, `matplotlib`.
 
---
 
## 8. Règles de collaboration (Git)
 
- Branche principale `main` : **on ne travaille pas directement dessus**.
- Une branche par tâche : `feature/donnees-synthetiques`, `feature/classification`, etc.
- Petits commits avec des messages clairs (ex. `Ajout des features FFT`).
- Pull Request et relecture par l'autre avant de fusionner dans `main`.
- Toujours faire `git pull --rebase` avant `git push`.
- Chaque notebook commence par une courte description de son objectif.
---
 
## 9. Points d'attention techniques
 
- **Fuite de données :** dans AI4I, exclure `TWF`, `HDF`, `PWF`, `OSF`, `RNF` des variables explicatives. Sur les données synthétiques, exclure `rul_h`, `age_h`, `time_h` et les identifiants (le notebook `03b` vérifie cette règle par un garde-fou).
- **Découpage train/test :** par machine ou par période, jamais au hasard sur des séries temporelles. Les données synthétiques sont découpées **par machine**, avec le même découpage dans tous les notebooks pour que les résultats soient comparables.
- **Classes déséquilibrées :** ne pas utiliser l'accuracy. Utiliser PR-AUC, recall, F1 et un seuil basé sur le coût.
- **Censure :** garder les machines encore en marche en fin d'observation dans les modèles de survie (26 % des cycles synthétiques sont censurés).
- **Valeurs de `rul_h` manquantes :** normales pour les cycles censurés (durée de vie inconnue). Ne pas les supprimer en survie.
---
 
## 10. Résultats
 
### Classification (AI4I, jeu de test)
 
| Modèle | PR-AUC | Recall | Précision | F1 |
|---|---|---|---|---|
| Régression logistique | 0,403 | 0,838 | 0,170 | 0,283 |
| Random Forest | **0,899** | 0,838 | **0,983** | **0,905** |
| XGBoost | 0,861 | 0,853 | 0,716 | 0,779 |
 
Dans AI4I, les pannes sont générées par des règles : les scores élevés des modèles à base d'arbres reflètent cela et ne garantissent pas la même performance sur des données réelles.
 
### Classification (données synthétiques, 15 machines de test)
 
| Modèle | PR-AUC (CV par machine) | PR-AUC (test) | Recall | Précision | F1 |
|---|---|---|---|---|---|
| Régression logistique | 0,868 | 0,834 | 0,98 | 0,45 | 0,62 |
| Random Forest | 0,850 | 0,883 | 0,92 | 0,70 | 0,79 |
| XGBoost | 0,866 | **0,892** | 0,97 | 0,63 | 0,77 |
 
Un modèle au hasard aurait une PR-AUC d'environ 0,018. Les scores sont très élevés parce que la dégradation est **simulée de façon lisse** : à ne pas extrapoler à une vraie usine.
 
### Détection d'anomalies (données synthétiques, machines jamais vues)
 
Entraînement **sans étiquette de panne**, uniquement sur le début de vie des machines (300 premières heures après une maintenance). Seuil d'alarme : 99ᵉ percentile des scores sur du sain, 3 mesures consécutives.
 
| Modèle | ROC-AUC | PR-AUC | Pannes détectées | Délai d'anticipation médian | Fausses alarmes (sain) |
|---|---|---|---|---|---|
| Isolation Forest | 0,941 | 0,314 | 43/44 | 1 038 h | 0,9 % |
| One-Class SVM | **0,983** | **0,576** | 43/44 | 667 h | 0,2 % |
| Autoencodeur | 0,955 | 0,391 | 43/44 | 592 h | 0 % |
 
C'est un **détecteur de dégradation**, pas un prédicteur de date de panne : l'étiquette « panne dans 48 h » est très étroite par rapport au début de la dérive, ce qui limite la PR-AUC.
 
### Survie (données synthétiques)
 
| Type | Durée de vie médiane (Kaplan-Meier) | Forme de Weibull ρ | Âge de remplacement T* | Économie vs attendre la panne |
|---|---|---|---|---|
| Broyeur | 1 740 h | 1,77 | 860 h | ≈ 33 % |
| Ventilateur | 2 701 h | 2,33 | 1 210 h | ≈ 49 % |
| Compresseur | 2 721 h | 2,36 | 1 330 h | ≈ 49 % |
 
- ρ > 1 pour les trois types : le risque **augmente avec l'âge** (usure), donc la maintenance préventive est justifiée.
- La charge raccourcit la vie : +10 % de charge ≈ +37 % de risque de panne (Cox).
- ⚠️ T* repose sur l'hypothèse **coût d'une panne = 8 × coût d'une préventive**, valeur illustrative.
- Le C-index (≈ 0,67) est modeste : type et charge ne suffisent pas ; les capteurs apportent l'information manquante.
### Étude de coût et comparaison des approches (données synthétiques, machines jamais vues)
 
Coût moyen de maintenance par heure de fonctionnement. Une alarme déclenche une intervention préventive : une fausse alarme coûte une préventive inutile, une panne non détectée coûte une corrective. Les seuils sont choisis **par le coût** sur des machines de calibration.
 
| Approche | Étiquettes nécessaires | PR-AUC | MAD / h | Économie vs corrective |
|---|---|---|---|---|
| Corrective (attendre la panne) | non | n/a | 457 | 0 % |
| Survie : remplacement à âge fixe | oui (durées de vie) | n/a | 298 | 35 % |
| Anomalies (One-Class SVM) | non | 0,576 | 106 | 77 % |
| Classifieur (XGBoost, seuil coût) | oui (« panne < 48 h ») | 0,892 | 79 | 83 % |
| Idéale (limite théorique) | n/a | n/a | 78 | 83 % |
 
- Les approches sont **complémentaires** : la survie planifie, les anomalies surveillent sans étiquettes, le classifieur déclenche l'intervention quand les étiquettes existent.
- Pour le classifieur, le seuil choisi par le coût (0,8) donne un gain **négligeable** par rapport au seuil 0,5 (79,15 contre 79,40 MAD/h) : l'essentiel du gain vient du modèle, pas du réglage du seuil.
- Le classement corrective > âge fixe > anomalies > classifieur reste identique quand la perte de production par heure est multipliée par 0,1 à 4.
- Les écarts faibles entre approches proches (anomalies, classifieur) ne sont **pas significatifs** : 15 machines de test seulement.
### Interprétabilité (SHAP, classifieur)
 
- Variables les plus influentes : `vib_rms_ratio_baseline`, `vib_rms_slope_12` et `vib_kurtosis_mean_4`.
- Par capteur : vibrations (RMS, kurtosis, crest factor) ≈ 65 % de l'influence totale, température ≈ 18 %, courant ≈ 12 %, pression ≈ 6 %.
- Par type de variable : écart à la baseline ≈ 30 %, moyenne glissante ≈ 28 %, pente ≈ 23 %, mesure brute ≈ 14 % : les features d'**évolution** pèsent plus que les mesures brutes, ce qui justifie le feature engineering temporel.
- ⚠️ SHAP retrouve ce qu'on a mis dans le simulateur (la dégradation agit surtout sur les vibrations) et ne prouve pas une cause physique.
*Les chiffres exacts sont dans `reports/tables/` et dans les notebooks.*
 
---
 
## 11. Limites
 
- **Données synthétiques :** la dégradation est lisse et très lisible, donc tous les scores sont très élevés (PR-AUC 0,8 à 0,9 contre 0,018 au hasard). Sur de vraies machines, on s'attend à des résultats nettement plus faibles, et l'avantage des modèles supervisés sur les anomalies est probablement surestimé.
- **Coûts illustratifs :** seul le rapport coût d'une panne / coût d'une préventive (≈ 6) compte, et il doit être justifié avec des données de cimenterie.
- **Peu de machines :** 15 machines de test et 9 de calibration, donc des seuils bruités et des écarts faibles non significatifs.
- **Alarme immédiate :** on suppose qu'une alarme déclenche une préventive sans délai de planification ni de pièces.
---
 
## 12. Perspectives
 
- Remplacer les coûts illustratifs par des coûts réels de cimenterie
- Tester sur des données réelles d'un partenaire industriel
- Dashboard Streamlit, déploiement d'une API de scoring de risque
- Alertes automatiques et planification des interventions
- En production, stocker les mesures dans une base de séries temporelles (TimescaleDB, InfluxDB)
---
 
## 📚 Références
 
- AI4I 2020 Predictive Maintenance Dataset (UCI)
- NASA C-MAPSS, NASA IMS Bearing Dataset, CWRU Bearing Dataset
- Documentation `lifelines`, `scikit-learn`, `shap`
## 📄 Master
 
AI
