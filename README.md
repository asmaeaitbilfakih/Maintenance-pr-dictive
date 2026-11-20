# 🔧 Maintenance prédictive pour les équipements industriels
 
> Prédire une panne **avant** qu'elle arrive, à partir des capteurs (vibrations, température, pression) et de l'historique de maintenance.
 
**Auteures :** Asmae Ait Bilfakih et Asma Azizi
**Contexte :** industrie marocaine, avec un cas d'usage en **cimenterie**
**Statut :** 🚧 en cours (classification, anomalies, survie et étude de coût terminées ; comparaison finale, SHAP et rapport à venir)
 
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
 
**Cimenterie** : ventilateurs, broyeurs et compresseurs d'air, trois équipements tournants dont l'arrêt immobilise la ligne de production. Le générateur de données synthétiques simule ces trois types de machines.
 
Nous estimons aussi le **gain métier** : combien coûte chaque politique de maintenance (attendre la panne, remplacer à âge fixe, intervenir sur alerte du modèle) en dirhams par heure de fonctionnement (voir §10). Les coûts utilisés sont des **hypothèses illustratives**, à remplacer par des chiffres justifiés.
 
---
 
## 3. Données
 
| Jeu de données | Utilisation |
|---|---|
| **AI4I 2020** (10 000 lignes, ~3,4 % de pannes) | Classification, prise en main |
| **Données synthétiques** (générées par nous) | Dégradation progressive, signaux de vibration, historiques de maintenance, modèles de survie |
| *(optionnel)* NASA C-MAPSS, CWRU, NASA IMS | Durée de vie résiduelle, signaux de vibration |
 
### Données synthétiques
 
`src/generate_data.py` simule 60 machines (3 types) qui vivent des **cycles successifs** : la durée de vie d'un cycle suit une loi de Weibull (usure), les capteurs dérivent avec l'âge, puis le cycle se termine par une panne ou reste **censuré** (machine encore en marche à la fin de l'observation).
 
| Fichier généré (`data/synthetic/`) | Contenu |
|---|---|
| `sensors.csv` | mesures toutes les 6 h : vibrations, température, pression, courant, RUL, étiquette « panne dans 48 h » |
| `signals.npy` + `signals_index.csv` | 4 048 fenêtres de signal brut de vibration (1 024 points à 2 048 Hz) |
| `survival.csv` | une ligne par cycle : durée, panne ou censure, type de machine, charge |
| `maintenance_log.csv`, `machines.csv` | historique des interventions, liste des machines |
 
 Ces données sont **simulées** : elles servent à valider la méthode, pas à mesurer la performance sur une vraie usine.
 
---
 
## 4. Méthodes utilisées
 
- **Classification :** régression logistique (référence), Random Forest, XGBoost
- **Détection d'anomalies :** Isolation Forest, One-Class SVM, autoencodeur
- **Modèles de survie :** Kaplan-Meier, Weibull, Cox, Weibull AFT
- **Interprétabilité :** SHAP *(à venir)*
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
 
### ✅ État d'avancement
 
| Étape | État |
|---|---|
| Prétraitement et classification AI4I | ✅ terminé |
| Données synthétiques et features | ✅ terminé |
| Détection d'anomalies | ✅ terminé |
| Modèles de survie et première règle de maintenance préventive | ✅ terminé |
| Étude de coût (corrective vs âge fixe vs alarme anomalies) | ✅ terminé |
| Classification sur les données synthétiques | 🔄 en cours (Asma) |
| Comparaison des approches + SHAP | ⏳ à faire |
| Tests, dashboard, rapport | ⏳ à faire |
 
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
│   ├── raw/            # données brutes (AI4I)
│   ├── synthetic/      # données générées (non versionnées, voir §7)
│   └── processed/      # données prêtes pour les modèles
├── notebooks/
│   ├── 01_pretraitement.ipynb   # AI4I : nettoyage, features, train/test
│   ├── 03_classification.ipynb  # LogReg, Random Forest, XGBoost
│   ├── 04_anomalies.ipynb       # Isolation Forest, One-Class SVM, autoencodeur
│   ├── 05_survie.ipynb          # Kaplan-Meier, Weibull, Cox, AFT, âge de remplacement
│   └── 07_etude_cout.ipynb      # Corrective vs âge fixe vs alarme anomalies (en MAD)
├── src/
│   ├── generate_data.py         # données synthétiques
│   └── features.py              # features vibrations (FFT...) et temporelles
├── models/             # modèles entraînés (classification)
└── reports/
    ├── figures/        # graphiques
    └── tables/         # tableaux de résultats (CSV)
```
 
*À venir : `06_comparaison.ipynb` (comparaison des approches + SHAP), classification sur données synthétiques, `src/models.py`, `app/` (dashboard Streamlit), rapport final.*
 
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
 
Les fichiers de `data/synthetic/` et les features ne sont **pas versionnés** (ils sont volumineux et se recréent à l'identique grâce à la graine `--seed 42`). Avant d'ouvrir les notebooks `04` et `05` :
 
```bash
python src/generate_data.py      # crée data/synthetic/
python src/features.py           # crée data/processed/features_*.csv
```
 
Options utiles de `generate_data.py` : `--n-machines`, `--horizon-h`, `--preventive-h 1200` (ajoute une maintenance préventive fixe), `--signal-every 0` (sans signal brut).
 
Ensuite, exécuter les notebooks : `01`, `03` (AI4I), puis `04`, `05`, `07` (données synthétiques). Les notebooks détectent eux-mêmes la racine du projet : ils peuvent être lancés depuis `notebooks/`.
 
Bibliothèques principales : `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lifelines`, `matplotlib`.
 
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
 
- **Fuite de données :** dans AI4I, exclure `TWF`, `HDF`, `PWF`, `OSF`, `RNF` des variables explicatives.
- **Découpage train/test :** par machine ou par période, jamais au hasard sur des séries temporelles. Les données synthétiques sont découpées **par machine**.
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
 
### Détection d'anomalies (données synthétiques, machines jamais vues)
 
Entraînement **sans étiquette de panne**, uniquement sur le début de vie des machines (300 premières heures après une maintenance). Seuil d'alarme : 99ᵉ percentile des scores sur du sain, 3 mesures consécutives.
 
| Modèle | ROC-AUC | PR-AUC | Pannes détectées | Délai d'anticipation médian | Fausses alarmes (sain) |
|---|---|---|---|---|---|
| Isolation Forest | 0,941 | 0,314 | 43/44 | 1 038 h | 0,9 % |
| One-Class SVM | **0,983** | **0,576** | 43/44 | 667 h | 0,2 % |
| Autoencodeur | 0,955 | 0,374 | 43/44 | 592 h | 0 % |
 
C'est un **détecteur de dégradation**, pas un prédicteur de date de panne : l'étiquette « panne dans 48 h » est très étroite par rapport au début de la dérive, ce qui limite la PR-AUC. Le délai est long parce que la dérive simulée est lisse ; il serait plus court sur des données réelles.
 
### Survie (données synthétiques)
 
| Type | Durée de vie médiane (Kaplan-Meier) | Forme de Weibull ρ | Âge de remplacement T* | Économie vs attendre la panne |
|---|---|---|---|---|
| Broyeur | 1 740 h | 1,77 | 860 h | ≈ 33 % |
| Ventilateur | 2 701 h | 2,33 | 1 210 h | ≈ 49 % |
| Compresseur | 2 721 h | 2,36 | 1 330 h | ≈ 49 % |
 
- ρ > 1 pour les trois types : le risque **augmente avec l'âge** (usure), donc la maintenance préventive est justifiée.
- La charge raccourcit la vie : +10 % de charge ≈ +37 % de risque de panne (Cox).
-  T* et l'économie reposent sur l'hypothèse **coût d'une panne = 8 × coût d'une préventive**, valeur illustrative à remplacer par des coûts justifiés dans l'étude de coût.
- Le C-index (≈ 0,67) est modeste : type et charge ne suffisent pas ; les capteurs apporteront l'information manquante.
### Étude de coût (données synthétiques, machines jamais vues)
 
Coût moyen de maintenance par heure de fonctionnement, moyenné sur les trois types de machine. Une alarme déclenche une intervention préventive : une fausse alarme coûte une préventive inutile, une panne non détectée coûte une corrective. Le seuil d'alarme est choisi **par le coût** sur des machines de calibration.
 
| Politique | MAD / h | MAD / an / machine | Économie vs corrective |
|---|---|---|---|
| Corrective (attendre la panne) | 457 | 3,66 M | 0 % |
| Préventive à âge fixe (Weibull) | 298 | 2,38 M | 35 % |
| **Prédictive (alarme One-Class SVM)** | **106** | **0,85 M** | **77 %** |
| Idéale (limite théorique) | 78 | 0,62 M | 83 % |
 
-  **Les coûts sont illustratifs** (ex. perte de production du broyeur : 30 000 MAD/h). Seul le rapport coût d'une panne / coût d'une préventive (≈ 6) compte, et il doit être justifié avec des données de cimenterie.
- La dégradation est simulée de façon lisse et lisible : sur des données réelles, l'avantage de la politique prédictive sur l'âge fixe sera nettement plus faible.
- Le classement corrective > âge fixe > prédictive reste identique quand la perte de production par heure est multipliée par 0,1 à 4.
*Les chiffres exacts sont dans `reports/tables/` et dans les notebooks.*
 
---
 
## 11. Perspectives
 
- Comparer les trois approches sur un même critère et expliquer les modèles avec SHAP
- Ajouter la classification (données synthétiques) à l'étude de coût et à la comparaison finale
- Remplacer les coûts illustratifs par des coûts réels de cimenterie
- Dashboard Streamlit
- Tester sur des données réelles d'un partenaire industriel
- Déployer une API de scoring de risque, alertes automatiques et planification des interventions
- En production, stocker les mesures dans une base de séries temporelles (TimescaleDB, InfluxDB)
---
 
##  Références
 
- AI4I 2020 Predictive Maintenance Dataset (UCI)
- NASA C-MAPSS, NASA IMS Bearing Dataset, CWRU Bearing Dataset
- Documentation `lifelines`, `scikit-learn`
##  Master
 
AI
