# virus_diag — Diagnostic Viral Assisté par IA (POO & Machine Learning)

Système complet de diagnostic clinique orienté objet en Python (3.10+), conçu pour prédire si un patient est **Infecté** ou **Sain** à partir de ses signes vitaux et symptômes cliniques.

---

## 1. Présentation et Objectifs du Projet

Le projet `virus_diag` a pour but d'illustrer les meilleures pratiques de **Programmation Orientée Objet (POO)** appliquées au **Machine Learning** et aux applications cliniques :

- **Conception découplée et modulaire** : Séparation stricte entre la manipulation des données (`PatientDataset`), les modèles IA (`Model`), la phase d'entraînement (`Trainer`), l'évaluation (`Evaluator`), et l'inférence clinique (`ClinicalPredictor`).
- **Polymorphisme et Duck Typing** : La classe `ClinicalPredictor` interagit avec n'importe quelle architecture de modèle (Régression Logistique, Perceptron Multicouche, ou modèles futurs) sans modification de son code source.
- **Reproductibilité et Sérialisation** : Le pipeline complet de prétraitement (`ColumnTransformer`, standardisation, imputation) est encapsulé directement dans l'objet modèle et persisté via `joblib`.
- **Déploiement API REST** : Exposition d'une API web Flask avec endpoints de santé (`/health`) et de prédiction (`/predict`).

---

## 2. Structure et Architecture du Répertoire

```text
virus_diag/
│
├── data/
│   └── patient_data.csv          # Données sources des patients (600 lignes, 8 features, 1 target)
│
├── core/
│   ├── __init__.py
│   ├── dataset.py                # Classe PatientDataset (chargement, validation, split stratifié)
│   ├── model.py                  # Classe abstraite Model (ABC : fit, predict, predict_proba)
│   ├── logistic_regression.py   # Modèle concret LogisticRegressionModel avec pipeline scikit-learn
│   ├── neural_network.py         # Modèle concret NeuralNetworkModel (MLPClassifier)
│   └── optimizer.py              # Fabrique (Factory Pattern) build_model()
│
├── pipeline/
│   ├── __init__.py
│   ├── trainer.py                # Classe Trainer (orchestration d'entraînement et persistance)
│   └── evaluator.py              # Classe Evaluator (rapport métrique complet sur test set)
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py          # ColumnTransformer : imputation et standardisation automatique
│   └── metrics.py                # Calcul des métriques (Accuracy, Précision, Rappel, F1, ROC-AUC)
│
├── app/
│   ├── __init__.py
│   ├── api.py                    # API Flask (/health, /predict)
│   └── interface_clinique.py     # Classe ClinicalPredictor (décision médicale selon seuil 0.5)
│
├── models/
│   ├── logistic.joblib           # Modèle de régression logistique sérialisé
│   └── neural_network.joblib     # Modèle réseau de neurones sérialisé
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py           # Tests du chargement et du split stratifié
│   ├── test_models.py            # Tests de la fabrique et des modèles
│   ├── test_predictor.py         # Tests du seuil 0.5, polymorphisme et fallback
│   └── test_api.py               # Tests des routes /health et /predict
│
├── requirements.txt              # Dépendances Python minimales
└── README.md                     # Documentation complète
```

---

## 3. Données Cliniques (`data/patient_data.csv`)

Le fichier source comprend 600 observations de patients sans aucune valeur manquante ni doublon.

### Variables explicatives ($X$) :
1. **`age`** (int) : Âge du patient en années (1 à 90 ans).
2. **`temperature`** (float) : Température corporelle en °C (35.0 à 40.2 °C).
3. **`cough`** (binaire 0/1) : Présence de toux.
4. **`fatigue`** (binaire 0/1) : Sensation de fatigue intense.
5. **`sore_throat`** (binaire 0/1) : Maux de gorge.
6. **`breathing_difficulty`** (binaire 0/1) : Difficultés respiratoires / dyspnée.
7. **`oxygen_saturation`** (float) : Saturation en oxygène $SpO_2$ en % (90.5 à 100 %).
8. **`heart_rate`** (int) : Fréquence cardiaque en bpm (49 à 123 bpm).

### Variable cible ($y$) :
- **`infected`** (binaire 0/1) :
  - `0` : **Sain** (476 cas, 79.33%)
  - `1` : **Infecté** (124 cas, 20.67%)

---

## 4. Installation

### Prérequis
- Python 3.10 ou supérieur.

### Installation des dépendances
```bash
cd virus_diag
pip install -r requirements.txt
```

---

## 5. Guide d'Utilisation

### A. Entraîner les modèles IA

Entraîner et sauvegarder automatiquement les modèles dans `models/` :

```bash
# Entraîner la Régression Logistique (sauvegarde dans models/logistic.joblib)
python -m pipeline.trainer --model logistic

# Entraîner le Réseau de Neurones MLP (sauvegarde dans models/neural_network.joblib)
python -m pipeline.trainer --model neural_network
```

Options disponibles :
- `--model` : `logistic` ou `neural_network` (défaut: `logistic`)
- `--data` : chemin optionnel vers un fichier CSV personnalisé
- `--output` : chemin de destination personnalisé du fichier `.joblib`

### B. Évaluer la performance des modèles

Calculer les métriques complètes (Accuracy, Précision, Rappel, F1, ROC-AUC, matrice de confusion) :

```bash
# Évaluation de la régression logistique
python -m pipeline.evaluator --model models/logistic.joblib

# Évaluation du réseau de neurones
python -m pipeline.evaluator --model models/neural_network.joblib
```

### C. Exécuter un diagnostic clinique (CLI)

Tester un cas patient à l'aide de `ClinicalPredictor` :

```bash
python -m app.interface_clinique --model models/logistic.joblib
```

### D. Lancer et interroger l'API Flask

Démarrer le serveur HTTP local :

```bash
python -m app.api
```

L'API écoute par défaut sur `http://localhost:5000`.

#### Tester l'état de santé (`GET /health`)
```bash
curl http://localhost:5000/health
```
Réponse :
```json
{
  "expected_features": [
    "age",
    "temperature",
    "cough",
    "fatigue",
    "sore_throat",
    "breathing_difficulty",
    "oxygen_saturation",
    "heart_rate"
  ],
  "model_path": ".../models/logistic.joblib",
  "status": "ok"
}
```

#### Émettre un diagnostic (`POST /predict`)
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "age": 67,
       "temperature": 38.9,
       "cough": 1,
       "fatigue": 1,
       "sore_throat": 1,
       "breathing_difficulty": 1,
       "oxygen_saturation": 91.5,
       "heart_rate": 105
     }'
```
Réponse :
```json
{
  "diagnosis": "Infecté",
  "probability": 0.9586
}
```

Pour un patient sain :
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "age": 25,
       "temperature": 36.6,
       "cough": 0,
       "fatigue": 0,
       "sore_throat": 0,
       "breathing_difficulty": 0,
       "oxygen_saturation": 98.5,
       "heart_rate": 70
     }'
```
Réponse :
```json
{
  "diagnosis": "Sain",
  "probability": 0.0812
}
```

---

## 6. Exécution des Tests Automatisés

Le projet inclut une suite de tests unitaires couvrant les 4 axes clés (dataset, modèles, prédicteur clinique, API) :

```bash
pytest -v
```
ou via le module natif unittest :
```bash
python -m unittest discover -s tests
```

---

## 7. Concepts de POO appliqués à l'IA

### 1. Encapsulation
Chaque classe regroupe ses données et ses méthodes en protégeant son état interne :
- `LogisticRegressionModel` et `NeuralNetworkModel` encapsulent leur propre `ColumnTransformer` et leur estimateur scikit-learn. L'utilisateur externe n'a pas besoin de manipuler manuellement les scalers.
- `PatientDataset` encapsule la logique de validation et de découpage stratifié.

### 2. Abstraction
La classe abstraite `Model` (`core/model.py`) dérivée de `abc.ABC` définit un contrat public strict (`fit()`, `predict()`, `predict_proba()`). L'utilisateur manipule un modèle abstrait sans se soucier des détails mathématiques internes (optimiseur L-BFGS, descente de gradient stochastique, rétropropagation, etc.).

### 3. Héritage
`LogisticRegressionModel` et `NeuralNetworkModel` héritent de la classe de base `Model`. Elles réutilisent la même signature contractuelle tout en fournissant une implémentation spécialisée de l'entraînement et du calcul des probabilités.

### 4. Polymorphisme et Inversion de Dépendance
`ClinicalPredictor` ne dépend d'aucune classe concrète : il reçoit un objet respectant le contrat `Model` (ou duck typing avec `predict()`).
Ainsi, le même code clinique fonctionne indifféremment avec un modèle linéaire ou un réseau de neurones profond :

```text
                     ┌──────────────────┐
                     │   Model (ABC)    │
                     └────────┬─────────┘
                              │ (Héritage)
                 ┌────────────┴────────────┐
                 ▼                         ▼
    ┌─────────────────────────┐  ┌────────────────────┐
    │ LogisticRegressionModel │  │ NeuralNetworkModel │
    └────────────┬────────────┘  └─────────┬──────────┘
                 │                         │
                 └────────────┬────────────┘
                              │ (Injection de dépendance)
                              ▼
                   ┌───────────────────────┐
                   │   ClinicalPredictor   │
                   └───────────────────────┘
```
