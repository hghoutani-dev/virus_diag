# virus_diag — exercice POO adaptée à l'IA

## Arborescence
```text
virus_diag/
├── data/patient_data.csv
├── core/
│   ├── dataset.py
│   ├── model.py
│   ├── logistic_regression.py
│   ├── neural_network.py
│   └── optimizer.py
├── pipeline/
│   ├── trainer.py
│   └── evaluator.py
├── utils/
│   ├── preprocessing.py
│   └── metrics.py
└── app/
    ├── api.py
    └── interface_clinique.py
```

## Installation
```bash
pip install -r requirements.txt
```

## Entraîner
```bash
python -m pipeline.trainer --model logistic --output models/logistic.joblib
python -m pipeline.trainer --model neural_network --output models/neural_network.joblib
```

## Évaluer
```bash
python -m pipeline.evaluator --model models/logistic.joblib
```

## Diagnostic
```bash
python -m app.interface_clinique --model models/logistic.joblib
```

## API
```bash
python -m app.api
```
