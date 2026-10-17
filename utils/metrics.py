from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calcule l'ensemble complet des métriques de classification clinique.
    
    Args:
        y_true: Vraies étiquettes de diagnostic (0 = Sain, 1 = Infecté).
        y_pred: Classes prédites par le modèle (0 ou 1).
        y_proba: Probabilités prédites pour la classe positive (optionnel, pour ROC-AUC).
        
    Returns:
        Dictionnaire contenant les métriques scalaires et détaillées.
    """
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Sain", "Infecté"],
            zero_division=0
        )
    }

    if y_proba is not None:
        try:
            # Si matrice (n, 2), extraire la colonne de la classe 1 (Infecté)
            if hasattr(y_proba, "ndim") and y_proba.ndim == 2:
                scores = y_proba[:, 1]
            else:
                scores = y_proba
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def display_metrics(metrics: Dict[str, Any]) -> None:
    """Affiche élégamment les métriques dans la console."""
    print("=" * 50)
    print("           RÉSULTATS DE L'ÉVALUATION")
    print("=" * 50)
    print(f"  Exactitude (Accuracy)  : {metrics['accuracy']:.4f}")
    print(f"  Précision (Precision) : {metrics['precision']:.4f}")
    print(f"  Rappel (Recall)        : {metrics['recall']:.4f}")
    print(f"  Score F1 (F1-Score)    : {metrics['f1']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"  Aire sous courbe ROC   : {metrics['roc_auc']:.4f}")
    
    cm = metrics.get("confusion_matrix")
    if cm:
        print("\nMatrice de Confusion :")
        print(f"  [VN={cm[0][0]:3d} | FP={cm[0][1]:3d}]  (Réels Sains)")
        print(f"  [FN={cm[1][0]:3d} | VP={cm[1][1]:3d}]  (Réels Infectés)")

    print("\nRapport de classification détaillé :")
    print(metrics.get("classification_report", ""))
    print("=" * 50)
