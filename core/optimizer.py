from typing import Any
from .model import Model
from .logistic_regression import LogisticRegressionModel
from .neural_network import NeuralNetworkModel


MODEL_REGISTRY = {
    "logistic": LogisticRegressionModel,
    "logistic_regression": LogisticRegressionModel,
    "neural_network": NeuralNetworkModel,
    "mlp": NeuralNetworkModel,
    "nn": NeuralNetworkModel
}


def build_model(name: str, **kwargs: Any) -> Model:
    """
    Patron Fabrique (Factory Pattern) pour instancier un modèle IA selon son nom.
    
    Args:
        name: Nom du modèle ("logistic", "neural_network", "mlp", etc.).
        **kwargs: Arguments optionnels passés au constructeur du modèle.
        
    Returns:
        Instance concrète sous-classe de Model.
        
    Raises:
        ValueError: Si le nom de modèle demandé n'est pas répertorié.
    """
    normalized_name = name.strip().lower()
    if normalized_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(set(MODEL_REGISTRY.keys())))
        raise ValueError(
            f"Modèle inconnu '{name}'. Modèles disponibles : {supported}"
        )

    model_class = MODEL_REGISTRY[normalized_name]
    return model_class(**kwargs)
