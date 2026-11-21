def build_model(name):
    from .logistic_regression import LogisticRegressionModel
    from .neural_network import NeuralNetworkModel
    name = name.lower()
    if name in {"logistic","logistic_regression"}:
        return LogisticRegressionModel()
    if name in {"neural_network","mlp","nn"}:
        return NeuralNetworkModel()
    raise ValueError(f"Modèle inconnu : {name}")
