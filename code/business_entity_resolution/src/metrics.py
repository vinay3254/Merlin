def f_beta_score(predicted: set, actual: set, beta: float = 0.5) -> float:
    if not actual and not predicted:
        return 1.0
    if not predicted or not actual:
        return 0.0
    true_positives = len(predicted & actual)
    if true_positives == 0:
        return 0.0
    precision = true_positives / len(predicted)
    recall = true_positives / len(actual)
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def macro_f_beta(predictions: dict, actuals: dict, beta: float = 0.5) -> float:
    if not actuals:
        return 0.0
    scores = [
        f_beta_score(predictions.get(key, set()), actual, beta=beta)
        for key, actual in actuals.items()
    ]
    return sum(scores) / len(scores)
