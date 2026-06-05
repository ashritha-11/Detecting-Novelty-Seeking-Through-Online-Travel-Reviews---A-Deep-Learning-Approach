import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model_weights(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model