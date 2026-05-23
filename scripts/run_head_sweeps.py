import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt

from utils.preprocess import Preprocessor
from models.TwinTransformer import TwinGDLTransformer
from utils.train import train_model
from utils.evaluate import evaluate_model
from scripts.download_datasets import download_and_prepare_dataset


def save_classwise_metrics(per_class, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    class_indices = list(range(len(per_class["precision"])))
    class_df = pd.DataFrame({
        "class": class_indices,
        "accuracy": per_class.get("accuracy", []),
        "precision": per_class.get("precision", []),
        "recall": per_class.get("recall", []),
        "f1": per_class.get("f1", []),
    })
    classwise_csv = os.path.join(save_dir, "classwise_metrics.csv")
    class_df.to_csv(classwise_csv, index=False)
    print(f"✅ Classwise metrics saved to {classwise_csv}")

    plt.figure(figsize=(14, 7))
    plt.plot(class_df["class"], class_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(class_df["class"], class_df["precision"], marker="o", label="Precision")
    plt.plot(class_df["class"], class_df["recall"], marker="o", label="Recall")
    plt.plot(class_df["class"], class_df["f1"], marker="o", label="F1 Score")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Classwise Metrics")
    plt.legend()
    plt.grid(True)
    classwise_plot = os.path.join(save_dir, "classwise_metrics_plot.png")
    plt.tight_layout()
    plt.savefig(classwise_plot)
    plt.close()
    print(f"📈 Classwise plot saved to {classwise_plot}")

    return class_df


def save_confusion_matrix(cm, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"📉 Confusion matrix saved to {cm_path}")


def run_experiment(n_heads, device, num_epochs=100, batch_size=64, log_dir=None):
    print(f"\n{'='*60}")
    print(f"🚀 Running Experiment with n_heads = {n_heads}")
    print(f"{'='*60}\n")

    preprocessor = Preprocessor(batch_size=batch_size)
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders()

    model = TwinGDLTransformer(
        input_dim=52,
        seq_len=500,
        d_model=64,
        n_heads=n_heads,
        n_classes=21,
        n_layers=3
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    save_dir = f"outputs/checkpoints/heads_{n_heads}"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    print(f"📚 Training model with {n_heads} heads for {num_epochs} epochs...\n")
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=num_epochs,
        save_path=best_model_path,
        log_dir=log_dir,
    )

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    metrics = evaluate_model(model, test_loader, device)

    print("\n📊 Test Results:")
    print(f"Overall Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Overall Precision:  {metrics['precision']:.4f}")
    print(f"Overall Recall:     {metrics['recall']:.4f}")
    print(f"Overall F1 Score:   {metrics['f1']:.4f}")

    if metrics.get("per_class"):
        print("\n📌 Classwise Metrics:")
        for idx in range(len(metrics["per_class"]["precision"])):
            acc = metrics["per_class"]["accuracy"][idx]
            prec = metrics["per_class"]["precision"][idx]
            rec = metrics["per_class"]["recall"][idx]
            f1 = metrics["per_class"]["f1"][idx]
            print(f"Class {idx:02d} | Acc {acc:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | F1 {f1:.4f}")

    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Using device: {device}\n")

    download_and_prepare_dataset()

    os.makedirs("outputs/results", exist_ok=True)

    n_heads = 4
    log_dir = os.path.join("outputs", "logs", f"heads_{n_heads}")
    metrics = run_experiment(n_heads, device, num_epochs=100, batch_size=64, log_dir=log_dir)
    per_class = metrics.get("per_class", {})

    result_row = {
        "n_heads": n_heads,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }
    for idx, value in enumerate(per_class.get("accuracy", [])):
        result_row[f"accuracy_class_{idx:02d}"] = value
    for idx, value in enumerate(per_class.get("precision", [])):
        result_row[f"precision_class_{idx:02d}"] = value
    for idx, value in enumerate(per_class.get("recall", [])):
        result_row[f"recall_class_{idx:02d}"] = value
    for idx, value in enumerate(per_class.get("f1", [])):
        result_row[f"f1_class_{idx:02d}"] = value

    results = [result_row]
    results_df = pd.DataFrame(results)

    results_csv = os.path.join("outputs", "results", "heads_comparison.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Results saved to {results_csv}")

    save_classwise_metrics(per_class, os.path.join("outputs", "results"))
    save_confusion_matrix(metrics["confusion_matrix"], os.path.join("outputs", "results"))

    plt.figure(figsize=(10, 6))
    plt.plot(results_df["n_heads"], results_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(results_df["n_heads"], results_df["f1"], marker="o", label="F1 Score")
    plt.plot(results_df["n_heads"], results_df["precision"], marker="o", label="Precision")
    plt.plot(results_df["n_heads"], results_df["recall"], marker="o", label="Recall")
    plt.xlabel("Number of Heads")
    plt.ylabel("Score")
    plt.title("Transformer Performance vs. Number of Heads")
    plt.legend()
    plt.grid(True)
    comparison_plot = os.path.join("outputs", "results", "heads_comparison_plot.png")
    plt.tight_layout()
    plt.savefig(comparison_plot)
    plt.close()
    print(f"📈 Comparison plot saved at {comparison_plot}")


if __name__ == "__main__":
    main()
