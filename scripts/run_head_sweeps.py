import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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


def run_experiment(
    n_heads,
    device,
    num_epochs=100,
    batch_size=64,
    data_dir="datasets",
    output_dir="outputs",
    log_dir=None,
    resume=False,
):
    print(f"\n{'='*60}")
    print(f"🚀 Running Experiment with n_heads = {n_heads}")
    print(f"{'='*60}\n")

    preprocessor = Preprocessor(data_dir=data_dir, batch_size=batch_size)
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders()

    model = TwinGDLTransformer(
        input_dim=52,
        seq_len=500,
        d_model=64,
        n_heads=n_heads,
        n_classes=21,
        n_layers=3,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    save_dir = os.path.join(output_dir, f"checkpoints/heads_{n_heads}")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    if resume and os.path.exists(checkpoint_path):
        print(f"♻️ Resuming training from checkpoint: {checkpoint_path}")

    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=num_epochs,
        save_path=best_model_path,
        checkpoint_path=checkpoint_path,
        resume=resume,
        log_dir=log_dir,
    )

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found after training: {best_model_path}")

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


def save_experiment_results(metrics, output_dir, n_heads):
    os.makedirs(output_dir, exist_ok=True)
    result_row = {
        "n_heads": n_heads,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }
    per_class = metrics.get("per_class", {})

    for idx, value in enumerate(per_class.get("accuracy", [])):
        result_row[f"accuracy_class_{idx:02d}"] = value
    for idx, value in enumerate(per_class.get("precision", [])):
        result_row[f"precision_class_{idx:02d}"] = value
    for idx, value in enumerate(per_class.get("recall", [])):
        result_row[f"recall_class_{idx:02d}"] = value
    for idx, value in enumerate(per_class.get("f1", [])):
        result_row[f"f1_class_{idx:02d}"] = value

    results_df = pd.DataFrame([result_row])
    results_csv = os.path.join(output_dir, "heads_comparison.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Results saved to {results_csv}")

    save_classwise_metrics(per_class, output_dir)
    save_confusion_matrix(metrics["confusion_matrix"], output_dir)

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
    comparison_plot = os.path.join(output_dir, "heads_comparison_plot.png")
    plt.tight_layout()
    plt.savefig(comparison_plot)
    plt.close()
    print(f"📈 Comparison plot saved at {comparison_plot}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run head sweep experiments with resume support.")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of transformer heads")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="datasets", help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint if available")
    parser.add_argument("--no-download", action="store_true", help="Skip dataset download/check preparation")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Using device: {device}\n")

    if not args.no_download:
        download_and_prepare_dataset()

    log_dir = os.path.join(args.output_dir, "logs", f"heads_{args.n_heads}")
    metrics = run_experiment(
        n_heads=args.n_heads,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        log_dir=log_dir,
        resume=args.resume,
    )

    save_experiment_results(metrics, os.path.join(args.output_dir, "results"), args.n_heads)


if __name__ == "__main__":
    main()
