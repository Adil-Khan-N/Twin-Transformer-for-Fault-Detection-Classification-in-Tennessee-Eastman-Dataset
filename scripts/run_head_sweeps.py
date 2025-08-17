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


def run_experiment(n_heads, device, num_epochs=5, batch_size=64):
    print(f"\n{'='*60}")
    print(f"🚀 Running Experiment with n_heads = {n_heads}")
    print(f"{'='*60}\n")

    # -------------------- Load Data --------------------
    preprocessor = Preprocessor(batch_size=batch_size)
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders()
    # -------------------- Define Model --------------------
    model = TwinGDLTransformer(
        input_dim=52,
        seq_len=500,
        d_model=64,
        n_heads=n_heads,
        n_classes=21,
        n_layers=3
    )
    model.to(device)

    # -------------------- Training Setup --------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Create save directory per head
    save_dir = f"outputs/checkpoints/heads_{n_heads}"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    # -------------------- Train --------------------
    print(f"📚 Training model with {n_heads} heads...\n")
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=num_epochs,
        save_path=best_model_path
    )

    # -------------------- Evaluate --------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    metrics = evaluate_model(model, test_loader, device)
    print("\n📊 Test Results:")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1 Score:   {metrics['f1']:.4f}")

    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Using device: {device}\n")
    
    download_and_prepare_dataset()

    results = []

    # Run experiments for heads = 4 to 10
    for n_heads in range(4, 11):
        metrics = run_experiment(n_heads, device, num_epochs=5, batch_size=64)
        results.append({
            "n_heads": n_heads,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs("outputs/results", exist_ok=True)
    results_csv = "outputs/results/heads_comparison.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Results saved to {results_csv}")

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["n_heads"], results_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(results_df["n_heads"], results_df["f1"], marker="o", label="F1 Score")
    plt.xlabel("Number of Heads")
    plt.ylabel("Score")
    plt.title("Transformer Performance vs. Number of Heads")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/results/heads_comparison_plot.png")
    print("📈 Comparison plot saved at outputs/results/heads_comparison_plot.png")


if __name__ == "__main__":
    main()
