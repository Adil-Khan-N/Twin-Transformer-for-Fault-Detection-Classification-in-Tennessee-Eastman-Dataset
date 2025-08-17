import torch
import torch.nn as nn
import torch.optim as optim

from utils.preprocess import Preprocessor
from models.TwinTransformer import TwinGDLTransformer
from utils.train import train_model
from utils.evaluate import evaluate_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Using device: {device}\n")

    # -------------------- Load Data --------------------
    print("📥 Loading data...")
    preprocessor = Preprocessor(data_dir="dataset", batch_size=64)
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders()
    print("✅ Data loaded.\n")

    # -------------------- Define Model --------------------
    print("🛠 Initializing model...")
    model = TwinGDLTransformer(
        input_dim=52,
        seq_len=500,
        d_model=64,
        n_heads=8,
        n_classes=21,
        n_layers=3
    )
    model.to(device)

    # -------------------- Training Setup --------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_model_path = "outputs/checkpoints/best_twin_gdl.pth"

    # -------------------- Train --------------------
    print("\n🚀 Starting Training...\n")
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=2,
        save_path=best_model_path
    )
    print("\n✅ Training finished. Best model saved at:", best_model_path)

    # -------------------- Evaluation --------------------
    print("\n🔍 Evaluating best model on Test Set...\n")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    metrics = evaluate_model(model, test_loader, device)

    # -------------------- Results --------------------
    print("\n📊 Final Test Results:")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1 Score:   {metrics['f1']:.4f}")
    print("\nConfusion Matrix:\n", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
