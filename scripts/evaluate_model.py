import os
import sys
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.preprocess import Preprocessor
from models.TwinTransformer import TwinGDLTransformer
from utils.evaluate import evaluate_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    preprocessor = Preprocessor(data_dir="datasets", batch_size=64)
    _, _, test_loader = preprocessor.get_dataloaders()
    
    # Load model
    model = TwinGDLTransformer(input_dim=52, seq_len=500, d_model=64, n_heads=8, n_classes=21, n_layers=3)
    model.load_state_dict(torch.load("outputs/checkpoints/best_twin_gdl.pth", map_location=device))
    model.to(device)
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device)
    print("\n📊 Test Results:")
    print(f"Overall Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Overall Precision: {metrics['precision']:.4f}")
    print(f"Overall Recall:    {metrics['recall']:.4f}")
    print(f"Overall F1 Score:  {metrics['f1']:.4f}")
    print("Confusion Matrix:\n", metrics["confusion_matrix"])

    if metrics.get("per_class"):
        print("\n📌 Classwise Metrics:")
        for idx in range(len(metrics["per_class"]["precision"])):
            acc = metrics["per_class"]["accuracy"][idx]
            prec = metrics["per_class"]["precision"][idx]
            rec = metrics["per_class"]["recall"][idx]
            f1 = metrics["per_class"]["f1"][idx]
            print(f"Class {idx:02d} | Acc {acc:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | F1 {f1:.4f}")

if __name__ == "__main__":
    main()
