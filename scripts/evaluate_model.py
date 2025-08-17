import torch
from utils.preprocess import Preprocessor
from models.TwinTransformer import TwinGDLTransformer
from utils.evaluate import evaluate_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    preprocessor = Preprocessor(data_dir="dataset", batch_size=64)
    _, _, test_loader = preprocessor.get_dataloaders()
    
    # Load model
    model = TwinGDLTransformer(input_dim=52, seq_len=500, d_model=64, n_heads=8, n_classes=21, n_layers=3)
    model.load_state_dict(torch.load("outputs/checkpoints/best_twin_gdl.pth"))
    model.to(device)
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device)
    print("\n📊 Test Results:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("Confusion Matrix:\n", metrics["confusion_matrix"])

if __name__ == "__main__":
    main()
