import torch
import torch.nn as nn
import torch.optim as optim

from utils.preprocess import Preprocessor
from models.TwinTransformer import TwinGDLTransformer
from utils.train import train_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    preprocessor = Preprocessor(data_dir="/datasets", batch_size=64)
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders()
    
    # Model
    model = TwinGDLTransformer(input_dim=52, seq_len=500, d_model=64, n_heads=8, n_classes=21, n_layers=3)
    model.to(device)
    
    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train
    best_model_path = "outputs/checkpoints/best_twin_gdl.pth"
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20, save_path=best_model_path)

if __name__ == "__main__":
    main()
