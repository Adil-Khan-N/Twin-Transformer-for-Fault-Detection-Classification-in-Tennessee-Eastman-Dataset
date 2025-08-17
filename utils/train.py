import torch
import torch.nn as nn
import os
from utils.evaluate import evaluate_model

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20, save_path="outputs/checkpoints/best_model.pth"):
    print(f"✅ Training setup complete")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print("-" * 50)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\n🚀 Starting Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            try:
                print(f"  🔹 Batch {batch_idx+1}/{len(train_loader)}")
                print(f"     X shape: {X_batch.shape}, y shape: {y_batch.shape}")

                # Move data to device
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Forward
                optimizer.zero_grad()
                outputs = model(X_batch)
                print(f"     Outputs shape: {outputs.shape}")

                # Loss
                loss = criterion(outputs, y_batch.long())
                print(f"     Loss: {loss.item():.4f}")

                # Backward + Optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            except Exception as e:
                print(f"❌ Error in batch {batch_idx+1}: {str(e)}")
                raise e

        avg_loss = running_loss / len(train_loader)
        print(f"\n📊 Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # Validation
        try:
            val_metrics = evaluate_model(model, val_loader, device)
            val_acc = val_metrics["accuracy"]
            print(f"✅ Validation done - Acc: {val_acc:.4f}, F1: {val_metrics['f1']:.4f}")
        except Exception as e:
            print(f"❌ Error during validation: {str(e)}")
            raise e

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved new best model (val_acc={best_val_acc:.4f})")

    print("\n🎉 Training finished successfully")
    return save_path
