import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from utils.evaluate import evaluate_model

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=20,
    save_path="outputs/checkpoints/best_model.pth",
    log_dir=None,
):
    print(f"✅ Training setup complete")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print("-" * 50)

    best_val_acc = 0.0
    writer = SummaryWriter(log_dir=log_dir) if log_dir else None

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

        if writer:
            writer.add_scalar("train/loss", avg_loss, epoch + 1)

        # Validation
        try:
            val_metrics = evaluate_model(model, val_loader, device)
            val_acc = val_metrics["accuracy"]
            print(f"✅ Validation done - Acc: {val_acc:.4f}, F1: {val_metrics['f1']:.4f}")

            if writer:
                writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch + 1)
                writer.add_scalar("val/precision", val_metrics["precision"], epoch + 1)
                writer.add_scalar("val/recall", val_metrics["recall"], epoch + 1)
                writer.add_scalar("val/f1", val_metrics["f1"], epoch + 1)

                per_class = val_metrics.get("per_class", {})
                for idx, value in enumerate(per_class.get("precision", [])):
                    writer.add_scalar(f"val/precision_class_{idx}", value, epoch + 1)
                for idx, value in enumerate(per_class.get("recall", [])):
                    writer.add_scalar(f"val/recall_class_{idx}", value, epoch + 1)
                for idx, value in enumerate(per_class.get("f1", [])):
                    writer.add_scalar(f"val/f1_class_{idx}", value, epoch + 1)
        except Exception as e:
            print(f"❌ Error during validation: {str(e)}")
            raise e

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved new best model (val_acc={best_val_acc:.4f})")

    if writer:
        writer.flush()
        writer.close()

    print("\n🎉 Training finished successfully")
    return save_path
