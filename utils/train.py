import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils.evaluate import evaluate_model


def _log_confusion_matrix(writer, cm, epoch):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_title("Validation Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.grid(False)
    plt.tight_layout()
    writer.add_figure("val/confusion_matrix", fig, epoch)
    plt.close(fig)


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=20,
    save_path="outputs/checkpoints/best_model.pth",
    checkpoint_path=None,
    resume=False,
    log_dir=None,
):
    print(f"✅ Training setup complete")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print("-" * 50)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(save_path), "checkpoint.pth")

    best_val_acc = 0.0
    start_epoch = 0

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, device)
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"♻️ Resuming from checkpoint epoch {start_epoch} / {num_epochs} (best_val_acc={best_val_acc:.4f})")

    writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    if start_epoch >= num_epochs:
        print("⚠️ Start epoch is already beyond or equal to target num_epochs. No training will be performed.")
        if writer:
            writer.close()
        return save_path

    for epoch in range(start_epoch, num_epochs):
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
                for idx, value in enumerate(per_class.get("accuracy", [])):
                    writer.add_scalar(f"val/accuracy_class_{idx}", float(value), epoch + 1)
                for idx, value in enumerate(per_class.get("precision", [])):
                    writer.add_scalar(f"val/precision_class_{idx}", float(value), epoch + 1)
                for idx, value in enumerate(per_class.get("recall", [])):
                    writer.add_scalar(f"val/recall_class_{idx}", float(value), epoch + 1)
                for idx, value in enumerate(per_class.get("f1", [])):
                    writer.add_scalar(f"val/f1_class_{idx}", float(value), epoch + 1)

                _log_confusion_matrix(writer, val_metrics["confusion_matrix"], epoch + 1)
        except Exception as e:
            print(f"❌ Error during validation: {str(e)}")
            raise e

        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved new best model (val_acc={best_val_acc:.4f})")

        # Save checkpoint every epoch so resume is possible
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"💾 Saved checkpoint for epoch {epoch+1} at {checkpoint_path}")

    if writer:
        writer.flush()
        writer.close()

    print("\n🎉 Training finished successfully")
    return save_path
