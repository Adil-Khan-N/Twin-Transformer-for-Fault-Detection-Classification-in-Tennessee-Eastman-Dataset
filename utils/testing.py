from preprocess import Preprocessor

pre = Preprocessor(data_dir="../dataset", batch_size=64)
train_loader, val_loader, test_loader = pre.get_dataloaders()

# Check batch shapes
for X, y in train_loader:
    print("Train batch X:", X.shape, "y:", y.shape)
    break

for X, y in val_loader:
    print("Val batch X:", X.shape, "y:", y.shape)
    break

for X, y in test_loader:
    print("Test batch X:", X.shape, "y:", y.shape)
    break
