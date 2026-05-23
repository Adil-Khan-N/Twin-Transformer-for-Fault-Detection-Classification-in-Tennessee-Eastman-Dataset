# Twin Transformer for Fault Detection and Classification

## Setup

1. Clone the repository:

   ```bash
   git clone <repo_url>
   cd "Twin-Transformer-for-Fault-Detection-Classification-in-Tennessee-Eastman-Dataset"
   ```

2. Create and activate a Python virtual environment:

   ```bash
   python -m venv venv
   ```

   On Windows:

   ```bash
   venv\Scripts\activate
   ```

   On Linux / macOS:

   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
   ```

## Running the Experiment

The main training script will automatically download and prepare the Tennessee Eastman dataset, convert the RData files to CSV, train the model for 100 epochs, evaluate test performance, and save output metrics and plots.

Run from the repository root:

```bash
python scripts/run_head_sweeps.py
```

Or, if you prefer module style:

```bash
python -m scripts.run_head_sweeps
```

## Outputs

After a successful run, the following files are created:

- `outputs/checkpoints/heads_4/best_model.pth`
- `outputs/results/heads_comparison.csv`
- `outputs/results/heads_comparison_plot.png`
- `outputs/results/classwise_metrics.csv`
- `outputs/results/classwise_metrics_plot.png`
- `outputs/results/confusion_matrix.png`
- `outputs/logs/heads_4/` (TensorBoard logs)

## TensorBoard

Training metrics are logged to TensorBoard. Launch it with:

```bash
tensorboard --logdir outputs/logs
```

Then open the URL shown in the terminal, usually:

```bash
http://localhost:6006
```

If you want to view only the current run:

```bash
tensorboard --logdir_spec heads_4:outputs/logs/heads_4
```

## Standalone Evaluation

If you want to re-run evaluation only, use:

```bash
python scripts/evaluate_model.py
```

## Notes

- The dataset downloader supports `kagglehub` by default and falls back to the Kaggle CLI if needed.
- `pyreadr` is required for converting `.RData` files to CSV.

