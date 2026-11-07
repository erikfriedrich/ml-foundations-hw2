# Machine Learning 1 — Homework 2

Solutions to Homework 2 of Data 37711 (Machine Learning & AI, Part I), Fall 2025, by Erik Friedrich.

- **Problem 1:** an MLP that predicts the conditional mean and standard deviation of NYC taxi trip durations, and the probability that a trip takes less than 45 minutes.
- **Problem 2:** target encoding vs. embeddings for two high-cardinality categorical features (retail transactions).
- **Problem 3:** Random Kitchen Sinks with ridge regression vs. a one-hidden-layer MLP on a 2D toy dataset.

## Files

| File | Contents |
|---|---|
| `hw2_Friedrich_Erik.ipynb` | The full solution: code, outputs and answers |
| `hw2_Friedrich_Erik.pdf` | The same notebook as a PDF |
| `models.py` | Random Kitchen Sinks and MLP models and their hyperparameter sweeps (Problem 3) |
| `dataloader.py`, `plotting.py` | Helper code provided with the assignment (Problem 3) |
| `taxi_trips.parquet` | Data for Problem 1 |
| `retail.csv` | Data for Problem 2 |
| `train.csv`, `val.csv`, `test.csv` | Data for Problem 3 (`test.csv` is unlabeled) |
| `rks_test_predictions.csv`, `mlp_test_predictions.csv` | Problem 3 predictions on `test.csv`, one per row, in the same order |
| `pdf_template/` | LaTeX template for exporting the notebook to PDF |

## Running it

Requires Python 3.10. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run the whole notebook and save the outputs in place
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 hw2_Friedrich_Erik.ipynb
```

All paths are relative to the repository root, so the command must be run from there. A run takes about 5–6 minutes on a laptop CPU. No GPU is needed or used.

Running the notebook also rewrites the two prediction CSVs and three plots (`train_surface.png`, `rks_surface.png`, `mlp_surface.png`).

**Reproducibility:** all random seeds are fixed, and all models run on the CPU with a single thread (`torch.set_num_threads(1)` and `threadpool_limits(1)` in the setup cell). GPUs, and CPUs with a different number of threads, round floating-point sums slightly differently, and for the neural networks these small differences grow during training into visibly different results. With this setup, a run reproduces the committed outputs exactly, independent of whether the machine has a GPU and how many cores it has. This was tested on an Apple Silicon Mac, both in a fresh environment installed from `requirements.txt` and with different thread settings. A different CPU architecture (e.g. x86 instead of ARM) may still round slightly differently. There, the neural-network results in Problems 1–3 could still differ somewhat from the committed outputs; everything else is unaffected.

## Exporting to PDF

Requires a LaTeX installation with `xelatex`:

```bash
jupyter nbconvert --to pdf --TemplateExporter.extra_template_basedirs=. --template pdf_template hw2_Friedrich_Erik.ipynb
```
