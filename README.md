# ARES: Automated Reasoning for Error Spotting

ARES detects erroneous cells in tabular data using three kinds of evidence: features computed from the target table, predictions transferred from classifiers trained on labeled historical tables, and labels and validation criteria produced by a language model. The target table does not need manually assigned clean/error labels for training. A clean copy is required by the supplied loader to construct historical labels and to evaluate a target run.

This README describes the code supplied with the project. The current scripts contain machine-specific paths and a few integration issues listed under **Before running**; adjust those before treating a new checkout as runnable.

## What the pipeline does

For each target column, `pipeline.py` computes value co-occurrence, pattern-frequency, and FastText features. It trains MLPs on eligible historical columns, groups target and historical columns by structural profiles, and summarizes matched models' predicted error probabilities by their mean, maximum, minimum, and standard deviation. An LLM then generates distribution analysis, guidelines, criteria functions, and labels for cells selected near K-Means centroids. High- and Medium-confidence labels are propagated across their clusters; if all parsed labels are Low-confidence, those labels are retained. An attribute-specific Random Forest combines the feature groups and propagated labels. When no propagated labels are available, the pipeline attempts historical pseudo-labels, then criteria-based pseudo-labels, then Isolation Forest.

The current `pipeline.py` also calls `generate_errors()` for each column and saves its output under `err_gen/`. This call is separate from representative labeling, but it still contacts the LLM and contributes to a run's prompt and completion usage.

## Files

| File | Role |
| --- | --- |
| `pipeline.py` | Main detection and evaluation pipeline |
| `run_config.yaml` | Target data, historical data, FastText, and ablation settings |
| `config.py`, `utility.py` | Default settings, CSV loading, logging, and LLM client |
| `zeroed_features.py`, `zeroed_llm.py` | Target features, criteria execution, prompts, and response parsing |
| `saged_meta.py`, `saged_profiler.py`, `saged_features.py` | Historical classifiers, profile matching, and supporting features |
| `llm_logging.py` | LLM request and approximate token logging |
| `run_incremental_historical.py`, `run_incremental_historical.sh` | Incremental historical-pool experiment |
| `universal-error-generator.py` | Separate data-error generation utility; not imported by the main pipeline |

Python imports use the unsuffixed names above. If your download contains files such as `pipeline(1).py`, rename or copy them to `pipeline.py`, and do the same for the other `(1)` files before running the project.

## Data layout

Put aligned dirty and clean CSVs under `data/` next to `config.py`. The loader recognizes either of these layouts for a dataset called `flights`:

```text
data/flights_clean.csv
data/flights_error-01.csv
```

or

```text
data/flights/clean.csv
data/flights/dirty.csv
```

The loader reads cells as strings and assigns the clean file's column names to the dirty file. Keep row order and column order aligned: evaluation identifies an erroneous cell by comparing the two copies at the same position. Historical dirty/clean pairs supply training labels; the target clean file is used for evaluation.

## Environment and configuration

The supplied imports require Python packages including `numpy`, `pandas`, `scikit-learn`, `PyYAML`, `openai`, `fasttext`, `gensim`, and `scipy`. Install versions compatible with your Python environment. A FastText `.bin` file is needed for semantic embeddings; set its location in `run_config.yaml`.

The default LLM client in `utility.py` sends OpenAI-compatible chat requests to `http://localhost:8000/v1` with `api_use=False`. Start a compatible inference server and configure the model identifier accepted by that server. The supplied identifier and FastText path are absolute paths from the original cluster and must be changed for another machine. The current LLM client selects its model in `utility.py`, not through `run_config.yaml`.

Key settings in `run_config.yaml` are:

```yaml
target_dataset: flights
historical_datasets: [billionaire, beers, rayyan, hospital]
fasttext_model_path: /path/to/cc.en.300.bin
fasttext_dim: 50
top_k_related: 5
n_clusters: 5
use_history: true
use_confidence: true
```

`use_history` and `use_confidence` select the full method or its ablations. `n_clusters` sets the representative-cell clustering count in the main pipeline and is also used by historical profile matching. When `historical_datasets` is empty or absent, `pipeline.py` scans `data/` for other datasets with top-level `*_clean.csv` files; an empty list does **not** explicitly mean “no history” in this script. To disable historical features, set `use_history: false`.

## Before running

The uploaded scripts have the following issues that need checking in the intended environment:

1. `pipeline.py` calls `generate_meta_features_with_models(..., dirty_feats=zeroed_feats)`, but the supplied `saged_meta.py` function accepts `dirty_features`. As supplied, the historical branch raises a keyword-argument error. Align the keyword and parameter name.
2. Paths are not consistently relative to the script: `config.py` locates `data/` beside itself, while `pipeline.py` writes under `ares-main/results/` and first looks for `ares-main/run_config.yaml`. Run from the directory expected by your deployment or make these paths consistent.
3. The incremental runner rewrites `run_config.yaml` with only target and history names. It also searches for JSON/TXT results, whereas the main pipeline writes `metrics.csv`. Its current summary therefore does not reliably collect F1, and an empty historical list triggers the main pipeline's automatic dataset scan. The shell wrapper hard-codes a conda environment and cluster directory.
4. The LLM model identifier and local server URL in `utility.py` are hard-coded. Confirm that the server exposes the identifier used by the client.

These are setup and integration notes, not claims that the published experiments should be changed. Keep the original run configuration, logs, and model identifiers with any reported results.

## Run and outputs

After resolving the points above and preparing the model server, data, and FastText file, run from the configured project directory:

```bash
python pipeline.py
```

The main script creates a timestamped directory under `ares-main/results/` relative to its working directory. It writes `run.log`, `metrics.csv` (dataset, variant, F1, precision, recall, and selected settings), `llm_logging.txt`, and per-column prompt/output files in `distri_analys/`, `guide/`, `funcs/`, `llm_label/`, and `err_gen/`. It does not currently write a CSV of the final cell-level predictions. LLM token counts in the logger are approximate.

`run_incremental_historical.py` is intended to rerun the target with progressively expanded historical pools and write a summary under `incremental_results/`. Use it only after correcting the path, no-history, and result-collection behavior described above.

## Reproducing the paper

Record the exact dirty/clean dataset versions, history pool and its order, `run_config.yaml`, FastText model, LLM model and server settings, and output logs for each run. The paper's tables cover multiple LLMs and ablations; the supplied configuration and hard-coded client do not automatically reproduce every table. The code was inspected for this README, but a full training and LLM run was not executed here.
