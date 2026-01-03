# ATSP Heterogeneous GNN + GPT NAS

This repo extends the original [`walidgeuttala/atsp_gnn`](https://github.com/walidgeuttala/atsp_gnn) baseline with:

- **GPT-driven heterogeneous graph neural architecture search** (HGNAS) inspired by [LLM4HGNAS](https://github.com/cold-rivers/LLM4HGNAS) and [GNAS4QUBO](https://github.com/Embrasse-moi1/GNAS4QUBO).
- Automatic **architecture logging** (`arch_search_runs/`) and **evaluation metrics** (time / gap / cost).
- Visualization updates that ingest search summaries directly.

The README summarizes everything needed to reproduce data generation, training/testing, NAS, and visualization.

---

## 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Torch + CUDA 12.1 (adjust to your GPU/driver)
pip install torch==2.3.0+cu121 torchvision==0.18.0 torchaudio==2.3.0 \
  --extra-index-url https://download.pytorch.org/whl/cu121
# DGL (match torch version)
pip install dgl==2.5.0+cu121 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
# Core deps
pip install numpy networkx pandas tqdm tsplib95 optuna matplotlib PyYAML python-dotenv
```

> CPU-only: install `torch==2.3.0` & `dgl==2.5.0` from PyPI without the CUDA wheels.

### LKH-3 (required for regret labels / optional refinement)

```bash
cd /root
wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.9.tgz
tar xf LKH-3.0.9.tgz && cd LKH-3.0.9
make
# Path used later: /root/LKH-3.0.9/LKH
```

Ensure `--lkh_path` points to the compiled binary whenever you generate data.

---

## 2. Data Pipeline

All commands assume the repo root `atsp_gnn/` and a virtualenv is active.

### 2.1 Generate ATSP instances + regret labels

```bash
DGL_NO_GRAPHBOLT=1 python src/data/dataset_generator.py 100 20 data \
  --lkh_path /root/LKH-3.0.9/LKH \
  --regret_mode row_best      # or fixed_edge_lkh / no_regret
```

Outputs `data/ATSP_20x100/` with raw `.pkl` graphs and `summary.csv`.

### 2.2 Preprocess (splits, scalers, templates)

```bash
DGL_NO_GRAPHBOLT=1 python -m src.data.preprocessor data/ATSP_20x100 \
  --atsp_size 20 \
  --relation_types ss st tt pp \
  --n_train 80 --n_val 10 --n_test 10 \
  --save_pyg
```

Creates:
- `train.txt / val.txt / test.txt`
- `scalers.pkl`
- `templates/` for every relation subset (DGL + PyG)

> Missing `train.txt` errors during training mean this step hasn’t been run.

---

## 3. Training & Evaluation (DGL)

Unified entrypoint: `python -m src.engine.run`.

### 3.1 Train HetroGAT (GPU example)

```bash
DGL_NO_GRAPHBOLT=1 python -m src.engine.run \
  --framework dgl --mode train \
  --data_dir data/ATSP_20x100 \
  --atsp_size 20 \
  --relation_types ss st tt pp \
  --model HetroGAT --agg concat \
  --input_dim 1 --hidden_dim 64 --output_dim 1 \
  --num_gnn_layers 3 --num_heads 8 \
  --batch_size 16 --n_epochs 20 \
  --device cuda
```

Artifacts: `runs/<timestamp>_HetroGATconcat_ATSP20_Comboss_st_tt_pp/trial_0/{best_model.pt, results.json}`.

### 3.2 Test + Guided Local Search

```bash
DGL_NO_GRAPHBOLT=1 python -m src.engine.run \
  --framework dgl --mode test \
  --data_path data/ATSP_20x100 \
  --atsp_size 20 \
  --relation_types ss st tt pp \
  --model HetroGAT --agg concat \
  --model_path runs/<timestamp>_HetroGATconcat_ATSP20_Comboss_st_tt_pp/trial_0/best_model.pt \
  --device cuda \
  --time_limit 0.1 --perturbation_moves 20 --perturbation_count 5
```

Results saved under `runs/<timestamp>.../trial_0/test_atsp20/results_test_20.json`, including per-instance `opt_cost`, `final_cost`, `init/final gap`, and `model/gls time`.

---

## 4. GPT-driven Heterogeneous NAS

### 4.1 Model & Spec

- `src/models/models_dgl.py` exposes `HGNASModel`, which consumes JSON-based architectures (`ArchitectureSpec`).
- Every `arch_search_runs/<timestamp>_atsp<size>/iterXX_candYY/` now contains:
  - `architecture.json`: GPT proposal decoded to a structured spec.
  - `best_model.pt`: checkpoint with the matching weights + metadata (relation types, dims, etc.).
  - `result.json`: validation/test metrics recorded during search.

### 4.2 Running NAS

1. **Configure API key** in `.env` (ignored by Git):
   ```
   OPENAI_API_KEY=sk-...
   # optional: OPENAI_BASE_URL=https://aihubmix.com/v1
   ```
   Load via `source .env` or pass `--llm_api_key`.

2. **Launch search** (set `--llm_offline` to sample randomly when no key/network):

```bash
DGL_NO_GRAPHBOLT=1 python -m src.engine.run \
  --framework dgl --mode arch \
  --data_dir data/ATSP_20x100 \
  --atsp_size 20 \
  --relation_types ss st tt pp \
  --model HGNASModel \
  --input_dim 1 --hidden_dim 64 --output_dim 1 \
  --num_gnn_layers 3 --num_heads 8 \
  --n_epochs 3 \
  --nas_iterations 5 --nas_batch_size 4 \
  --nas_metric gap \
  --device cuda \
  --llm_api_key \"$OPENAI_API_KEY\"  # or rely on .env
```

3. **Metrics**: each `summary.json` reports:
   - `avg_final_gap`, `avg_final_cost`, `avg_total_time` (model + GLS)
   - best architecture vector & spec path

4. **Scoring criterion**: choose `--nas_metric gap|cost|time|val_loss`. Training still uses regret MSE; the NAS objective controls architecture ranking (requirement (4)).

### 4.3 Re-test saved architectures

Each candidate checkpoint can be reloaded later (no need to retrain) together with the frozen architecture JSON:

```bash
DGL_NO_GRAPHBOLT=1 python -m src.tools.eval_architecture \
  --checkpoint arch_search_runs/<run>/iter00_cand00/best_model.pt \
  --data_path data/ATSP_20x100 \
  --device cuda --time_limit 0.1 --perturbation_moves 20 --perturbation_count 5
```

The helper picks up the `architecture_path`, hidden dimensions, and relation types from the checkpoint metadata, rebuilds `HGNASModel`, and invokes the standard tester. Use `--output_dir ...` if you want to store the new `results_test_XX.json` somewhere other than `iterXX_candYY/test_atspXX/`.

---

## 5. Visualization

`src/visualization/plot_result.py` merges the hard-coded baselines and any NAS `summary.json` found under `arch_search_runs/`. Run:

```bash
python src/visualization/plot_result.py
```

Generates `result_plot_full.pdf` (execution time vs gap on log scale), with NAS entries labelled `LLM HGNAS (n=XX)`.

---

## 6. Repository Layout

```
arch_search_runs/      # architecture samples + summary
data/                  # generated ATSP datasets (ignored by Git)
runs/                  # training/test outputs
src/
  arch_search/         # ArchitectureSpec, search space, GPT runner
  data/                # dataset generator, preprocessor, DGL/PyG datasets
  engine/              # train/test/search entrypoints
  models/              # HetroGAT, HGNASModel, legacy models
  visualization/       # plot_result.py
```

---

## 7. Common Issues

| Symptom | Fix |
| --- | --- |
| `FileNotFoundError: data/.../train.txt` | Run both `dataset_generator.py` and `python -m src.data.preprocessor`. |
| `ModuleNotFoundError: torch_geometric` during preprocessing | Install `torch_geometric` or run preprocessor via `python -m src.data.preprocessor` (which resolves relative imports). |
| LKH errors / missing regret | Verify `--lkh_path`, consider `--regret_mode row_best` or `--no_regret` for quick tests. |
| NAS command fails with network/auth errors | Ensure correct `OPENAI_API_KEY` (or `AIHUBMIX_API_KEY`) and reachable `base_url`. Use `--llm_offline` offline. |
| Visualization missing new results | Confirm `arch_search_runs/<timestamp>/summary.json` exists before running `plot_result.py`. |

---

## 8. Quick Reference

| Task | Command |
| --- | --- |
| Generate data | `python src/data/dataset_generator.py ...` |
| Preprocess | `python -m src.data.preprocessor ...` |
| Train | `python -m src.engine.run --mode train ...` |
| Test | `python -m src.engine.run --mode test ...` |
| NAS | `python -m src.engine.run --mode arch ...` |
| Visualization | `python src/visualization/plot_result.py` |

This README captures the updated workflow and satisfies the project requirements: architecture search with GPT, structured logging, evaluation metrics (time/gap/cost), alternative NAS metrics, and visualization compatibility. Refer to `GPU_RUN.md` for a shorter command list when running on GPUs. Good luck! 💪
