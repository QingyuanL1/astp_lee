# Run Commands (NAS → Optuna → Batch Test)

This file summarizes the full set of commands we used to run the workflow in this repo:

- Architecture search (LLM / offline)
- Post-architecture Optuna combo search
- Batch evaluation on ATSP sizes 100/150/250

> Notes
>
>- Commands assume you run from the repo root: `atsp_gnn/`
>- Use `DGL_NO_GRAPHBOLT=1` if your DGL wheel does not bundle GraphBolt.
>- **Never paste real API keys into git-tracked files.** Use environment variables or a local `.env` that is gitignored.

---

## 0) Environment

Activate venv:

```bash
source .venv/bin/activate
```

Install Optuna (needed for `search_all_combo.py`):

```bash
pip install optuna
```

---

## 1) Big Search (NAS → auto Optuna)

### 1.1 Offline NAS + auto Optuna (no network)

```bash
DGL_NO_GRAPHBOLT=1 python -m src.engine.run \
  --framework dgl --mode arch \
  --device cuda \
  --data_dir saved_dataset/ATSP_2560x50 \
  --atsp_size 50 \
  --relation_types ss st tt pp \
  --model HGNASModel \
  --input_dim 1 --hidden_dim 64 --output_dim 1 \
  --num_gnn_layers 3 --num_heads 8 \
  --n_epochs 3 \
  --nas_iterations 2 --nas_batch_size 3 \
  --nas_metric gap \
  --llm_offline --llm_seed 1 \
  --combo_after_arch \
  --combo_n_trials 5
```

### 1.2 Online NAS + auto Optuna (OpenAI-compatible API)

Set your key in the shell (recommended):

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

Then run:

```bash
DGL_NO_GRAPHBOLT=1 python -m src.engine.run \
  --framework dgl --mode arch \
  --device cuda \
  --data_dir saved_dataset/ATSP_2560x50 \
  --atsp_size 50 \
  --relation_types ss st tt pp \
  --model HGNASModel \
  --input_dim 1 --hidden_dim 64 --output_dim 1 \
  --num_gnn_layers 3 --num_heads 8 \
  --n_epochs 3 \
  --nas_iterations 2 --nas_batch_size 3 \
  --nas_metric gap \
  --llm_api_key "$OPENAI_API_KEY" \
  --llm_base_url https://aihubmix.com/v1 \
  --combo_after_arch \
  --combo_n_trials 5
```

If you use OpenAI official endpoint, omit `--llm_base_url`.

---

## 2) Run only the post-arch Optuna combo search (reuse an existing architecture)

If you already have a best architecture JSON from NAS (e.g. under `arch_search_runs/.../architecture.json`), you can run only Optuna:

```bash
DGL_NO_GRAPHBOLT=1 python -m src.engine.search_all_combo \
  --framework dgl \
  --device cuda \
  --model HGNASModel \
  --architecture_path /ABS/PATH/TO/arch_search_runs/<run>/iterXX_candYY/architecture.json \
  --relation_types ss st tt pp \
  --relation_subsets ss,st,tt,pp \
  --n_trials 5
```

---

## 2.1) (Recommended) Optuna combo search on ATSP100 (pp_ss_st_tt_sum) + evaluate on ATSP100

This is the exact workflow we just ran to make Optuna train/val on size 100 and then evaluate on ATSP100.

### 2.1.1 Prepare a small Optuna dataset dir from `ATSP_30x100`

Create `saved_dataset/ATSP_30x100_optuna` with non-empty `train.txt/val.txt/test.txt`:

```bash
python - <<'PY'
from pathlib import Path
import shutil

src = Path("saved_dataset/ATSP_30x100")
dst = Path("saved_dataset/ATSP_30x100_optuna")

if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True, exist_ok=True)

pkls = sorted([p for p in src.glob("*.pkl") if p.name != "scalers.pkl"])
assert len(pkls) >= 30, f"Need >=30 pkls, got {len(pkls)}"

for p in pkls[:30]:
    shutil.copy2(p, dst / p.name)

shutil.copy2(src / "scalers.pkl", dst / "scalers.pkl")
shutil.copytree(src / "templates", dst / "templates")

names = [p.name for p in pkls[:30]]
train = names[:24]
val = names[24:27]
test = names[27:30]

(dst / "train.txt").write_text("\n".join(train) + "\n")
(dst / "val.txt").write_text("\n".join(val) + "\n")
(dst / "test.txt").write_text("\n".join(test) + "\n")

print("[OK] created:", dst)
print("train/val/test =", len(train), len(val), len(test))
PY
```

### 2.1.2 Run Optuna on ATSP100 (train/val on size 100)

```bash
DGL_NO_GRAPHBOLT=1 python -m src.engine.search_all_combo \
  --framework dgl \
  --device cuda \
  --model HGNASModel \
  --agg sum \
  --relation_types pp ss st tt \
  --relation_subsets pp,ss,st,tt \
  --atsp_size 100 \
  --data_dir saved_dataset/ATSP_30x100_optuna \
  --n_trials 20
```

Outputs:

- Best checkpoint: `search/no_slurm_job/best_model_rel_pp_ss_st_tt_sum.pt`
- Optuna best params: `search/no_slurm_job/result_rel_pp_ss_st_tt_sum.json`

### 2.1.3 Backup summary CSV before re-running batch tests

Batch tests overwrite these CSVs, so back them up if you want to keep old results:

```bash
cp search/batch_test_summary.csv search/batch_test_summary.atsp50.csv
cp search/no_slurm_job/summary.csv search/no_slurm_job/summary.atsp50.csv
```

### 2.1.4 Evaluate on ATSP100

```bash
DGL_NO_GRAPHBOLT=1 python -m src.engine.batch_test_search_models \
  --search_root search \
  --sizes 100 \
  --override_sizes 100 \
  --device cuda \
  --time_limit 0.1666666667 \
  --perturbation_moves 30
```

Outputs:

- Global summary: `search/batch_test_summary.csv`
- Per-dir summary: `search/no_slurm_job/summary.csv`

### 2.1.5 If you need to change settings

- **Change ATSP size**:
  - Set `--atsp_size <N>`
  - Ensure `--data_dir` points to a dataset dir with non-empty `train.txt/val.txt` and matching `templates/`.
- **Change relation subset**:
  - Example: `--relation_subsets ss,tt` (must be a subset of `--relation_types`).
- **Change trials**:
  - Set `--n_trials <K>`.

---

## 2.2) Generate a larger ATSP100 training dataset (2560 instances) with LKH

If you want Optuna to generalize, use a larger dataset than `ATSP_30x100`.

### 2.2.1 Generate raw instances + labels

This repo’s generator names the folder as `ATSP_{n_nodes}x{n_samples}`.
So `2560 100 saved_dataset` creates: `saved_dataset/ATSP_100x2560/`.

Test a small run first:

```bash
DGL_NO_GRAPHBOLT=1 python -m src.data.dataset_generator \
  10 100 saved_dataset \
  --lkh_path /root/LKH-3.0.9/LKH \
  --regret_mode fixed_edge_lkh
```

Then the full dataset (can be very slow):

```bash
DGL_NO_GRAPHBOLT=1 python -m src.data.dataset_generator \
  2560 100 saved_dataset \
  --lkh_path /root/LKH-3.0.9/LKH \
  --regret_mode fixed_edge_lkh \
  --parallel \
  --processes 8
```

### 2.2.2 Preprocess (split + scalers + templates)

```bash
DGL_NO_GRAPHBOLT=1 python -m src.data.preprocessor \
  saved_dataset/ATSP_100x2560 \
  --n_train 2400 \
  --n_val 80 \
  --n_test 80 \
  --atsp_size 100 \
  --relation_types pp ss st tt
```

Then run Optuna with:

```bash
--data_dir saved_dataset/ATSP_100x2560
```

---

## 3) Batch evaluation on larger sizes (100/150/250)

Run batch testing over the checkpoints under `search/`:

```bash
DGL_NO_GRAPHBOLT=1 python src/engine/batch_test_search_models.py \
  --search_root /root/autodl-tmp/web/atsp_gnn/search \
  --only_dirs no_slurm_job \
  --sizes 100 150 250 \
  --device cuda
```

Outputs:

- Global summary: `search/batch_test_summary.csv`
- Per-dir summary: `search/no_slurm_job/summary.csv`

---

## 4) Common variants

### 4.1 Force re-evaluation for specific sizes (override)

```bash
DGL_NO_GRAPHBOLT=1 python src/engine/batch_test_search_models.py \
  --search_root /root/autodl-tmp/web/atsp_gnn/search \
  --only_dirs no_slurm_job \
  --sizes 100 150 250 \
  --override_sizes 100 150 250 \
  --device cuda
```

### 4.2 Reuse saved predictions (skip model forward, rerun GLS only)

Requires existing per-instance files (with `regret_pred:`) for those sizes:

```bash
DGL_NO_GRAPHBOLT=1 python src/engine/batch_test_search_models.py \
  --search_root /root/autodl-tmp/web/atsp_gnn/search \
  --only_dirs no_slurm_job \
  --sizes 100 150 250 \
  --override_sizes 100 150 250 \
  --reuse_predictions \
  --device cuda
```

### 4.3 Only evaluate one subfolder under `search/`

```bash
DGL_NO_GRAPHBOLT=1 python src/engine/batch_test_search_models.py \
  --search_root /root/autodl-tmp/web/atsp_gnn/search \
  --only_dirs <DIR_NAME> \
  --device cuda
```

---

## 5) Where results are written

- NAS runs: `arch_search_runs/<timestamp>_atsp<size>/...`
- Combo search outputs: `search/<SLURM_JOB_ID or no_slurm_job>/...`
- Batch test outputs (per checkpoint):
  - `search/<dir>/<ckpt_stem>/test_atsp{size}/results.json`
  - `search/<dir>/<ckpt_stem>/trial_0/test_atsp{size}/instance*.txt`
