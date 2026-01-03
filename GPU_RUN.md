# GPU Training & Testing Notes

Environment assumptions:
- Python virtualenv already created (`.venv`), torch==2.3.0+cu121 installed
- DGL CUDA wheel: `pip install dgl==2.5.0+cu121 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html`
- Dataset prepared at `data/ATSP_20x100` (run generator + preprocessor beforehand)

## Generate ATSP Data + Regret Labels
```bash
source .venv/bin/activate
DGL_NO_GRAPHBOLT=1 python src/data/dataset_generator.py 100 20 data \
  --lkh_path /root/LKH-3.0.9/LKH \
  --regret_mode row_best    # or fixed_edge_lkh
```
Outputs `data/ATSP_20x100/` with raw `.pkl` instances and `summary.csv`.

## Preprocess (splits, scalers, templates)
```bash
source .venv/bin/activate
DGL_NO_GRAPHBOLT=1 python src/data/preprocessor.py data/ATSP_20x100 \
  --atsp_size 20 \
  --relation_types ss st tt pp \
  --n_train 80 --n_val 10 --n_test 10 \
  --save_pyg
```
Creates `train/val/test` lists, `scalers.pkl`, and `templates/`.

## Train on GPU
```bash
source .venv/bin/activate
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
Outputs: `runs/<timestamp>_HetroGATconcat_ATSP20_Comboss_st_tt_pp/trial_0/{best_model.pt,results.json}`.

## Evaluate / Test on GPU
```bash
source .venv/bin/activate
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
Creates `runs/<timestamp>.../trial_0/test_atsp20/results.json` (per-instance stats).

## Notes
- `DGL_NO_GRAPHBOLT=1` skips optional native GraphBolt module (not bundled in wheel).
- Ensure `/root/LKH-3.0.9/LKH` exists if you regenerate datasets/regrets (generator uses it).
- Adjust `batch_size`, `n_epochs`, and dataset path as needed for larger instances.
