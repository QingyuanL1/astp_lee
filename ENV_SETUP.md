# Environment Setup (.venv)

This project is designed to run inside a Python virtual environment. The steps below create `.venv`, install CPU/GPU dependencies, and prepare optional extras (torch_geometric, OpenAI SDK, etc.).

## 1. Create and Activate Virtualenv

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip wheel
```

## 2. Install PyTorch + DGL

Choose **one** of the following depending on your hardware.

### GPU (CUDA 12.1 example)
```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0 torchaudio==2.3.0 \
  --extra-index-url https://download.pytorch.org/whl/cu121
pip install dgl==2.5.0+cu121 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
```

### CPU
```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
pip install dgl==2.5.0
```

> Adjust versions if you require a different CUDA toolkit (cu118/cu124 etc.). Always match DGL’s wheel to the installed torch.

## 3. Core Python Dependencies

```bash
pip install numpy networkx pandas tqdm tsplib95 optuna matplotlib PyYAML python-dotenv
```

Used for data processing, NAS logging, and visualization.

## 4. Optional Packages

| Package | Why | Install |
| --- | --- | --- |
| `torch_geometric` | Preprocessor saves PyG templates (`--save_pyg`). | `pip install torch_geometric` |
| `lkh` (Python wrapper) | Included automatically with `dataset_generator.py`; ensure the native LKH binary is built separately. | Already in `requirements`; no action if pip succeeded. |
| `openai` | Needed for GPT NAS when not using offline mode. | `pip install openai` |

> Ensure the system has the compiled `LKH` executable (see README step 1) and set `--lkh_path /path/to/LKH`.

## 5. Environment Variables (for NAS)

- Create `.env` (ignored by Git) with:
  ```
  OPENAI_API_KEY=sk-...
  # optional:
  # OPENAI_BASE_URL=https://aihubmix.com/v1
  ```
- Load via `source .env` or pass `--llm_api_key` / `--llm_base_url` when running `python -m src.engine.run --mode arch ...`.

## 6. Quick Validation

```bash
source .venv/bin/activate
python - <<'PY'
import torch, dgl
print('Torch CUDA:', torch.cuda.is_available())
print('DGL Version:', dgl.__version__)
PY
```

If the imports succeed and CUDA availability matches expectation, the environment is ready.
