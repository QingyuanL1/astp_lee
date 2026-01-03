import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from src.arch_search.spec import load_architecture_spec
from src.engine.test_dgl import ATSPTesterDGL
from src.models.models_dgl import HGNASModel
from src.utils import fix_seed


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload a saved HGNAS architecture + checkpoint and run the standard ATSP tester."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt checkpoint.")
    parser.add_argument(
        "--spec",
        type=str,
        default=None,
        help="Optional architecture.json path. Defaults to the checkpoint metadata.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Processed dataset directory (contains train/val/test splits).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--relation_types",
        nargs="+",
        default=None,
        help="Override relation types. Defaults to the checkpoint metadata.",
    )
    parser.add_argument("--atsp_size", type=int, default=None, help="Override ATSP size.")
    parser.add_argument("--time_limit", type=float, default=5.0 / 30.0)
    parser.add_argument("--perturbation_moves", type=int, default=30)
    parser.add_argument("--perturbation_count", type=int, default=5)
    parser.add_argument("--undirected", action="store_true", help="Treat template graphs as undirected.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory to place test results (default: next to checkpoint).",
    )
    return parser.parse_args()


def _resolve_spec_path(spec_arg: str | None, ckpt_args: dict, checkpoint_path: Path) -> Path:
    spec_path = spec_arg or ckpt_args.get("architecture_path")
    if not spec_path:
        raise ValueError("Architecture JSON path was not provided and is missing from checkpoint args.")
    spec_path = Path(spec_path)
    if not spec_path.is_absolute():
        spec_path = checkpoint_path.parent / spec_path
    if not spec_path.exists():
        raise FileNotFoundError(f"Architecture spec not found: {spec_path}")
    return spec_path


def _build_model(ckpt: dict, spec_path: Path, device: torch.device, rel_types) -> HGNASModel:
    ckpt_args = ckpt.get("args", {})
    required = ["input_dim", "hidden_dim", "output_dim", "num_gnn_layers", "num_heads"]
    missing = [key for key in required if key not in ckpt_args]
    if missing:
        raise KeyError(f"Checkpoint args missing required fields: {missing}")

    spec = load_architecture_spec(spec_path)
    model = HGNASModel(
        input_dim=ckpt_args["input_dim"],
        hidden_dim=ckpt_args["hidden_dim"],
        output_dim=ckpt_args["output_dim"],
        relation_types=rel_types,
        num_gnn_layers=ckpt_args["num_gnn_layers"],
        num_heads=ckpt_args["num_heads"],
        architecture=spec,
    ).to(device)
    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint missing 'model_state_dict'.")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _tester_args(cli_args: argparse.Namespace, ckpt_args: dict, checkpoint_path: Path) -> SimpleNamespace:
    rel_types = cli_args.relation_types or ckpt_args.get("relation_types")
    if rel_types is None:
        raise ValueError("relation_types missing; provide --relation_types or ensure checkpoint stored them.")
    atsp_size = cli_args.atsp_size or ckpt_args.get("atsp_size")
    if atsp_size is None:
        raise ValueError("atsp_size missing; provide --atsp_size or ensure checkpoint stored it.")

    return SimpleNamespace(
        data_path=cli_args.data_path,
        atsp_size=atsp_size,
        relation_types=rel_types,
        undirected=cli_args.undirected or ckpt_args.get("undirected", False),
        device=cli_args.device,
        time_limit=cli_args.time_limit,
        perturbation_moves=cli_args.perturbation_moves,
        perturbation_count=cli_args.perturbation_count,
        model_path=str(checkpoint_path),
        results_dir=cli_args.output_dir,
    )


def main():
    cli_args = parse_cli()
    fix_seed(cli_args.seed)
    checkpoint_path = Path(cli_args.checkpoint).expanduser().resolve()
    device = torch.device(cli_args.device if cli_args.device else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    if not ckpt_args:
        raise KeyError("Checkpoint does not contain 'args' metadata. Retrain or pass explicit overrides.")

    tester_args = _tester_args(cli_args, ckpt_args, checkpoint_path)
    relation_types = tester_args.relation_types
    spec_path = _resolve_spec_path(cli_args.spec, ckpt_args, checkpoint_path)

    model = _build_model(ckpt, spec_path, device, relation_types)
    tester = ATSPTesterDGL(tester_args)
    results = tester.run_test(model)
    result_dir = tester_args.results_dir
    if not result_dir:
        result_dir = str(Path(tester_args.model_path).parent / f"test_atsp{tester_args.atsp_size}")
    result_path = Path(result_dir) / f"results_test_{tester_args.atsp_size}.json"
    if result_path.exists():
        print(f"Saved evaluation summary to {result_path}")
    else:
        print(f"Finished evaluation. Expected results JSON at {result_path}")


if __name__ == "__main__":
    main()
