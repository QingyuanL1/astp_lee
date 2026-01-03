from __future__ import annotations

import copy
import datetime
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from src.engine.train_dgl import ATSPTrainerDGL
from src.engine.test_dgl import ATSPTesterDGL
from src.models.models_dgl import HGNASModel
from src.utils import fix_seed
from .llm_client import ChatCompletionClient
from .search_space import HGNSearchSpace, HistoryEntry
from .spec import save_architecture_spec

logger = logging.getLogger("arch_search")


class ArchitectureSearchRunner:
    """Coordinates GPT-driven architecture sampling and evaluation."""

    def __init__(self, args):
        self.args = args
        num_layers = args.arch_num_layers or args.num_gnn_layers
        self.space = HGNSearchSpace(args.relation_types, num_layers)
        self.dataset_desc = f"ATSP size {args.atsp_size} with relations {','.join(args.relation_types)}"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(args.arch_save_dir or "arch_search_runs")
        self.run_dir = base_dir / f"{timestamp}_atsp{args.atsp_size}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[HistoryEntry] = []
        self.records: List[Dict[str, Any]] = []
        self.random = random.Random(args.llm_seed or args.seed or 42)
        self.llm_offline = bool(getattr(args, "llm_offline", False))
        self.client = None
        if not self.llm_offline:
            self.client = ChatCompletionClient(
                api_key=args.llm_api_key,
                model=args.llm_model,
                base_url=args.llm_base_url,
                timeout=args.llm_timeout,
            )
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.run_dir / "search.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    def run(self) -> Dict[str, Any]:
        logger.info("Starting architecture search. Results at %s", self.run_dir)
        n_iterations = self.args.nas_iterations
        per_iter = self.args.nas_batch_size

        for iteration in range(n_iterations):
            logger.info("=== Iteration %d ===", iteration)
            vectors = self._propose_vectors(iteration, per_iter)
            if not vectors:
                logger.warning("No architectures parsed in iteration %d; using random fallback", iteration)
                vectors = [self.space.random_vector(self.random) for _ in range(per_iter)]

            for cand_idx, vector in enumerate(vectors):
                key = tuple(vector)
                if any((key == tuple(entry.vector) for entry in self.history)):
                    continue
                try:
                    record = self._evaluate(vector, iteration, cand_idx)
                except Exception as e:
                    logger.exception("Evaluation failed for vector %s: %s", vector, e)
                    continue
                self.records.append(record)
                self.history.append(
                    HistoryEntry(vector=vector, score=record["score"], summary=record["summary"])
                )
                self._write_history()

        summary = self._write_summary()
        best = summary.get("best_record")
        if best:
            logger.info("Architecture search complete. Best score %.4f", best["score"])
        else:
            logger.info("Architecture search finished but no valid records were saved.")
        return summary

    def _propose_vectors(self, iteration: int, per_iter: int) -> List[List[int]]:
        prompt = self.space.build_prompt(
            dataset_desc=self.dataset_desc,
            batch_size=per_iter,
            iteration=iteration,
            history=self.history,
        )
        if self.llm_offline:
            logger.info("Offline mode: sampling %d random architectures", per_iter)
            return [self.space.random_vector(self.random) for _ in range(per_iter)]

        messages = [
            {
                "role": "system",
                "content": "You are an expert hetero-GNN NAS assistant. Respond with integer vectors only.",
            },
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat(
            messages,
            temperature=self.args.llm_temperature,
            max_tokens=self.args.llm_max_tokens,
        )
        raw_path = self.run_dir / f"llm_iter{iteration:02d}.txt"
        raw_path.write_text(response, encoding="utf-8")
        vectors = self.space.parse_response(response)
        # ensure we have per_iter suggestions
        while len(vectors) < per_iter:
            vectors.append(self.space.random_vector(self.random))
        return vectors[:per_iter]

    def _evaluate(self, vector: Sequence[int], iteration: int, cand_idx: int) -> Dict[str, Any]:
        fix_seed(self.args.seed + iteration * 100 + cand_idx if hasattr(self.args, "seed") else 42)
        spec = self.space.decode_vector(vector, hidden_dim=self.args.hidden_dim, num_heads=self.args.num_heads)
        arch_dir = self.run_dir / f"iter{iteration:02d}_cand{cand_idx:02d}"
        arch_dir.mkdir(parents=True, exist_ok=True)
        spec_path = arch_dir / "architecture.json"
        save_architecture_spec(spec, spec_path)

        model = HGNASModel(
            input_dim=self.args.input_dim,
            hidden_dim=self.args.hidden_dim,
            output_dim=self.args.output_dim,
            relation_types=self.args.relation_types,
            num_gnn_layers=self.args.num_gnn_layers,
            num_heads=self.args.num_heads,
            architecture=spec,
        )
        train_args = copy.deepcopy(self.args)
        train_args.architecture_path = spec_path.name
        train_args.model = "HGNASModel"
        trainer = ATSPTrainerDGL(train_args, save_model=False)
        start = time.time()
        train_results, best_state = trainer.train(model, trial_id=0, return_best_state=True)
        train_time = time.time() - start
        checkpoint_path = arch_dir / "best_model.pt"
        torch.save(
            {
                "epoch": train_results.get("best_epoch"),
                "val_loss": train_results.get("best_val_loss"),
                "model_state_dict": best_state,
                "args": vars(train_args),
            },
            checkpoint_path,
        )

        test_args = copy.deepcopy(self.args)
        test_args.data_path = test_args.data_dir
        tester = ATSPTesterDGL(test_args)
        test_dataset = tester.create_test_dataset()
        model.eval()
        with torch.no_grad():
            test_metrics = tester.test_all(model, test_dataset)
        eval_time = test_metrics.get("total_model_times", 0.0) or 0.0
        total_time = train_time + eval_time

        record = {
            "vector": list(vector),
            "spec_path": str(spec_path),
            "checkpoint_path": str(checkpoint_path),
            "train_results": train_results,
            "test_metrics": test_metrics,
            "train_time": train_time,
            "total_time": total_time,
        }
        record["score"] = self._score(train_results, test_metrics)
        record["summary"] = self._summary_text(train_results, test_metrics)
        result_path = arch_dir / "result.json"
        result_path.write_text(json.dumps(record, indent=2, default=self._json_default), encoding="utf-8")
        return record

    def _score(self, train_results: Dict[str, Any], test_metrics: Dict[str, Any]) -> float:
        metric = self.args.nas_metric
        if metric == "val_loss":
            return float(train_results.get("best_val_loss", float("inf")))
        if metric == "gap":
            return float(test_metrics.get("avg_final_gaps", float("inf")))
        if metric == "cost":
            return float(test_metrics.get("avg_final_costs", float("inf")))
        if metric == "time":
            return float(
                (test_metrics.get("avg_model_times", float("inf")) or float("inf"))
                + (test_metrics.get("avg_gls_times", float("inf")) or 0.0)
            )
        raise ValueError(f"Unknown NAS metric '{metric}'")

    def _summary_text(self, train_results: Dict[str, Any], test_metrics: Dict[str, Any]) -> str:
        gap = test_metrics.get("avg_final_gaps")
        cost = test_metrics.get("avg_final_costs")
        model_time = test_metrics.get("avg_model_times")
        gls_time = test_metrics.get("avg_gls_times")
        parts = []
        if gap is not None:
            parts.append(f"gap={gap:.3f}%")
        if cost is not None:
            parts.append(f"cost={cost:.2f}")
        if model_time is not None and gls_time is not None:
            parts.append(f"time={model_time + gls_time:.2f}s")
        val = train_results.get("best_val_loss")
        if val is not None:
            parts.append(f"val_loss={val:.5f}")
        return ", ".join(parts)

    def _write_history(self) -> None:
        hist_path = self.run_dir / "history.json"
        payload = [
            {"vector": entry.vector, "score": entry.score, "summary": entry.summary}
            for entry in self.history
        ]
        hist_path.write_text(json.dumps(payload, indent=2, default=self._json_default), encoding="utf-8")

    def _write_summary(self) -> Dict[str, Any]:
        if not self.records:
            summary = {"best_record": None, "records": []}
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary
        best = min(self.records, key=lambda r: r["score"])
        summary = {
            "config": {
                "nas_iterations": self.args.nas_iterations,
                "nas_batch_size": self.args.nas_batch_size,
                "metric": self.args.nas_metric,
                "vector_length": self.space.vector_length,
                "atsp_size": self.args.atsp_size,
                "relation_types": list(self.args.relation_types),
            },
            "best_metrics": {
                "avg_final_gap": best["test_metrics"].get("avg_final_gaps"),
                "avg_final_cost": best["test_metrics"].get("avg_final_costs"),
                "avg_total_time": (
                    (best["test_metrics"].get("avg_model_times") or 0.0)
                    + (best["test_metrics"].get("avg_gls_times") or 0.0)
                ),
                "avg_model_time": best["test_metrics"].get("avg_model_times"),
                "avg_gls_time": best["test_metrics"].get("avg_gls_times"),
            },
            "best_record": best,
            "records": self.records,
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=self._json_default), encoding="utf-8"
        )
        return summary

    @staticmethod
    def _json_default(obj):
        try:
            return float(obj)
        except Exception:
            return str(obj)
