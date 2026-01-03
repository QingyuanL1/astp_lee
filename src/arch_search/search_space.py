from __future__ import annotations

import random
import re
import textwrap
from dataclasses import dataclass
from typing import List, Sequence

from .spec import ArchitectureSpec, LayerSpec


@dataclass
class HistoryEntry:
    vector: List[int]
    score: float
    summary: str


class HGNSearchSpace:
    """Search space helper for hetero GNN architecture vectors."""

    def __init__(
        self,
        relation_types: Sequence[str],
        num_layers: int,
        op_choices: Sequence[str] | None = None,
        agg_choices: Sequence[str] | None = None,
    ):
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.relation_types = list(relation_types)
        self.num_layers = int(num_layers)
        self.op_choices = list(op_choices or ["zero", "gcn", "gat", "sage"])
        self.agg_choices = list(agg_choices or ["sum", "mean", "max", "min", "attn"])

    @property
    def vector_length(self) -> int:
        return self.num_layers * (len(self.relation_types) + 1)

    def describe(self) -> str:
        ops = "\n".join(f"{idx}:{name}" for idx, name in enumerate(self.op_choices))
        aggs = "\n".join(f"{idx}:{name}" for idx, name in enumerate(self.agg_choices))
        return textwrap.dedent(
            f"""
            Search space:
              - Relation types per layer: {", ".join(self.relation_types)}
              - Number of layers: {self.num_layers}
              - Vector length: {self.vector_length} integers
              - For each layer, provide |relations| op ids followed by one aggregator id.
            Operation ids:
            {ops}

            Aggregator ids:
            {aggs}
            """
        ).strip()

    def decode_vector(self, vector: Sequence[int], hidden_dim: int, num_heads: int) -> ArchitectureSpec:
        if len(vector) != self.vector_length:
            raise ValueError(f"Expected vector of length {self.vector_length}, got {len(vector)}")
        idx = 0
        layers: List[LayerSpec] = []
        for _ in range(self.num_layers):
            ops = {}
            for rel in self.relation_types:
                op_idx = vector[idx]
                idx += 1
                if op_idx < 0 or op_idx >= len(self.op_choices):
                    raise ValueError(f"Invalid op index {op_idx} for relation {rel}")
                ops[rel] = self.op_choices[op_idx]
            agg_idx = vector[idx]
            idx += 1
            if agg_idx < 0 or agg_idx >= len(self.agg_choices):
                raise ValueError(f"Invalid aggregator index {agg_idx}")
            layers.append(LayerSpec(ops=ops, aggregator=self.agg_choices[agg_idx]))
        return ArchitectureSpec(
            relation_types=list(self.relation_types),
            layers=layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            metadata={"vector": list(vector)},
        )

    def parse_response(self, text: str) -> List[List[int]]:
        pattern = re.compile(r"arch\d+\s*=\s*\[([0-9,\s-]+)\]", re.IGNORECASE)
        vectors: List[List[int]] = []
        for match in pattern.finditer(text):
            payload = match.group(1)
            try:
                vector = [int(tok.strip()) for tok in payload.split(",") if tok.strip()]
            except ValueError:
                continue
            if len(vector) == self.vector_length:
                vectors.append(vector)
        # Fallback: attempt to parse a single JSON-like list
        if not vectors:
            fallback = re.findall(r"\[([0-9,\s-]+)\]", text)
            for payload in fallback:
                try:
                    vector = [int(tok.strip()) for tok in payload.split(",") if tok.strip()]
                except ValueError:
                    continue
                if len(vector) == self.vector_length:
                    vectors.append(vector)
        return vectors

    def random_vector(self, rng: random.Random) -> List[int]:
        vec: List[int] = []
        for _ in range(self.num_layers):
            for _ in self.relation_types:
                vec.append(rng.randrange(len(self.op_choices)))
            vec.append(rng.randrange(len(self.agg_choices)))
        return vec

    def build_prompt(
        self,
        dataset_desc: str,
        batch_size: int,
        iteration: int,
        history: List[HistoryEntry],
    ) -> str:
        intro = textwrap.dedent(
            f"""
            You are searching for heterogeneous GNN architectures for {dataset_desc}.
            Always respond with {batch_size} NEW architectures using the pattern:
            arch1=[...]
            arch2=[...]
            ...
            Each architecture must contain {self.vector_length} integers.
            """
        )
        history_text = ""
        if history:
            top = sorted(history, key=lambda x: x.score)[: min(len(history), 5)]
            history_lines = [
                f"arch(vector={item.vector}) -> score={item.score:.4f}, notes={item.summary}"
                for item in top
            ]
            history_text = "\nRecent evaluations:\n" + "\n".join(history_lines) + "\n"
        guidance = (
            "Explore novel combinations early on; once multiple evaluations exist, exploit the best-performing ones."
            if iteration < 3
            else "Focus on improving around the best historical architectures, avoid repeating weak patterns."
        )
        example = (
            "Example format:\narch1=[0,1,2,3,...]\narch2=[3,3,1,0,...]\nOnly provide the sequences."
        )
        return "\n".join(
            [
                intro.strip(),
                self.describe(),
                guidance,
                history_text.strip(),
                example,
            ]
        )
