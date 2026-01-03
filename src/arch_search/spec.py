from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class LayerSpec:
    """Per-layer heterograph configuration."""

    ops: Dict[str, str]
    aggregator: str

    def to_dict(self) -> Dict[str, Any]:
        return {"ops": self.ops, "aggregator": self.aggregator}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerSpec":
        return cls(ops=dict(data.get("ops", {})), aggregator=data.get("aggregator", "sum"))


@dataclass
class ArchitectureSpec:
    """Architecture description for HGNASModel."""

    relation_types: List[str]
    layers: List[LayerSpec]
    hidden_dim: int
    num_heads: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_types": self.relation_types,
            "layers": [layer.to_dict() for layer in self.layers],
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "metadata": self.metadata,
        }

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def ensure_valid(self) -> None:
        if not self.layers:
            raise ValueError("ArchitectureSpec.layers must be non-empty")
        for idx, layer in enumerate(self.layers):
            missing = [rel for rel in self.relation_types if rel not in layer.ops]
            if missing:
                raise ValueError(f"Layer {idx} missing ops for relations: {missing}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchitectureSpec":
        layers = [LayerSpec.from_dict(ld) for ld in data.get("layers", [])]
        spec = cls(
            relation_types=list(data.get("relation_types", [])),
            layers=layers,
            hidden_dim=int(data.get("hidden_dim", 64)),
            num_heads=int(data.get("num_heads", 4)),
            metadata=dict(data.get("metadata", {})),
        )
        spec.ensure_valid()
        return spec

    @classmethod
    def default(
        cls,
        relation_types: List[str],
        num_layers: int,
        default_op: str = "gat",
        aggregator: str = "sum",
        hidden_dim: int = 64,
        num_heads: int = 8,
    ) -> "ArchitectureSpec":
        layers = [
            LayerSpec({rel: default_op for rel in relation_types}, aggregator=aggregator)
            for _ in range(num_layers)
        ]
        return cls(
            relation_types=list(relation_types),
            layers=layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            metadata={"origin": "default"},
        )


def load_architecture_spec(path: str | Path) -> ArchitectureSpec:
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)
    return ArchitectureSpec.from_dict(data)


def save_architecture_spec(spec: ArchitectureSpec, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(spec.to_dict(), f, indent=2)
