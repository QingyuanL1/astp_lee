import inspect
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from src.arch_search.spec import ArchitectureSpec, LayerSpec, load_architecture_spec

# Helper classes (from model_utils.py; included here for completeness)
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, skip=False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d((hidden_dim))
        self.skip = skip

    def forward(self, x):
        h = F.relu(self.batch_norm(self.fc1(x)))
        h = self.fc2(h) + h if self.skip else self.fc2(h)
        return h

class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, G=None):
        if G is not None:
            y = self.module(G, x).flatten(1)
        else:
            y = self.module(x)
        return x + y

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.gat = SkipConnection(dglnn.GATConv(embed_dim, embed_dim // num_heads, num_heads))

        self.feed_forward = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embed_dim)
                ),
            ),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, g, h):
        h = self.gat(h, G=g).flatten(1)
        return self.feed_forward(h)

# Consolidated EdgePropertyPredictionModel (covers variants 0-3)
class EdgePropertyPredictionModel(nn.Module):
    """
    Consolidated model for edge property prediction.
    - Original: jk=None, skip_connection=False
    - Original+SC: jk=None, skip_connection=True
    - Original+JK: jk='cat, skip_connection=False
    - Original+JK+SC: jk='cat, skip_connection=True
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_gnn_layers=4, num_heads=16, jk=None, skip_connection=False):
        super().__init__()
        self.embed_layer = MLP(input_dim, hidden_dim, hidden_dim, skip=True)
        self.message_passing_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim, num_heads, hidden_dim*2) for _ in range(num_gnn_layers)]
        )
        self.skip_connection = skip_connection
        self.jk = None
        jk_dim = hidden_dim
        if jk:
            self.jk = dglnn.JumpingKnowledge(jk)
            if jk == 'cat':
                jk_dim = hidden_dim * (num_gnn_layers + 1)
        self.decision_layer = MLP(jk_dim, hidden_dim, output_dim)

    def forward(self, G, x):
        h = self.embed_layer(x)
        xs = [h]
        for l in self.message_passing_layers:
            h_new = l(G, h)
            if self.skip_connection:
                h_new += h
            h = h_new
            xs.append(h)
        if self.jk:
            h = self.jk(xs)
        h = self.decision_layer(h)
        return h

# Consolidated HetroGAT (covers Sum and Concat)
class HetroGAT(nn.Module):
    """
    Heterogeneous GAT with robust sum, concat (sum-like), and attention.
    - 'sum' path is exactly the original.
    - 'concat' path now matches sum results if needed (per-relation MLP applied independently).
    - 'attn' path computes per-hidden-dim attention across relation outputs.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, relation_types,
                 num_gnn_layers=4, num_heads=16, agg='sum'):
        super().__init__()
        assert agg in ('sum', 'concat', 'attn')
        self.relation_types = relation_types
        self.num_edge_types = len(self.relation_types)
        self.agg = agg
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.embed_layer = MLP(input_dim, hidden_dim, hidden_dim, skip=True)

        agg_mode = 'stack' if agg in ('concat', 'attn') else 'sum'

        self.gnn_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.relation_mlps = nn.ModuleList() if agg == 'concat' else None

        for _ in range(num_gnn_layers):
            conv_dict = {rel: dglnn.GATConv(hidden_dim, hidden_dim // num_heads, num_heads)
                         for rel in self.relation_types}
            self.gnn_layers.append(dglnn.HeteroGraphConv(conv_dict, aggregate=agg_mode))

            if agg == 'concat':
                # MLP per relation, applied independently to match sum
                self.relation_mlps.append(nn.ModuleList([
                    MLP(hidden_dim, max(hidden_dim // 2, 1), hidden_dim) for _ in self.relation_types
                ]))
                # single mlp_layer used after sum of per-relation outputs
                self.mlp_layers.append(MLP(hidden_dim, max(hidden_dim // 2, 1), hidden_dim))
            else:
                self.mlp_layers.append(MLP(hidden_dim, max(hidden_dim // 2, 1), hidden_dim))

        if agg == 'attn':
            self.attn_layers = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.num_edge_types, hidden_dim)) for _ in range(num_gnn_layers)]
            )
        else:
            self.attn_layers = None

        self.decision_layer = MLP(hidden_dim, max(hidden_dim // 2, 1), output_dim)

    def _stack_rel_out(self, rel_out):
        """Convert hetero outputs to [N, R, F] tensor."""
        if isinstance(rel_out, list):
            return torch.stack([t.flatten(1) if t.dim() > 2 else t for t in rel_out], dim=1)
        t = rel_out
        if t.dim() == 4:  # [N, R, H, D]
            return t.flatten(2)
        elif t.dim() == 3:
            if t.shape[1] == self.num_edge_types:  # [N, R, F]
                return t
            else:  # single relation with heads
                return t.flatten(1).unsqueeze(1)
        elif t.dim() == 2:
            return t.unsqueeze(1)
        else:
            raise RuntimeError(f"Unexpected tensor dim {t.dim()}")

    def forward(self, graph, inputs):
        with graph.local_scope():
            inputs = self.embed_layer(inputs)
            ntype = graph.ntypes[0]
            h = {ntype: inputs}

            for layer_idx, (gnn_layer, mlp_layer) in enumerate(zip(self.gnn_layers, self.mlp_layers)):
                out = gnn_layer(graph, h)
                rel_out = out[ntype]
                stacked = self._stack_rel_out(rel_out)  # [N, R, F]

                if self.agg == 'sum':
                    # rel_out from HeteroGraphConv with aggregate='sum' is [N, num_heads, per_head_dim]
                    # Flatten to [N, hidden_dim] before MLP
                    v = rel_out.flatten(1)  # [N, num_heads * per_head_dim] = [N, hidden_dim]
                    v = F.leaky_relu(v)
                    v = mlp_layer(v)
                    v = v + h[ntype]
                    h = {ntype: v}

                elif self.agg == 'concat':
                    # Apply per-relation MLP independently
                    per_rel = [mlp(t) for t, mlp in zip(stacked.transpose(0,1), self.relation_mlps[layer_idx])]
                    # Sum to preserve sum-like behaviour
                    v = sum(per_rel)
                    v = mlp_layer(v)
                    v = v + h[ntype]
                    h = {ntype: v}

                elif self.agg == 'attn':
                    attn_param = self.attn_layers[layer_idx]  # [R, F]
                    attn_scores = F.softmax(attn_param, dim=0).unsqueeze(0)  # [1, R, F]
                    weighted = (stacked * attn_scores).sum(dim=1)
                    weighted = mlp_layer(weighted)
                    weighted = weighted + h[ntype]
                    h = {ntype: weighted}

            h_out = torch.cat(list(h.values()), dim=1) if len(h) > 1 else list(h.values())[0]
            return self.decision_layer(h_out)


class ZeroConv(nn.Module):
    """A placeholder operator that disables a relation."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, graph, feat):
        if isinstance(feat, tuple):
            feat_dst = feat[1] if feat[1] is not None else feat[0]
        else:
            feat_dst = feat
        return torch.zeros(
            (feat_dst.shape[0], self.hidden_dim),
            device=feat_dst.device,
            dtype=feat_dst.dtype,
        )


class RelationOp(nn.Module):
    """Wraps DGL ops so every relation outputs the same hidden_dim."""

    def __init__(self, op_type: str, hidden_dim: int, num_heads: int):
        super().__init__()
        self.op_type = op_type
        self.hidden_dim = hidden_dim
        self.num_heads = max(int(num_heads), 1)

        if op_type == "zero":
            self.conv = ZeroConv(hidden_dim)
        elif op_type == "gcn":
            self.conv = dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        elif op_type == "gat":
            per_head = max(hidden_dim // self.num_heads, 1)
            self.conv = dglnn.GATConv(hidden_dim, per_head, self.num_heads)
        elif op_type == "sage":
            self.conv = dglnn.SAGEConv(hidden_dim, hidden_dim, aggregator_type="mean")
        else:
            raise ValueError(f"Unsupported op type '{op_type}'")

        self.proj = None
        if op_type == "gat":
            out_dim = self.num_heads * max(hidden_dim // self.num_heads, 1)
            if out_dim != hidden_dim:
                self.proj = nn.Linear(out_dim, hidden_dim)

    def forward(self, graph, feat):
        h = self.conv(graph, feat)
        if isinstance(h, tuple):
            h = h[0]
        if h.dim() == 3:
            h = h.flatten(1)
        if self.proj is not None and h.shape[1] != self.hidden_dim:
            h = self.proj(h)
        return h


class HGNASLayer(nn.Module):
    """Single layer used by HGNASModel."""

    def __init__(self, relation_ops, aggregator: str, hidden_dim: int, num_heads: int):
        super().__init__()
        self.aggregator = aggregator
        self.hidden_dim = hidden_dim
        self.conv = dglnn.HeteroGraphConv(
            {rel: RelationOp(op, hidden_dim, num_heads) for rel, op in relation_ops.items()},
            aggregate="stack",
        )
        if aggregator == "attn":
            self.attn_param = nn.Parameter(torch.zeros(len(relation_ops), hidden_dim))
        else:
            self.attn_param = None
        self.post_mlp = MLP(hidden_dim, max(hidden_dim // 2, 1), hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _aggregate(self, stacked: torch.Tensor) -> torch.Tensor:
        if stacked.dim() == 2:
            return stacked
        if self.aggregator == "sum":
            agg = stacked.sum(dim=1)
        elif self.aggregator == "mean":
            agg = stacked.mean(dim=1)
        elif self.aggregator == "max":
            agg, _ = stacked.max(dim=1)
        elif self.aggregator == "min":
            agg, _ = stacked.min(dim=1)
        elif self.aggregator == "attn":
            weights = F.softmax(self.attn_param, dim=0).unsqueeze(0)
            agg = (stacked * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregator '{self.aggregator}'")
        return agg

    def forward(self, graph, inputs: torch.Tensor) -> torch.Tensor:
        ntype = graph.ntypes[0]
        rel_out = self.conv(graph, {ntype: inputs})[ntype]
        agg = self._aggregate(rel_out)
        agg = self.post_mlp(agg)
        agg = self.norm(agg)
        return agg


class HGNASModel(nn.Module):
    """
    Heterogeneous GNN that materializes architecture specs emitted by GPT.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        relation_types,
        num_gnn_layers: int = 2,
        num_heads: int = 8,
        architecture: ArchitectureSpec | None = None,
        architecture_path: str | None = None,
        arch_num_layers: int | None = None,
        arch_default_op: str = "gat",
        arch_default_agg: str = "sum",
    ):
        super().__init__()
        self.relation_types = list(relation_types)
        self.num_heads = num_heads
        target_layers = arch_num_layers or num_gnn_layers

        if architecture is None and architecture_path:
            architecture = load_architecture_spec(architecture_path)
        if architecture is None:
            architecture = ArchitectureSpec.default(
                relation_types=self.relation_types,
                num_layers=target_layers,
                default_op=arch_default_op,
                aggregator=arch_default_agg,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            )
        elif target_layers is not None and int(target_layers) != int(architecture.num_layers):
            target_layers = int(target_layers)
            if target_layers < architecture.num_layers:
                architecture.layers = architecture.layers[:target_layers]
            else:
                last = architecture.layers[-1]
                for _ in range(target_layers - architecture.num_layers):
                    architecture.layers.append(LayerSpec(ops=dict(last.ops), aggregator=last.aggregator))
        architecture.ensure_valid()
        if sorted(architecture.relation_types) != sorted(self.relation_types):
            raise ValueError(
                f"Architecture relation types {architecture.relation_types} "
                f"do not match runtime {self.relation_types}"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.architecture = architecture
        self.embed_layer = MLP(input_dim, self.hidden_dim, self.hidden_dim, skip=True)
        self.layers = nn.ModuleList(
            [
                HGNASLayer(layer.ops, layer.aggregator, self.hidden_dim, self.num_heads)
                for layer in architecture.layers
            ]
        )
        self.decision_layer = MLP(self.hidden_dim, max(self.hidden_dim // 2, 1), output_dim)

    def forward(self, graph, inputs):
        h = self.embed_layer(inputs)
        for layer in self.layers:
            h_new = layer(graph, h)
            h = F.relu(h_new + h)
        return self.decision_layer(h)

# Simple GCN model (unchanged)
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = dglnn.GraphConv(input_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, output_dim)

    def forward(self, G, x):
        h = F.relu(self.conv1(G, x))
        h = self.conv2(G, h)
        return h

# Updated get_model to handle consolidated classes
def get_dgl_model(args):
    import torch
    import inspect
    from src.models.models_dgl import EdgePropertyPredictionModel, HetroGAT, HGNASModel, GCNModel

    model_classes = {
        'EdgePropertyPredictionModel': EdgePropertyPredictionModel,
        'HetroGAT': HetroGAT,
        'HGNASModel': HGNASModel,
        'GCNModel': GCNModel,
    }

    if args.model not in model_classes:
        raise ValueError(f"Model '{args.model}' not recognized")

    model_class = model_classes[args.model]
    model_signature = inspect.signature(model_class)

    if args.model_path and args.model_path.strip():
        # Load checkpoint dict
        checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
        ckpt_args = checkpoint.get("args", {})
        arch_key = ckpt_args.get("architecture_path")
        if arch_key and not os.path.isabs(arch_key):
            ckpt_args["architecture_path"] = os.path.join(os.path.dirname(args.model_path), arch_key)

        # Keep only keys that match model constructor
        model_args = {k: ckpt_args[k] for k in model_signature.parameters if k in ckpt_args}
        # Instantiate model and load weights
        model = model_class(**model_args).to(args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(
            f"Loaded model '{args.model}' from '{args.model_path}' "
            f"(epoch={checkpoint.get('epoch', '?')}, val_loss={checkpoint.get('val_loss', '?')})"
        )
    else:
        # Fresh model from current args
        model_args = {k: getattr(args, k) for k in model_signature.parameters if hasattr(args, k)}
        model = model_class(**model_args).to(args.device)
        print(f"Created fresh model '{args.model}' from args.")

    return model
