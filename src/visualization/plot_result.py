import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_baselines():
    base_sizes = [100, 150, 250, 500, 1000]
    methods = [
        {
            "label": "MatNet",
            "gap": [1.84, 82.92, 181.68, None, None],
            "time": [3.24, 4.24, 8.48, None, None],
            "color": "#ff7f0e",
            "marker": "^",
            "linestyle": (0, (5, 2, 1, 2, 1, 2)),
        },
        {
            "label": "GLOP",
            "gap": [8.85, 17.91, 27.96, 38.96, 47.77],
            "time": [14.25, 13.99, 14.86, 17.06, 22.57],
            "color": "#2ca02c",
            "marker": "o",
            "linestyle": (0, (1, 2)),
        },
        {
            "label": "Het-GAT-Concat + E2",
            "gap": [9.39, 10.84, 13.49, 17.03, None],
            "time": [3.32, 4.44, 6.63, 15.05, None],
            "heuristic": [2.67, 3.79, 5.97, 14.30, None],
            "color": "#1f77b4",
            "marker": "x",
            "linestyle": (0, (5, 5)),
        },
        {
            "label": "Het-GAT-Concat + E3",
            "gap": [2.84, 4.88, 8.34, 7.89, None],
            "time": [2.35, 2.80, 0.75, 7.20, None],
            "heuristic": [1.70, 2.15, 0.09, 6.45, None],
            "color": "#d62728",
            "marker": "s",
            "linestyle": (0, (5, 2, 1, 2)),
        },
        {
            "label": "Het-GAT-Attn + E2",
            "gap": [10.86, 12.08, 14.29, 16.25, 58.94],
            "time": [3.23, 4.66, 8.26, 37.17, 95.50],
            "heuristic": [2.65, 3.83, 5.87, 14.25, 52.48],
            "color": "#9467bd",
            "marker": "D",
            "linestyle": (0, (3, 1, 1, 1)),
        },
        {
            "label": "Het-GAT-Attn + E3",
            "gap": [3.45, 4.76, 8.50, 8.54, 53.14],
            "time": [2.52, 3.11, 2.48, 23.37, 52.73],
            "heuristic": [1.94, 2.28, 0.09, 0.45, 9.71],
            "color": "#8c564b",
            "marker": "P",
            "linestyle": (0, (1, 1)),
        },
    ]

    return base_sizes, methods


def update_metric_maps(entries, base_sizes):
    for entry in entries:
        entry["gap_map"] = {size: val for size, val in zip(base_sizes, entry.get("gap", []))}
        entry["time_map"] = {size: val for size, val in zip(base_sizes, entry.get("time", []))}
        if "heuristic" in entry:
            entry["heuristic_map"] = {size: val for size, val in zip(base_sizes, entry["heuristic"])}
        entry.setdefault("cost_map", {})


def load_architecture_methods(base_dir: str = "arch_search_runs"):
    base = Path(base_dir)
    if not base.exists():
        return []
    arch_entries = []
    for summary_path in sorted(base.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        best_metrics = summary.get("best_metrics", {})
        size = summary.get("config", {}).get("atsp_size")
        if size is None or best_metrics.get("avg_final_gap") is None:
            continue
        label = f"LLM HGNAS (n={size})"
        color = "#1f9a8a"
        entry = {
            "label": label,
            "gap_map": {size: best_metrics["avg_final_gap"]},
            "time_map": {size: best_metrics["avg_total_time"]},
            "cost_map": {size: best_metrics.get("avg_final_cost")},
            "marker": "o",
            "linestyle": "-.",
            "color": color,
        }
        arch_entries.append(entry)
    return arch_entries


base_sizes, methods = load_baselines()
update_metric_maps(methods, base_sizes)
methods.extend(load_architecture_methods())

all_sizes = sorted({size for m in methods for size in set(m.get("gap_map", {}).keys())})

def map_to_series(mapping, sizes):
    return [mapping.get(size) for size in sizes]

for entry in methods:
    entry["gap"] = map_to_series(entry.get("gap_map", {}), all_sizes)
    entry["time"] = map_to_series(entry.get("time_map", {}), all_sizes)
    if "heuristic_map" in entry:
        entry["heuristic"] = map_to_series(entry["heuristic_map"], all_sizes)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

# Plot execution time
for m in methods:
    # Main time
    ax1.plot(all_sizes, m["time"], marker=m["marker"], markersize=16,
             markeredgewidth=2.5, markerfacecolor='none',
             linestyle=m["linestyle"], linewidth=4,
             color=m["color"], label=m["label"])
    # Heuristic phase (if available)
    if "heuristic" in m:
        ax1.plot(all_sizes, m["heuristic"], marker=m["marker"], markersize=10,
                 markeredgewidth=1.5, markerfacecolor='none',
                 linestyle=":", linewidth=2,
                 color=m["color"], alpha=0.7, label=f"{m['label']} (heuristic)")

# Plot gap comparison
for m in methods:
    ax2.plot(all_sizes, m["gap"], marker=m["marker"], markersize=16,
             markeredgewidth=2.5, markerfacecolor='none',
             linestyle=m["linestyle"], linewidth=4,
             color=m["color"], label=m["label"])

# Log scale and formatting
for ax in (ax1, ax2):
    ax.set_xlabel('ATSP Size', fontsize=22)
    ax.set_xticks(all_sizes)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, linewidth=1.5, which='both')
    ax.set_yscale('log')
    ax.set_ylim(1, 200)

ax1.set_title('Execution Time Comparison (log scale)', fontsize=26, pad=25)
ax1.set_ylabel('Time (s)', fontsize=22)
ax2.set_title('Average Gap Comparison (log scale)', fontsize=26, pad=25)
ax2.set_ylabel('Avg Gap (%)', fontsize=22)

# Legends
ax1.legend(loc='upper left', fontsize=14, framealpha=0.9, handlelength=3, handletextpad=1)
ax2.legend(loc='upper left', fontsize=14, framealpha=0.9, handlelength=3, handletextpad=1)

plt.tight_layout(pad=5.0)
plt.savefig('result_plot_full.pdf', bbox_inches='tight', dpi=300)
plt.close()
