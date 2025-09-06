# -*- coding: utf-8 -*-
"""
Plots for MIS experiments:
- Scatter: Fidelity vs Hardness
- Scatter: Fidelity vs Min Energy Gap
- Scatter: Hardness vs Min Energy Gap
- Example graphs: top-K highest Hardness and top-K lowest Hardness

Requirements:
    pip install pandas matplotlib networkx openpyxl
"""

import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ─────────────────────────────────────────────
# USER SETTINGS (edit if needed)
EXCEL_PATH = Path(r"C:\Users\nadav\OneDrive\Documents\פרויקט גמר\MIS\mis_results.xlsx")
OUTPUT_DIR = EXCEL_PATH.parent  # save figures next to the Excel file
TOP_K = 3                       # how many highest/lowest-hardness graphs to draw
RANDOM_LAYOUT_SEED = 42         # for spring_layout reproducibility

# ─────────────────────────────────────────────
# LOAD DATA
df = pd.read_excel(EXCEL_PATH)

# Defensive normalize of column names (strip spaces)
df.columns = [c.strip() for c in df.columns]

# Basic checks
required_cols = {"G(N,E)", "Hardness", "Fidelity", "Min Energy Gap"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Excel file is missing expected columns: {missing}")

# ─────────────────────────────────────────────
# PARSE EDGE SETS  {(i,j), (k,l), ...}  →  list[(i,j),...]
edge_pat = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")

def parse_edges(edge_str):
    """Return list of (i,j) tuples from a string like '{(0,1), (2,5), ...}'."""
    if not isinstance(edge_str, str):
        return []
    pairs = edge_pat.findall(edge_str)
    return [(int(i), int(j)) for i, j in pairs]

def infer_num_qubits(edges):
    """Infer N from the largest vertex index (fallback if 'N' not present)."""
    if not edges:
        return None
    max_idx = max(max(e) for e in edges)
    return max_idx + 1

# Pre-parse once for all rows
parsed_edges = df["G(N,E)"].apply(parse_edges)
if "N" in df.columns:
    Ns = df["N"].tolist()
else:
    Ns = [infer_num_qubits(e) for e in parsed_edges]

# ─────────────────────────────────────────────
# SCATTER PLOTS (one chart per figure, no explicit colors)
def scatter_xy(x, y, xlab, ylab, title, outname):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    out_path = OUTPUT_DIR / outname
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

# 1) Fidelity vs Hardness
scatter_xy(
    df["Hardness"].values,
    df["Fidelity"].values,
    xlab="Hardness",
    ylab="Fidelity",
    title="Fidelity vs Hardness",
    outname="scatter_fidelity_vs_hardness.png"
)

# 2) Fidelity vs Min Energy Gap
scatter_xy(
    df["Min Energy Gap"].values,
    df["Fidelity"].values,
    xlab="Min Energy Gap",
    ylab="Fidelity",
    title="Fidelity vs Min Energy Gap",
    outname="scatter_fidelity_vs_min_gap.png"
)

# 3) Hardness vs Min Energy Gap
scatter_xy(
    df["Min Energy Gap"].values,
    df["Hardness"].values,
    xlab="Min Energy Gap",
    ylab="Hardness",
    title="Hardness vs Min Energy Gap",
    outname="scatter_hardness_vs_min_gap.png"
)

# ─────────────────────────────────────────────
# DRAW EXAMPLE GRAPHS FOR HIGHEST/LOWEST HARDNESS
def draw_graph(edges, N, title, outname):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)

    plt.figure(figsize=(6.5, 5.5))
    pos = nx.spring_layout(G, seed=RANDOM_LAYOUT_SEED)
    nx.draw(G, pos, with_labels=True, node_size=700, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=2)
    plt.title(title)
    plt.tight_layout()
    out_path = OUTPUT_DIR / outname
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

# Sort rows by Hardness
df_sorted = df.copy()
df_sorted["_idx"] = np.arange(len(df_sorted))  # remember original row order
df_sorted["__edges"] = parsed_edges
df_sorted["__N"] = Ns

# Highest hardness examples
top_rows = df_sorted.sort_values("Hardness", ascending=False).head(TOP_K)
for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
    edges = row["__edges"]
    N = int(row["__N"]) if not pd.isna(row["__N"]) else infer_num_qubits(edges)
    title = (f"High Hardness Example #{rank}  |  H={row['Hardness']:.4f}  "
             f"|  Fidelity={row['Fidelity']:.4f}  |  Gap={row['Min Energy Gap']:.6f}")
    outname = f"graph_high_hardness_{rank}.png"
    draw_graph(edges, N, title, outname)

# Lowest hardness examples
low_rows = df_sorted.sort_values("Hardness", ascending=True).head(TOP_K)
for rank, (_, row) in enumerate(low_rows.iterrows(), start=1):
    edges = row["__edges"]
    N = int(row["__N"]) if not pd.isna(row["__N"]) else infer_num_qubits(edges)
    title = (f"Low Hardness Example #{rank}  |  H={row['Hardness']:.4f}  "
             f"|  Fidelity={row['Fidelity']:.4f}  |  Gap={row['Min Energy Gap']:.6f}")
    outname = f"graph_low_hardness_{rank}.png"
    draw_graph(edges, N, title, outname)

print("All figures saved next to the Excel file.")
