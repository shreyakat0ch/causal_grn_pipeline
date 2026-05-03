#!/usr/bin/env python3
"""
causal_pipeline_v1.py
Builds a causal DAG from RNA-seq TPM data using knockoffs,
random forests, GraphSAGE, and DG scoring.
"""

import sys
import os
import json
import time
import argparse
import urllib.request
import numpy as np
import pandas as pd
import networkx as nx

# Base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor

# --- paths and config ---

# Data paths (Consolidated in the same folder)
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = BASE_DIR
PPI_FILE    = os.path.join(BASE_DIR, "priors", "STRING_PPI_filtered.txt")
TF_MEME     = os.path.join(BASE_DIR, "HOCOMOCO12_human.meme")
JASPAR_TF   = os.path.join(BASE_DIR, "data", "jaspar_tfs.txt")

EXPR_FILE   = os.path.join(DATA_DIR, "normalized_expression.npy")
GENE_FILE   = os.path.join(DATA_DIR, "filtered_genes.txt")
SAMP_FILE   = os.path.join(DATA_DIR, "sample_names.txt")

# Checkpoints/caches
CACHE_DIR   = os.path.join(BASE_DIR, "data")
KNOCKOFF_FILE = os.path.join(CACHE_DIR, "knockoff_matrix_14k.tsv")
RF_CACHE    = os.path.join(CACHE_DIR, "rf_skeleton_14k.json")

# Hyperparameters
PPI_THRESHOLD = 700
RF_N_ESTIMATORS = 300
RF_IMPORTANCE_THRESHOLD = 0.0005
RF_N_JOBS = 64

# GraphSAGE Config
GNN_EPOCHS = 150
GNN_LR = 0.001
GNN_WEIGHT_DECAY = 1e-4
GNN_HIDDEN_DIM = 256
GNN_N_LAYERS = 2
GNN_BATCH_GENES = 500
FDR_Q = 0.10
FDR_OFFSET = 1.0

DISCRETE_THRESHOLD = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- data loading and knockoff generation ---

def load_data():
    os.makedirs(CACHE_DIR, exist_ok=True)
    print("\n[Stage 1] Loading Preprocessed Data...")
    
    genes = [l.strip() for l in open(GENE_FILE)]
    samples = [l.strip() for l in open(SAMP_FILE)]
    expr = np.load(EXPR_FILE)
    
    print(f"  Shape: {expr.shape[0]} samples × {expr.shape[1]} genes")
    
    df = pd.DataFrame(expr, index=samples, columns=genes)
    return df, genes

def generate_knockoffs(df: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(KNOCKOFF_FILE):
        print(f"  Loading knockoffs from: {KNOCKOFF_FILE}")
        return pd.read_csv(KNOCKOFF_FILE, sep="\t", index_col=0)
        
    print(f"\n  Generating knockoffs (LedoitWolf, n={df.shape[0]}, p={df.shape[1]})...")
    X = df.values.astype(np.float64)
    n, p = X.shape
    
    t0 = time.time()
    lw = LedoitWolf(assume_centered=False).fit(X)
    Sigma = lw.covariance_
    print(f"    Covariance fit: {time.time()-t0:.1f}s")
    
    s_val = min(2 * np.linalg.eigvalsh(Sigma).min(), 1.0)
    s_val = max(s_val, 1e-4)
    S = np.diag(np.full(p, s_val))
    
    Sigma_inv = np.linalg.pinv(Sigma)
    Sigma_tilde = 2 * S - S @ Sigma_inv @ S + np.eye(p) * 1e-6
    L_tilde = np.linalg.cholesky(Sigma_tilde)
    
    mu = X.mean(axis=0)
    X_c = X - mu
    X_ko = (X_c - X_c @ Sigma_inv @ S + np.random.randn(n, p) @ L_tilde.T) + mu
    print(f"    Knockoff construction: {time.time()-t0:.1f}s")
    
    combined = np.hstack([X, X_ko]).astype(np.float32)
    ko_cols = [f"{c}_knockoff" for c in df.columns]
    df_ko = pd.DataFrame(combined, index=df.index, columns=df.columns.tolist() + ko_cols)
    
    print(f"  Saving to: {KNOCKOFF_FILE}")
    df_ko.to_csv(KNOCKOFF_FILE, sep="\t")
    return df_ko

# --- RF skeleton ---

def _rf_one_gene(i: int, gene: str, X: np.ndarray, cols: list) -> list:
    y = X[:, i]
    mask = np.ones(X.shape[1], dtype=bool)
    mask[i] = False
    X_rest = X[:, mask]
    rest_cols = [c for j, c in enumerate(cols) if j != i]
    
    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, n_jobs=1, random_state=42)
    rf.fit(X_rest, y)
    
    edges = []
    for j, imp in enumerate(rf.feature_importances_):
        if imp >= RF_IMPORTANCE_THRESHOLD:
            edges.append((gene, rest_cols[j]))
    return edges

def build_rf_skeleton(df: pd.DataFrame) -> set:
    if os.path.exists(RF_CACHE):
        print(f"\n[Stage 2] Loaded RF skeleton from cache: {RF_CACHE}")
        with open(RF_CACHE, "r") as f:
            return set(json.load(f))
            
    print(f"\n[Stage 2] Building RF skeleton (n_jobs={RF_N_JOBS})...")
    X = df.values.astype(np.float32)
    cols = df.columns.tolist()
    
    results = Parallel(n_jobs=RF_N_JOBS)(
        delayed(_rf_one_gene)(i, gene, X, cols) 
        for i, gene in enumerate(tqdm(cols, desc="  RF Regressor"))
    )
    
    # AND rule: edge exists if geneA selects geneB AND geneB selects geneA
    directed = set()
    for edges in results:
        for a, b in edges:
            directed.add((a, b))
            
    skeleton = set()
    for a, b in directed:
        if (b, a) in directed:
            skeleton.add("___".join(sorted([a, b])))
            
    print(f"  Directed edges: {len(directed):,}")
    print(f"  AND-rule pairs: {len(skeleton):,}")
    
    with open(RF_CACHE, "w") as f:
        json.dump(list(skeleton), f)
    return skeleton

# --- PPI filter ---

def load_ppi(genes: list) -> tuple:
    print(f"\n[Stage 3] Loading PPI (threshold={PPI_THRESHOLD})...")
    ppi = pd.read_csv(PPI_FILE, sep="\t")
    ppi = ppi[ppi['combined_score'] >= PPI_THRESHOLD]
    
    gene_set = set(genes)
    ppi_genes = set()
    ppi_pairs = set()
    
    for _, row in ppi.iterrows():
        g1, g2 = row['GeneA'], row['GeneB']
        if g1 in gene_set and g2 in gene_set and g1 != g2:
            ppi_genes.add(g1)
            ppi_genes.add(g2)
            ppi_pairs.add("___".join(sorted([g1, g2])))
            
    print(f"  PPI Genes: {len(ppi_genes):,} | Pairs: {len(ppi_pairs):,}")
    return ppi_genes, ppi_pairs

# --- GraphSAGE knockoff selection ---

class GraphSAGEKnockoff(nn.Module):
    def __init__(self, n_samples: int, hidden_dim: int, n_layers: int):
        super().__init__()
        in_dim = n_samples * 2
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else hidden_dim // 2
            self.convs.append(SAGEConv(in_dim, out_dim))
            in_dim = out_dim
        self.predictor = nn.Linear(hidden_dim // 2, n_samples)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        return self.predictor(x)

def _kfilter(W: np.ndarray, offset: float, q: float) -> float:
    t = np.insert(np.abs(W[W != 0]), 0, 0)
    t = np.sort(t)
    ratio = np.zeros(len(t))
    for i in range(len(t)):
        ratio[i] = (offset + np.sum(W <= -t[i])) / max(1.0, np.sum(W >= t[i]))
    idx = np.where(ratio <= q)[0]
    return float("inf") if len(idx) == 0 else t[idx[0]]

def run_gnn_knockoff(df_ko: pd.DataFrame, skeleton: set, ppi_genes: set) -> set:
    print("\n[Stage 4] GraphSAGE Knockoff Selection...")
    n_samp = df_ko.shape[0]
    p_full = df_ko.shape[1] // 2
    col_list = df_ko.columns.tolist()[:p_full]
    
    nbrs_dict = {}
    for key in skeleton:
        a, b = key.split("___")
        nbrs_dict.setdefault(a, set()).add(b)
        nbrs_dict.setdefault(b, set()).add(a)
        
    target_genes = [c for c in col_list if c in ppi_genes and c in nbrs_dict]
    gene_to_idx = {g: i for i, g in enumerate(target_genes)}
    n_genes = len(target_genes)
    
    print(f"  Target genes: {n_genes:,}")
    
    x_orig_full = df_ko.values[:, :p_full].astype(np.float32)
    x_ko_full   = df_ko.values[:, p_full:].astype(np.float32)
    col_idx_map = {c: i for i, c in enumerate(col_list)}
    
    x_orig = np.zeros((n_genes, n_samp), dtype=np.float32)
    x_ko   = np.zeros((n_genes, n_samp), dtype=np.float32)
    
    for local_i, gene in enumerate(target_genes):
        global_i = col_idx_map[gene]
        x_orig[local_i] = x_orig_full[:, global_i]
        x_ko[local_i]   = x_ko_full[:, global_i]
        
    mu = x_orig.mean(1, keepdims=True)
    std = x_orig.std(1, keepdims=True) + 1e-8
    x_orig = (x_orig - mu) / std
    x_ko   = (x_ko   - mu) / std
    
    node_feats = np.concatenate([x_orig, x_ko], axis=1)
    
    edge_src, edge_dst = [], []
    for key in skeleton:
        a, b = key.split("___")
        if a in gene_to_idx and b in gene_to_idx:
            ia, ib = gene_to_idx[a], gene_to_idx[b]
            edge_src += [ia, ib]
            edge_dst += [ib, ia]
            
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(DEVICE)
    x_tensor   = torch.tensor(node_feats, dtype=torch.float32).to(DEVICE)
    y_tensor   = torch.tensor(x_orig, dtype=torch.float32).to(DEVICE)
    
    model = GraphSAGEKnockoff(n_samp, GNN_HIDDEN_DIM, GNN_N_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=GNN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, GNN_EPOCHS, eta_min=1e-5)
    
    n_val = max(1, int(n_genes * 0.2))
    val_idx = torch.arange(0, n_val, device=DEVICE)
    train_idx = torch.arange(n_val, n_genes, device=DEVICE)
    
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    patience = 20
    
    for epoch in range(GNN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(x_tensor, edge_index)
        train_loss = F.mse_loss(out[train_idx], y_tensor[train_idx])
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            out_val = model(x_tensor, edge_index)
            val_loss = F.mse_loss(out_val[val_idx], y_tensor[val_idx]).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break
            
    model.load_state_dict(best_state)
    model.eval()
    
    print("\n  Computing knockoff attributions...")
    assoc_set = set()
    n_batches = (n_genes + GNN_BATCH_GENES - 1) // GNN_BATCH_GENES
    
    for batch_idx in tqdm(range(n_batches), desc="  Attribution"):
        start = batch_idx * GNN_BATCH_GENES
        end = min(start + GNN_BATCH_GENES, n_genes)
        
        x_b = x_tensor.clone().detach().requires_grad_(True)
        out_b = model(x_b, edge_index)
        
        batch_loss = F.mse_loss(out_b[start:end], y_tensor[start:end])
        batch_loss.backward()
        
        grads = x_b.grad.cpu().numpy()
        
        for local_i in range(start, end):
            gene_i = target_genes[local_i]
            nbrs_i = [g for g in nbrs_dict.get(gene_i, set()) if g in gene_to_idx]
            
            W_stats, nbr_names = [], []
            for nbr in nbrs_i:
                j = gene_to_idx[nbr]
                grad_j = grads[j]
                w_ij = float(np.linalg.norm(grad_j[:n_samp])) - float(np.linalg.norm(grad_j[n_samp:]))
                W_stats.append(w_ij)
                nbr_names.append(nbr)
                
            if W_stats:
                threshold = _kfilter(np.array(W_stats), offset=FDR_OFFSET, q=FDR_Q)
                for nbr, w in zip(nbr_names, W_stats):
                    if w > threshold:
                        assoc_set.add("___".join(sorted([gene_i, nbr])))
                        
        del x_b, out_b, batch_loss, grads
        torch.cuda.empty_cache()
        
    print(f"  GNN Associations: {len(assoc_set):,}")
    return assoc_set

# --- edge orientation ---

def load_tfs():
    tf_set = set()
    # 1. JASPAR
    if os.path.exists(JASPAR_TF):
        with open(JASPAR_TF) as f:
            tf_set.update(l.strip().upper() for l in f if l.strip())
        print(f"  Loaded {len(tf_set)} TFs from JASPAR cache")
    else:
        print("  Fetching TFs from JASPAR REST API...")
        try:
            url = "https://jaspar.elixir.no/api/v1/matrix/?species=9606&format=json&page_size=1000"
            while url:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                for entry in data.get("results", []):
                    name = entry.get("name", "").strip().upper()
                    if name: tf_set.add(name)
                url = data.get("next")
            os.makedirs(os.path.dirname(JASPAR_TF), exist_ok=True)
            with open(JASPAR_TF, "w") as f:
                f.write("\n".join(sorted(tf_set)))
            print(f"  Downloaded {len(tf_set)} TFs from JASPAR")
        except Exception as e:
            print(f"  WARNING: JASPAR API failed ({e})")
            
    # 2. HOCOMOCO (if not empty)
    if os.path.exists(TF_MEME) and os.path.getsize(TF_MEME) > 0:
        with open(TF_MEME) as f:
            for line in f:
                if line.startswith("MOTIF"):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        tf_name = parts[2].split("_")[0].upper()
                        tf_set.add(tf_name)
    return tf_set

def orient_edges(df: pd.DataFrame, final_edges: set, tf_set: set) -> nx.DiGraph:
    print(f"\n[Stage 5] Orienting edges (DG Score)...")
    from causal.DegenerateGaussianScore import DegenerateGaussianScore
    
    edge_genes = set()
    for key in final_edges:
        a, b = key.split("___")
        edge_genes.add(a)
        edge_genes.add(b)
        
    cols = [c for c in df.columns if c in edge_genes]
    sub = df[cols]
    
    print(f"  Computing inv_cov on {len(cols)}x{len(cols)} subset...")
    inv_np = np.linalg.pinv(sub.cov().values.astype(np.float64))
    corr_inv = pd.DataFrame(inv_np, index=cols, columns=cols)
    
    dg = DegenerateGaussianScore(df, discrete_threshold=DISCRETE_THRESHOLD)
    col_map = {col: i for i, col in enumerate(df.columns)}
    
    graph = nx.DiGraph()
    for key in tqdm(final_edges, desc="  Orienting"):
        f1, f2 = key.split("___")
        if abs(float(corr_inv.loc[f1, f2])) <= 0.0:
            continue
            
        s1 = dg.localScore(col_map[f1], {col_map[f2]})
        s2 = dg.localScore(col_map[f2], {col_map[f1]})
        
        # partial correlation for sign
        pcorr = -corr_inv.loc[f1, f2] / np.sqrt(corr_inv.loc[f1, f1] * corr_inv.loc[f2, f2])
        reg = "Positive (Activation)" if pcorr > 0 else "Negative (Repression)"
        
        if s1 < s2:
            graph.add_edge(f2, f1, weight=s2-s1, regulation=reg)
        elif s1 > s2:
            graph.add_edge(f1, f2, weight=s1-s2, regulation=reg)
            
    print("\n[Stage 6] Enforcing DAG...")
    cycles = list(nx.simple_cycles(graph))
    removed = 0
    while cycles:
        cycle = cycles[0]
        weak = min(
            ((cycle[i], cycle[(i+1)%len(cycle)]) for i in range(len(cycle))),
            key=lambda e: graph[e[0]][e[1]]["weight"]
        )
        graph.remove_edge(*weak)
        removed += 1
        cycles = list(nx.simple_cycles(graph))
        
    print(f"  Removed {removed} cycle edges.")
    
    # Annotate TFs
    for u, v in graph.edges():
        graph[u][v]["cause_is_TF"] = u in tf_set
        graph[u][v]["effect_is_TF"] = v in tf_set
        
    return graph

# --- main ---

def main():
    print("="*60)
    print("Causal GRN Pipeline v1")
    print("="*60)
    
    df, genes = load_data()
    tf_set = load_tfs()
    
    df_ko = generate_knockoffs(df)
    skeleton = build_rf_skeleton(df)
    
    ppi_genes, ppi_pairs = load_ppi(genes)
    
    # GNN filtering
    gnn_assoc = run_gnn_knockoff(df_ko, skeleton, ppi_genes)
    
    final_edges = skeleton & gnn_assoc
    print(f"\n  RF ∩ GNN Intersection: {len(final_edges):,} edges")
    
    graph = orient_edges(df, final_edges, tf_set)
    
    print("\n[Stage 7] Saving Outputs...")
    os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)
    
    # 1. Full DAG
    full_rows = []
    for u, v, data in graph.edges(data=True):
        full_rows.append({
            "Cause": u, "Effect": v, 
            "EffectSize": data["weight"], "Regulation": data["regulation"],
            "cause_is_TF": data["cause_is_TF"], "effect_is_TF": data["effect_is_TF"]
        })
    df_full = pd.DataFrame(full_rows)
    out_full = os.path.join(OUTPUT_DIR, "results", "causal_dag_full.csv")
    df_full.to_csv(out_full, index=False)
    print(f"  Saved full DAG: {out_full} ({len(df_full):,} edges)")
    
    # 2. TF-centric DAG
    df_tf = df_full[df_full["cause_is_TF"]].copy()
    out_tf = os.path.join(OUTPUT_DIR, "results", "causal_dag_tf.csv")
    df_tf.to_csv(out_tf, index=False)
    print(f"  Saved TF DAG  : {out_tf} ({len(df_tf):,} edges)")
    
    # 3. TF regulatory profiles JSON
    tf_targets = {}
    for _, row in df_tf.iterrows():
        tf = row["Cause"]
        if tf not in tf_targets:
            tf_targets[tf] = {"activation": [], "repression": []}
        if "Positive" in row["Regulation"]:
            tf_targets[tf]["activation"].append(row["Effect"])
        else:
            tf_targets[tf]["repression"].append(row["Effect"])
            
    out_json = os.path.join(OUTPUT_DIR, "results", "tf_targets.json")
    with open(out_json, "w") as f:
        json.dump(tf_targets, f, indent=2)
    print(f"  Saved TF targets: {out_json}")
    
    print("="*60)
    print("Done.")

if __name__ == "__main__":
    main()
