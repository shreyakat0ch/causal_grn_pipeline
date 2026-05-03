#!/usr/bin/env python3
"""
validate_network.py
Validates predicted DAG against TRRUST and SIGNOR.
"""

import os
import argparse
import urllib.request
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = BASE_DIR
DAG_FILE    = os.path.join(OUTPUT_DIR, "results", "causal_dag_full.csv")
TRRUST_FILE = os.path.join(BASE_DIR, "validation", "trrust_rawdata.human.tsv")
SIGNOR_FILE = os.path.join(BASE_DIR, "validation", "signor_human.tsv")
KEGG_CACHE  = os.path.join(OUTPUT_DIR, "data", "kegg_pairs.txt")
REACT_CACHE = os.path.join(OUTPUT_DIR, "data", "reactome_pairs.txt")

def validate_trrust(df: pd.DataFrame):
    print("\n" + "="*50)
    print("TRRUST Validation (TF -> Target)")
    print("="*50)
    
    if not os.path.exists(TRRUST_FILE):
        print(f"TRRUST file not found: {TRRUST_FILE}")
        return
        
    trrust = pd.read_csv(TRRUST_FILE, sep='\t', names=['TF','Target','Direction','PMID'], header=None)
    trrust_dir = set(zip(trrust['TF'], trrust['Target']))
    trrust_undir = trrust_dir | set(zip(trrust['Target'], trrust['TF']))
    
    print(f"  Loaded {len(trrust_dir)} directed TRRUST edges")
    
    # 1. Precision at threshold
    print("\n  Precision at different thresholds:")
    for thr in [0.9, 0.7, 0.5, 0.3, 0.1]:
        top = df[df['EffectSize'].abs() >= thr]
        if len(top) == 0: continue
        pred_set = set(zip(top['Cause'], top['Effect']))
        overlap = pred_set & trrust_undir
        print(f"    |w|>={thr:.1f}: {len(pred_set):5d} edges | {len(overlap):3d} known | Precision={len(overlap)/len(pred_set):.4f}")
        
    # 2. AUROC/AUPRC
    labels_u, labels_d, scores = [], [], []
    for _, row in df.iterrows():
        s, t = row['Cause'], row['Effect']
        labels_u.append(1 if (s,t) in trrust_undir else 0)
        labels_d.append(1 if (s,t) in trrust_dir else 0)
        scores.append(abs(row['EffectSize']))
        
    pos_u, pos_d = sum(labels_u), sum(labels_d)
    
    if pos_u > 0 and pos_u < len(labels_u):
        print(f"\n  Undirected AUROC: {roc_auc_score(labels_u, scores):.4f} (random=0.5)")
        print(f"  Undirected AUPRC: {average_precision_score(labels_u, scores):.4f} (random={pos_u/len(labels_u):.4f})")
    
    if pos_d > 0 and pos_d < len(labels_d):
        print(f"  Directed AUROC  : {roc_auc_score(labels_d, scores):.4f}")
        print(f"  Directed AUPRC  : {average_precision_score(labels_d, scores):.4f}")
        
    # 3. Direction accuracy
    corr_dir, wrong_dir = 0, 0
    for _, row in df.iterrows():
        s, t = row['Cause'], row['Effect']
        if (s, t) in trrust_dir:
            corr_dir += 1
        elif (t, s) in trrust_dir:
            wrong_dir += 1
            
    tot_dir = corr_dir + wrong_dir
    if tot_dir > 0:
        print(f"\n  Direction Accuracy: {corr_dir}/{tot_dir} ({corr_dir/tot_dir*100:.1f}%)")
        
    # 4. Sign accuracy
    sign_map = {(r['TF'], r['Target']): r['Direction'] for _, r in trrust.iterrows()}
    corr_sign, wrong_sign = 0, 0
    for _, row in df.iterrows():
        s, t = row['Cause'], row['Effect']
        known = sign_map.get((s, t))
        if known:
            pred = 'Activation' if 'Positive' in row['Regulation'] else 'Repression'
            if pred == known: corr_sign += 1
            else: wrong_sign += 1
            
    tot_sign = corr_sign + wrong_sign
    if tot_sign > 0:
        print(f"  Sign Accuracy     : {corr_sign}/{tot_sign} ({corr_sign/tot_sign*100:.1f}%)")

def validate_signor(df: pd.DataFrame):
    print("\n" + "="*50)
    print("SIGNOR Validation (Signalling)")
    print("="*50)
    
    if not os.path.exists(SIGNOR_FILE):
        print(f"SIGNOR file not found: {SIGNOR_FILE}")
        return
        
    sig = pd.read_csv(SIGNOR_FILE, sep='\t')
    if 'ENTITYA' in sig.columns and 'ENTITYB' in sig.columns:
        sig_dir = set(zip(sig['ENTITYA'], sig['ENTITYB']))
        sig_undir = sig_dir | set(zip(sig['ENTITYB'], sig['ENTITYA']))
        
        pred_set = set(zip(df['Cause'], df['Effect']))
        overlap = pred_set & sig_undir
        overlap_dir = pred_set & sig_dir
        
        print(f"  Loaded {len(sig_dir)} directed SIGNOR edges")
        print(f"  Predicted edges in SIGNOR (undir) : {len(overlap)} / {len(pred_set)}")
        print(f"  Predicted edges in SIGNOR (dir)   : {len(overlap_dir)} / {len(pred_set)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default=DAG_FILE)
    args = parser.parse_args()
    
    if not os.path.exists(args.network):
        print(f"Network file not found: {args.network}")
        print("Please run causal_pipeline_v1.py first.")
        return
        
    df = pd.read_csv(args.network)
    print(f"Validating {len(df):,} edges from {args.network}")
    
    validate_trrust(df)
    validate_signor(df)
    
    print("\nValidation complete.")

if __name__ == "__main__":
    main()
