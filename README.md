# Causal GRN Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-High_Performance-red)
![License](https://img.shields.io/badge/License-Proprietary-critical)
![Profile Views](https://komarev.com/ghpvc/?username=shreyakat0ch&label=Profile+Views&color=blue&style=flat)
![Repo Views](https://visitor-badge.laobi.icu/badge?page_id=shreyakat0ch.causal_grn_pipeline)

A genome-wide causal discovery pipeline for building gene regulatory networks (GRNs) from RNA-seq TPM data. The pipeline outputs a directed acyclic graph (DAG) of regulatory interactions, annotated with transcription factors and edge signs (activation vs repression).

Designed to scale to ~14,000 genes on standard research hardware.

---

## Background

Most GRN inference methods either rely on simple correlation (which is not causal) or don't scale to genome-wide data. This pipeline combines several techniques to get around those limitations:

- **Knockoff statistics** for FDR-controlled variable selection
- **Random Forest regressions** to capture nonlinear regulatory relationships
- **STRING PPI** as a biological prior to narrow the search space
- **GraphSAGE** with gradient attribution to filter out indirect/spurious associations
- **Degenerate Gaussian scoring** to orient edges and determine causal direction

The end result is a DAG where each edge represents a likely direct regulatory relationship, with direction and sign.

---

## Project layout

```
causal_grn10/
├── causal_pipeline_v1.py       # main pipeline (runs all 5 stages)
├── validate_network.py         # validates output against TRRUST and SIGNOR
├── visualize_network.py        # generates an interactive D3.js network viewer
│
├── causal/
│   ├── __init__.py
│   └── DegenerateGaussianScore.py   # scoring module for causal orientation
│
├── data/
│   ├── normalized_expression.npy    # RNA-seq expression matrix
│   ├── filtered_genes.txt
│   ├── sample_names.txt
│   ├── jaspar_tfs.txt               # downloaded automatically if missing
│   └── knockoff_matrix_14k.tsv      # generated and cached on first run
│
├── priors/
│   └── STRING_PPI_filtered.txt      # STRING PPI (confidence >= 700)
│
├── validation/
│   ├── trrust_rawdata.human.tsv
│   └── signor_human.tsv
│
├── results/                         # created automatically
│   ├── causal_dag_full.csv
│   ├── causal_dag_tf.csv
│   └── tf_targets.json
│
└── outputs/                         # created automatically
    ├── network_full_d3.html
    └── network_top50_d3.html
```

---

## How to run

### Requirements

Python 3.8+, high-performance hardware is recommended for Stage 4.

```bash
pip install numpy pandas networkx scikit-learn joblib tqdm torch torchvision torch-geometric scipy
```

Install PyTorch with the appropriate support for your system from [pytorch.org](https://pytorch.org/get-started/locally/) before installing `torch-geometric`.

### Running the pipeline

```bash
python causal_pipeline_v1.py
```

The pipeline runs in 5 stages and caches intermediate results (knockoff matrix, RF skeleton) so you don't have to rerun everything from scratch. First run on 14k genes takes several hours depending on your hardware.

### Visualize

```bash
python visualize_network.py
```

Generates two HTML files in `outputs/` — one for the full network and one focused on the top 50 hub genes. Opens in any browser, no server needed.

### Validate

```bash
python validate_network.py
```

Compares predicted edges against TRRUST and SIGNOR and reports precision, AUROC, AUPRC, direction accuracy, and sign accuracy.

---

## Pipeline stages

**Stage 1 — Knockoff generation**
Fits a covariance matrix with LedoitWolf shrinkage and generates second-order Gaussian knockoffs for every gene. These act as negative controls for FDR control downstream.

**Stage 2 — RF skeleton**
Runs a Random Forest regression for each gene against all others. An edge is kept only if gene A predicts gene B *and* gene B predicts gene A (AND rule). This gives an undirected skeleton.

**Stage 3 — PPI filter**
Intersects the skeleton with STRING PPI pairs (combined score >= 700) to remove biologically implausible interactions and reduce noise.

**Stage 4 — GNN knockoff selection**
Trains a GraphSAGE model on the filtered skeleton. Gradient attribution is used to compute a knockoff statistic W = ||grad_original|| - ||grad_knockoff|| for each neighbor. Edges are kept at FDR q=0.10.

**Stage 5 — Orientation and DAG enforcement**
Uses the Degenerate Gaussian score to determine edge direction. Remaining cycles are broken by removing the weakest edge. TFs are annotated using JASPAR and HOCOMOCO.

---

## Output format

`results/causal_dag_full.csv` columns:

| Column | Description |
|--------|-------------|
| Cause | source gene |
| Effect | target gene |
| EffectSize | DG score difference (edge strength) |
| Regulation | Positive (Activation) or Negative (Repression) |
| cause_is_TF | whether source is a known TF |
| effect_is_TF | whether target is a known TF |

---

## Config

All main hyperparameters are at the top of `causal_pipeline_v1.py`:

```python
PPI_THRESHOLD = 700          # STRING confidence cutoff
RF_N_ESTIMATORS = 300        # trees per forest
RF_IMPORTANCE_THRESHOLD = 0.0005
RF_N_JOBS = 64               # parallel workers
GNN_EPOCHS = 150
GNN_HIDDEN_DIM = 256
FDR_Q = 0.10                 # target FDR
DISCRETE_THRESHOLD = 0.01
```

---

## External data sources

- [STRING](https://string-db.org/) — PPI prior
- [JASPAR](https://jaspar.elixir.no/) — TF list (fetched automatically)
- [HOCOMOCO](https://hocomoco12.autosome.org/) — additional TF annotations
- [TRRUST](https://www.grnpedia.org/trrust/) — validation ground truth
- [SIGNOR](https://signor.uniroma2.it/) — validation ground truth

---

## References

If you use this pipeline, please cite the following foundational works:

- **Knockoffs**: Candès, E. J., Fan, J., Janson, L., & Lv, J. (2018). Panning for Gold: Model-X Knockoffs for High-Dimensional Controlled Variable Selection. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*.
- **GraphSAGE**: Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *Advances in Neural Information Processing Systems (NeurIPS)*.
- **DG Score**: Andrews, B., Ramsey, J., & Cooper, G. F. (2019). Learning High-dimensional Directed Acyclic Graphs with Mixed Data-Types. *Proceedings of the 35th Conference on Uncertainty in Artificial Intelligence (UAI)*.
- **STRING Database**: Szklarczyk, D. et al. (2021). The STRING database in 2021: customizable protein-protein networks, and functional characterization of user-uploaded gene/measurement sets. *Nucleic Acids Research*.
- **JASPAR**: Castro-Mondragon, J. A. et al. (2022). JASPAR 2022: the 9th release of the open-access database of transcription factor binding profiles. *Nucleic Acids Research*.

## License

See [LICENSE](LICENSE). Viewing and citing is permitted. Modification, redistribution, and commercial use are not.
