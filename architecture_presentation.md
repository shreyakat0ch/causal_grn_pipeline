# 🧬 Genome-Wide Causal GRN Architecture
## High-Performance GPU-Accelerated Causal Discovery

### 1. Overview
This architecture implements a multi-stage discovery pipeline to infer **Directed Acyclic Graphs (DAGs)** from RNA-seq TPM data. It combines nonlinear dependency capture, Graph Neural Networks (GNNs), and rigorous causal orientation.

---

### 2. The 5-Stage Pipeline

#### **Stage 1: Stable Knockoff Generation**
*   **Technique**: Second-order Gaussian Knockoffs via **LedoitWolf Shrinkage**.
*   **Purpose**: Creates "synthetic twins" of every gene that preserve the correlation structure but are independent of the causal mechanism. This allows for rigorous **False Discovery Rate (FDR)** control.

#### **Stage 2: Nonlinear Skeleton Discovery**
*   **Technique**: High-dimensional **Random Forest** Regressions.
*   **Purpose**: Captures nonlinear regulatory relationships that traditional linear models (like PC or Glasso) miss.
*   **Constraint**: Uses an "AND-rule" to ensure mutual predictive importance between gene pairs.

#### **Stage 3: Prior Knowledge Integration**
*   **Technique**: **STRING PPI** (Protein-Protein Interaction) Filtering.
*   **Purpose**: Restricts the search space to biologically plausible interactions (Confidence Score ≥ 700), significantly reducing computational noise.

#### **Stage 4: GNN-Based Knockoff Selection**
*   **Technique**: **GraphSAGE** with Gradient Attribution ($W_{ij} = \|\nabla_{orig}\|_2 - \|\nabla_{ko}\|_2$).
*   **Purpose**: Uses a Deep Learning model to predict gene expression based on neighbors. By comparing gradients of original genes vs. their knockoffs, we filter out redundant/indirect associations.
*   **Advantage**: Fully **GPU-accelerated** for genome-wide scalability.

#### **Stage 5: Causal Orientation & DAG Enforcement**
*   **Technique**: **Degenerate Gaussian (DG) Score**.
*   **Purpose**: Determines the direction of the arrow ($A \rightarrow B$ vs $B \rightarrow A$) by comparing local log-likelihood scores.
*   **DAG Enforcement**: Automatically removes remaining cycles by pruning the weakest edges.

---

### 3. Key Innovations
1.  **Nonlinear Discovery**: Moves beyond simple correlation to capture complex TF-Target kinetics.
2.  **Scalability**: Optimized to handle **14,000+ genes** across hundreds of samples on a single GPU.
3.  **TF-Centric Output**: Specifically annotates Transcription Factors (via JASPAR/HOCOMOCO) for downstream Binding and Histone Mark modeling.
4.  **Portable Design**: Entirely self-contained folder with relative paths and internal causal modules.

---

### 4. Interactive Visualization Suite
The architecture produces a **D3.js-based Visualizer** featuring:
*   **Force-Directed Layout**: Dynamic settling of 16K+ edges.
*   **Hub Identification**: Visual emphasis on "Master Regulators" (Top 50 hubs).
*   **Functional Toggles**: Instant filtering of **Activation** (Positive) vs. **Repression** (Negative) edges.
*   **Search & Highlight**: Real-time gene search and regulatory neighborhood highlighting.

---

### 5. Summary of Outputs
*   📄 `causal_dag_full.csv`: The complete genome-wide regulatory map.
*   📄 `causal_dag_tf.csv`: A focused subset of TF-to-Target interactions.
*   📦 `tf_targets.json`: Ready-to-use lists for TF binding influence models.
*   🌐 `network_top50_hubs.html`: Premium interactive dashboard.
