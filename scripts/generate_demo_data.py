import numpy as np
import os

# Create directory
os.makedirs('example_data', exist_ok=True)

# 1. Gene names (50 genes)
genes = [f"GENE_{i}" for i in range(50)]
# Include a few TFs for the TF-centric logic
genes[0] = "SOX2"
genes[10] = "POU5F1"
genes[20] = "NANOG"

with open('example_data/filtered_genes.txt', 'w') as f:
    f.write('\n'.join(genes))

# 2. Sample names (20 samples)
samples = [f"Sample_{i}" for i in range(20)]
with open('example_data/sample_names.txt', 'w') as f:
    f.write('\n'.join(samples))

# 3. Expression data (20 samples x 50 genes)
# Create some correlations so the pipeline actually finds edges
data = np.random.randn(20, 50).astype(np.float32)
# Make SOX2 regulate some genes
data[:, 1] = 0.8 * data[:, 0] + 0.2 * np.random.randn(20)
data[:, 2] = -0.7 * data[:, 0] + 0.3 * np.random.randn(20)

np.save('example_data/normalized_expression.npy', data)

# 4. Tiny PPI file
with open('example_data/string_ppi_tiny.txt', 'w') as f:
    f.write("GeneA\tGeneB\tcombined_score\n")
    f.write("SOX2\tGENE_1\t999\n")
    f.write("SOX2\tGENE_2\t999\n")
    f.write("POU5F1\tNANOG\t900\n")

print("Example data generated in example_data/")
