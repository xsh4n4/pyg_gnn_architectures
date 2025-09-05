
---

### **`ML/link_prediction/README.md`**

```markdown
# Link Prediction Models

This directory contains GNN models for the task of link prediction. The goal is to predict whether an edge is likely to exist between any two nodes in a graph. This is framed as a binary classification problem on edges (positive for existing, negative for non-existing).

## Dataset

The models are built using the **Cora** dataset[cite: 5, 29, 33]. The dataset's edges are split into training, validation, and test sets using PyG's `RandomLinkSplit` transform. This transform also generates negative samples (non-existent edges) for training[cite: 5, 29, 33].

## Methodology

The models use an encoder-decoder architecture:
1.  **Encoder**: A GNN (GCN, GAT, or GraphSAGE) acts as an encoder to generate rich node embeddings for all nodes in the graph[cite: 6, 30, 34].
2.  **Decoder**: A `LinkPredictor` module takes the embeddings of a pair of nodes and computes their dot product to produce a score indicating the likelihood of a link between them[cite: 7, 31, 35].

The model is trained using a `BCEWithLogitsLoss` criterion[cite: 7, 31].

## Evaluation

The model's performance is evaluated using the **Area Under the Receiver Operating Characteristic Curve (ROC AUC)** score on the validation and test sets[cite: 8, 32, 36].

## Models

* `gcn.py`: Uses a `GCNEncoder` to generate node embeddings[cite: 29, 30].
* `gat.py`: Uses a `GATEncoder` to generate node embeddings[cite: 33, 34].
* `graphsage.py`: Uses a `GraphSAGEEncoder` to generate node embeddings[cite: 5, 6].

## How to Run

To train and evaluate a model, run its script. The training loss, validation AUC, and test AUC will be printed periodically[cite: 8, 36].

```bash
# Run the GCN model
python gcn.py

# Run the GAT model
python gat.py

# Run the GraphSAGE model
python graphsage.py