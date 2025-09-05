
---

### **`ML/graph_classification/README.md`**

```markdown
# Graph Classification Models

This directory contains implementations of GNN models for the task of graph classification. The objective is to predict a single categorical label for each graph in a dataset.

## Dataset

The models are trained and evaluated on the **MUTAG** dataset[cite: 1, 21, 25]. This dataset consists of graphs representing chemical compounds, where the task is to predict their mutagenic effect. A `DataLoader` is used to handle batching of multiple graphs during training and testing[cite: 1, 21, 25].

## Methodology

Each model follows a similar structure:
1.  Multiple graph convolution layers process the node features and graph structure to generate node embeddings[cite: 1, 22, 26].
2.  A `global_mean_pool` layer aggregates the node embeddings into a single graph-level embedding[cite: 2, 22, 26].
3.  A final linear layer, along with dropout for regularization, classifies the graph embedding into one of the output classes[cite: 1, 2, 22, 26].

## Models

* `gcn.py`: Implements a Graph Convolutional Network (GCN) for graph classification[cite: 21, 22].
* `gat.py`: Implements a Graph Attention Network (GAT) for graph classification[cite: 25, 26].
* `graphsage.py`: Implements a GraphSAGE model for graph classification[cite: 1, 2].

## How to Run

To train and evaluate a model, run its corresponding Python script from this directory. The script will print the training loss and accuracy at regular intervals[cite: 4, 24, 28].

```bash
# Run the GCN model
python gcn.py

# Run the GAT model
python gat.py

# Run the GraphSAGE model
python graphsage.py