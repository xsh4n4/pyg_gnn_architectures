
---

### **`ML/node_classification/README.md`**

```markdown
# Node Classification Models

This directory contains implementations of GNN models for the task of semi-supervised node classification. The goal is to predict the class label of each node in a graph, given the labels of only a small subset of nodes.

## Dataset

All models use the **Cora** dataset, a standard benchmark for this task[cite: 9, 13, 17]. It is a citation network where nodes represent academic papers and edges represent citations. The dataset provides masks (`train_mask`, `val_mask`, `test_mask`) to identify which nodes to use for training, validation, and testing.

## Models

All models are composed of two GNN layers. They take the node features and the graph's adjacency information as input and output a class probability distribution for each node.

* `gcn.py`: A two-layer Graph Convolutional Network (GCN)[cite: 17]. It uses ReLU activation and dropout for regularization[cite: 18].
* `gat.py`: A two-layer Graph Attention Network (GAT) that leverages multi-head attention in its first layer[cite: 13]. It uses ELU as its activation function[cite: 14].
* `graphsage.py`: A two-layer GraphSAGE model that uses 'mean' as its aggregation strategy[cite: 9]. It also uses ReLU and dropout[cite: 10].

## Training and Evaluation

The models are trained by minimizing the cross-entropy loss on the labeled training nodes[cite: 19]. After each training epoch, the model's accuracy is calculated on the training, validation, and test sets[cite: 11, 12, 16, 20].

## How to Run

To run a specific model, execute its Python script from within this directory. The script will output the training loss and the train/validation/test accuracy every 10 epochs[cite: 12].

```bash
# Run the GCN model
python gcn.py

# Run the GAT model
python gat.py

# Run the GraphSAGE model
python graphsage.py