# Graph Neural Network Implementations in PyTorch Geometric

This repository provides implementations of several common Graph Neural Network (GNN) models for various graph-based machine learning tasks. The models are built using PyTorch and the PyTorch Geometric (PyG) library.

## Project Structure

The repository is organized into directories based on the machine learning task:

-   `graph_classification/`: Contains models for classifying entire graphs. The goal is to assign a label to each graph in a dataset.
-   `node_classification/`: Contains models for classifying individual nodes within a single graph.
-   `link_prediction/`: Contains models for predicting the existence of an edge between two nodes in a graph.

## Models Implemented

Across the different tasks, the following GNN architectures are implemented:
* [cite_start]**GCN (Graph Convolutional Network)** [cite: 17, 21, 29]
* [cite_start]**GAT (Graph Attention Network)** [cite: 13, 25, 33]
* [cite_start]**GraphSAGE (Graph Sample and Aggregated)** [cite: 1, 5, 9]

## Datasets

The scripts automatically download and use standard benchmark datasets:
* [cite_start]**MUTAG**: A dataset of chemical compounds, used for the graph classification task[cite: 1, 21, 25].
* [cite_start]**Cora**: A citation network dataset, used for node classification and link prediction tasks[cite: 5, 9, 13, 17, 29, 33].

## Dependencies

To run these scripts, you will need Python and the following libraries:
* `torch`
* `torch_geometric`
* `scikit-learn`

You can install them via pip:
```bash
pip install torch torch_geometric scikit-learn