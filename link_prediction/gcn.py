import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

transform = T.RandomLinkSplit(is_undirected=True, add_negative_train_samples=True)
train_data, val_data, test_data = transform(data)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class LinkPredictor(torch.nn.Module):
    def forward(self, z, edge_label_index):
        edge_feat_u = z[edge_label_index[0]]
        edge_feat_v = z[edge_label_index[1]]
        # Dot product as the decoder
        return (edge_feat_u * edge_feat_v).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = LinkPredictor()

    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z, edge_label_index)

# --- Model Initialization ---
model = Model(dataset.num_node_features, 128, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

# --- Training Loop ---
def train():
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index, train_data.edge_label_index)
    loss = criterion(out, train_data.edge_label)
    loss.backward()
    optimizer.step()
    return float(loss)

# --- Testing ---
@torch.no_grad()

    test_auc = test(test_data)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')