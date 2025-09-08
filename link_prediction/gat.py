import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score

# Load dataset and prepare for link prediction
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
transform = T.RandomLinkSplit(is_undirected=True, add_negative_train_samples=True)
train_data, val_data, test_data = transform(data)

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class LinkPredictor(torch.nn.Module):
    def forward(self, z, edge_label_index):
        edge_feat_u = z[edge_label_index[0]]
        edge_feat_v = z[edge_label_index[1]]
        return (edge_feat_u * edge_feat_v).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = LinkPredictor()

    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z, edge_label_index)

# --- Model Initialization and Training/Testing ---
model = Model(dataset.num_node_features, 128, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index, train_data.edge_label_index)
    loss = criterion(out, train_data.edge_label)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

for epoch in range(1, 101):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')