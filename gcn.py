import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Sample list of tensors (node features)

tensor_zeros = torch.zeros(1, 256)

# Print the size and content of the tensor
print(tensor_zeros.shape)

num_nodes = 2
feature_dim = 256
node_features_list = [torch.randn(feature_dim) for _ in range(num_nodes)]

# Create a list of edges connecting each node to the next node
edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes - 1)], dtype=torch.long).t().contiguous()

# Stack the list of tensors into a single tensor
node_features = torch.stack(node_features_list, dim=0)

# Create a Data object
data = Data(x=node_features, edge_index=edge_index)

# Define and initialize the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN(feature_dim, 16, 256)  # Adjust the input and output channels as per your dataset

# Pass the data through the GCN model
output = model(data)

output = output.mean(dim=0, keepdim=True)  # Take the mean along dimension 0 and keep it as a single row

# Now, output will be of shape [1, 7]
print(output.shape)
