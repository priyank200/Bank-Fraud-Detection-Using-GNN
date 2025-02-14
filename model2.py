#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class EdgeEnhancedGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, edge_dim=3):
        super(EdgeEnhancedGAT, self).__init__()
        
        # GATConv layer 1: takes node features and edge features (edge_dim)
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6, edge_dim=edge_dim)
        
        # GATConv layer 2: takes output from the first GAT layer and transforms the features further
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6, edge_dim=edge_dim)

        # MLP to process edge features before applying GATConv
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),  # Transform edge features to edge_dim
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)  # This could be made more complex
        )

    def forward(self, x, edge_index, edge_feat):
        # Transform edge features using the MLP
        edge_feat = self.edge_mlp(edge_feat)

        # Apply dropout and GATConv layer 1
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_feat))
        
        # Apply dropout and GATConv layer 2
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_feat)
        
        return x

# Example usage
# Initialize the model
# model = EdgeEnhancedGAT(in_channels=7, hidden_channels=16, out_channels=2, heads=3, edge_dim=3)
# print(model)

# # %%

# %%
