# gnn_model2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class ToggleGNNv2(nn.Module):
    def __init__(self, 
                 num_logic_families=26,   # number of logic families in your cell library
                 embed_dim=8,             # embedding dimension for cell types
                 in_channels=35,          # total feature vector length from FeatureBuilder
                 hidden_channels=64,
                 dropout=0.2):
        super(ToggleGNNv2, self).__init__()

        self.num_logic_families = num_logic_families
        self.embed_dim = embed_dim
        self.dropout = dropout

        # Linear layer to convert logic family one-hot to learnable embedding
        # Assuming logic family one-hot starts at index 6 in feature vector
        self.cell_type_embed = nn.Linear(num_logic_families, embed_dim)

        # LayerNorm for normalized embeddings
        self.cell_type_ln = nn.LayerNorm(embed_dim)

        # Compute new in_channels after embedding replacement
        # base 6 + embed_dim + rest (fanin/fanout/level/toggle)
        self.feat_start_idx = 6
        self.feat_end_idx = 6 + num_logic_families
        remaining_feats = in_channels - self.feat_end_idx
        self.new_in_channels = 6 + embed_dim + remaining_feats

        # GraphSAGE layers with residual connections
        self.conv1 = SAGEConv(self.new_in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        # Final regression layer
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        # ------------------------
        # Replace logic family one-hot with normalized embedding
        # ------------------------
        node_type_feat = x[:, :6]
        logic_feat_onehot = x[:, self.feat_start_idx:self.feat_end_idx]
        other_feats = x[:, self.feat_end_idx:]

        logic_embed = self.cell_type_embed(logic_feat_onehot)
        logic_embed = self.cell_type_ln(logic_embed)  # Normalize embedding
        x = torch.cat([node_type_feat, logic_embed, other_feats], dim=1)

        # ------------------------
        # GraphSAGE layers with residuals
        # ------------------------
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = F.relu(self.conv2(h1, edge_index) + h1)  # residual
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        h3 = F.relu(self.conv3(h2, edge_index) + h2)  # residual

        # ------------------------
        # Output regression
        # ------------------------
        out = self.fc(h3)

        return out.squeeze()  # [num_nodes]