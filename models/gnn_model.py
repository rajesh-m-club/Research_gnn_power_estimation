import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class ToggleGNN(nn.Module):

    def __init__(self, in_channels=35, hidden_channels=64, dropout=0.2):

        super(ToggleGNN, self).__init__()

        # GraphSAGE layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # Final regression layer
        self.fc = nn.Linear(hidden_channels, 1)

        self.dropout = dropout

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index

        # First GraphSAGE layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GraphSAGE layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Output layer
        x = self.fc(x)

        # Return shape [num_nodes]
        return x.squeeze()