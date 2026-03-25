# models/gnn_hetero_conv.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData

class ToggleHeteroConvGNN(nn.Module):
    def __init__(self,
                 net_feat_dim=4,
                 pin_feat_dim=3,
                 cell_feat_dim=32,
                 cell_type_dim=26,
                 hidden_dim=64,
                 num_layers=4,
                 dropout=0.2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.cell_type_dim = cell_type_dim

        # -----------------------------
        # Cell type embedding
        # -----------------------------
        self.cell_type_embed = nn.Linear(cell_type_dim, hidden_dim)

        # -----------------------------
        # Input projections
        # -----------------------------
        self.net_lin = nn.Linear(net_feat_dim, hidden_dim)
        self.pin_in_lin = nn.Linear(pin_feat_dim, hidden_dim)
        self.pin_out_lin = nn.Linear(pin_feat_dim, hidden_dim)
        self.cell_struct_lin = nn.Linear(cell_feat_dim - cell_type_dim, hidden_dim)

        # -----------------------------
        # Cell MLP (structural + type embedding)
        # -----------------------------
        self.cell_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # -----------------------------
        # HeteroConv layers
        # -----------------------------
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('net', 'net_to_pin_in', 'pin_in'): SAGEConv(hidden_dim, hidden_dim),
                ('pin_in', 'pin_in_to_cell', 'cell'): SAGEConv(hidden_dim, hidden_dim),
                ('cell', 'cell_to_pin_out', 'pin_out'): SAGEConv(hidden_dim, hidden_dim),
                ('pin_out', 'pin_out_to_net', 'net'): SAGEConv(hidden_dim, hidden_dim),
            }, aggr='mean')
            self.convs.append(conv)

        # -----------------------------
        # Output layer
        # -----------------------------
        self.out_lin = nn.Linear(hidden_dim, 1)

    def forward(self, data: HeteroData):
        # -----------------------------
        # Initial node projections
        # -----------------------------
        net_x = F.relu(self.net_lin(data['net'].x))
        pin_in_x = F.relu(self.pin_in_lin(data['pin_in'].x))
        pin_out_x = F.relu(self.pin_out_lin(data['pin_out'].x))

        # -----------------------------
        # Cell features split
        # -----------------------------
        cell_feat = data['cell'].x
        cell_type_onehot = cell_feat[:, :self.cell_type_dim]
        cell_struct = cell_feat[:, self.cell_type_dim:]

        cell_struct = F.relu(self.cell_struct_lin(cell_struct))
        cell_type_emb = F.relu(self.cell_type_embed(cell_type_onehot))

        cell_x = torch.cat([cell_struct, cell_type_emb], dim=1)
        cell_x = self.cell_mlp(cell_x)

        # -----------------------------
        # Build node feature dict
        # -----------------------------
        x_dict = {
            'net': net_x,
            'pin_in': pin_in_x,
            'pin_out': pin_out_x,
            'cell': cell_x
        }

        # -----------------------------
        # HeteroConv message passing with residuals
        # -----------------------------
        for conv in self.convs:
            x_new = conv(x_dict, data.edge_index_dict)
            # Residual update + ReLU
            x_dict = {k: F.relu(x_dict[k] + x_new[k]) for k in x_dict}
            # Dropout only for net nodes (as before)
            x_dict['net'] = F.dropout(x_dict['net'], p=self.dropout, training=self.training)

        # -----------------------------
        # Output only for nets
        # -----------------------------
        out = self.out_lin(x_dict['net'])
        return out