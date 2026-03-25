import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData


class ToggleHeteroGNN(nn.Module):

    def __init__(self,
                 net_feat_dim=4,
                 pin_feat_dim=3,
                 cell_feat_dim=32,   # will adjust automatically
                 cell_type_dim=26,   # MUST match cell_library
                 embed_dim=8,
                 hidden_dim=64,
                 num_layers=4,
                 dropout=0.2):

        super(ToggleHeteroGNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.cell_type_dim = cell_type_dim

        # -----------------------------
        # Cell type embedding
        # -----------------------------
        self.cell_type_embed = nn.Linear(cell_type_dim, embed_dim)

        # -----------------------------
        # Input projections
        # -----------------------------
        self.net_lin = nn.Linear(net_feat_dim, hidden_dim)
        self.pin_in_lin = nn.Linear(pin_feat_dim, hidden_dim)
        self.pin_out_lin = nn.Linear(pin_feat_dim, hidden_dim)

        # cell structural part
        self.cell_struct_lin = nn.Linear(cell_feat_dim - cell_type_dim, hidden_dim)

        # -----------------------------
        # Cell MLP (logic aware)
        # -----------------------------
        self.cell_mlp = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # -----------------------------
        self.out_lin = nn.Linear(hidden_dim, 1)

    # ------------------------------------------------
    def aggregate(self, src_x, edge_index, out_size):

        msg = src_x[edge_index[0]]

        agg = torch.zeros((out_size, src_x.size(1)), device=src_x.device)
        agg.index_add_(0, edge_index[1], msg)

        deg = torch.bincount(edge_index[1], minlength=out_size).clamp(min=1).unsqueeze(1)
        agg = agg / deg

        return agg

    # ------------------------------------------------
    def forward(self, data: HeteroData):

        # -----------------------------
        # Initial projections
        # -----------------------------
        net_x = F.relu(self.net_lin(data['net'].x))
        pin_in_x = F.relu(self.pin_in_lin(data['pin_in'].x))
        pin_out_x = F.relu(self.pin_out_lin(data['pin_out'].x))

        # -----------------------------
        # Cell feature split
        # -----------------------------
        cell_feat = data['cell'].x

        cell_type_onehot = cell_feat[:, :self.cell_type_dim]
        cell_struct = cell_feat[:, self.cell_type_dim:]

        cell_struct = F.relu(self.cell_struct_lin(cell_struct))
        cell_type_emb = self.cell_type_embed(cell_type_onehot)

        cell_x = cell_struct

        # =================================================
        # Message Passing
        # =================================================
        for _ in range(self.num_layers):

            # net → pin_in
            edge_index = data['net', 'net_to_pin_in', 'pin_in'].edge_index
            pin_agg = self.aggregate(net_x, edge_index, pin_in_x.size(0))
            pin_in_x = F.relu(pin_in_x + pin_agg)

            # pin_in → cell
            edge_index = data['pin_in', 'pin_in_to_cell', 'cell'].edge_index
            cell_agg = self.aggregate(pin_in_x, edge_index, cell_x.size(0))
            cell_x = F.relu(cell_x + cell_agg)

            # cell → pin_out
            cell_input = torch.cat([cell_x, cell_type_emb], dim=1)
            cell_out = self.cell_mlp(cell_input)

            edge_index = data['cell', 'cell_to_pin_out', 'pin_out'].edge_index
            pout_agg = self.aggregate(cell_out, edge_index, pin_out_x.size(0))
            pin_out_x = F.relu(pin_out_x + pout_agg)

            # pin_out → net
            edge_index = data['pin_out', 'pin_out_to_net', 'net'].edge_index
            net_agg = self.aggregate(pin_out_x, edge_index, net_x.size(0))

            # residual update
            net_x = F.relu(net_x + net_agg)

            net_x = F.dropout(net_x, p=self.dropout, training=self.training)

        # -----------------------------
        # Output
        # -----------------------------
        out = self.out_lin(net_x)

        return out