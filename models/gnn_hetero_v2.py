import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData


class ToggleHeteroGNN_v2(nn.Module):

    def __init__(self,
                 net_feat_dim=4,
                 pin_feat_dim=3,
                 cell_feat_dim=32,
                 cell_type_dim=26,
                 embed_dim=8,
                 hidden_dim=64,
                 num_layers=4,
                 dropout=0.2):

        super(ToggleHeteroGNN_v2, self).__init__()

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

        self.cell_struct_lin = nn.Linear(cell_feat_dim - cell_type_dim, hidden_dim)

        # -----------------------------
        # Normalization (NEW)
        # -----------------------------
        self.net_norm = nn.LayerNorm(hidden_dim)
        self.pin_in_norm = nn.LayerNorm(hidden_dim)
        self.pin_out_norm = nn.LayerNorm(hidden_dim)
        self.cell_norm = nn.LayerNorm(hidden_dim)

        # -----------------------------
        # Cell MLP
        # -----------------------------
        self.cell_mlp = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # -----------------------------
        # Output
        # -----------------------------
        self.out_lin = nn.Linear(hidden_dim, 1)

    # ------------------------------------------------
    # Improved aggregation (degree-normalized)
    # ------------------------------------------------
    def aggregate(self, src_x, edge_index, out_size):

        src, dst = edge_index

        msg = src_x[src]

        # normalize by source degree
        deg = torch.bincount(src, minlength=src_x.size(0)).clamp(min=1)
        norm = 1.0 / deg[src]

        msg = msg * norm.unsqueeze(1)

        agg = torch.zeros((out_size, src_x.size(1)), device=src_x.device)
        agg.index_add_(0, dst, msg)

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

            # ---------- net → pin_in ----------
            edge_index = data['net', 'net_to_pin_in', 'pin_in'].edge_index
            pin_agg = self.aggregate(net_x, edge_index, pin_in_x.size(0))

            pin_in_x = self.pin_in_norm(pin_in_x + pin_agg)
            pin_in_x = F.relu(pin_in_x)

            # ---------- pin_in → cell ----------
            edge_index = data['pin_in', 'pin_in_to_cell', 'cell'].edge_index
            cell_agg = self.aggregate(pin_in_x, edge_index, cell_x.size(0))

            cell_x = self.cell_norm(cell_x + cell_agg)
            cell_x = F.relu(cell_x)

            # ---------- cell → pin_out ----------
            cell_input = torch.cat([cell_x, cell_type_emb], dim=1)
            cell_out = self.cell_mlp(cell_input)

            edge_index = data['cell', 'cell_to_pin_out', 'pin_out'].edge_index
            pout_agg = self.aggregate(cell_out, edge_index, pin_out_x.size(0))

            pin_out_x = self.pin_out_norm(pin_out_x + pout_agg)
            pin_out_x = F.relu(pin_out_x)

            # ---------- pin_out → net ----------
            edge_index = data['pin_out', 'pin_out_to_net', 'net'].edge_index
            net_agg = self.aggregate(pin_out_x, edge_index, net_x.size(0))

            net_x = self.net_norm(net_x + net_agg)
            net_x = F.relu(net_x)

            net_x = F.dropout(net_x, p=self.dropout, training=self.training)

        # -----------------------------
        # Output
        # -----------------------------
        out = self.out_lin(net_x)

        return out