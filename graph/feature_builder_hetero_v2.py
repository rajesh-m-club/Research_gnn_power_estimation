import json
import math
import torch
from collections import defaultdict, deque


class FeatureBuilderHetero_v2:

    def __init__(self, data, saif_data, cell_library_file):

        self.data = data
        self.saif_data = saif_data

        with open(cell_library_file) as f:
            self.cell_library = json.load(f)

        self.logic_family_to_id = self.build_logic_family_map()

    # ------------------------------------------------
    def build_logic_family_map(self):
        families = set()
        for cell in self.cell_library.values():
            families.add(cell["logic_family"])
        families = sorted(list(families))
        return {fam: i for i, fam in enumerate(families)}

    def encode_logic_family(self, family):
        vec = [0] * len(self.logic_family_to_id)
        if family in self.logic_family_to_id:
            vec[self.logic_family_to_id[family]] = 1
        return vec

    # ------------------------------------------------
    def build_global_graph(self):

        node_list = []
        node_type = []
        node_index = {}

        for ntype in self.data.node_map:
            for name, idx in self.data.node_map[ntype].items():
                gid = len(node_list)
                node_list.append(name)
                node_type.append(ntype)
                node_index[(ntype, idx)] = gid

        edges = []

        for (src_type, rel, dst_type) in self.data.edge_types:
            edge_index = self.data[(src_type, rel, dst_type)].edge_index

            for i in range(edge_index.shape[1]):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()

                gsrc = node_index[(src_type, src)]
                gdst = node_index[(dst_type, dst)]

                edges.append((gsrc, gdst))

        return node_list, node_type, node_index, edges

    # ------------------------------------------------
    def compute_fanin_fanout(self, node_type, edges):

        n = len(node_type)
        fanin = [0] * n
        fanout = [0] * n

        for src, dst in edges:
            fanout[src] += 1
            fanin[dst] += 1

        return fanin, fanout

    # ------------------------------------------------
    def compute_logic_levels(self, node_type, edges):

        n = len(node_type)

        adj = defaultdict(list)
        indegree = [0] * n
        level = [0] * n

        for src, dst in edges:
            adj[src].append(dst)
            indegree[dst] += 1

        queue = deque()

        for i in range(n):
            if indegree[i] == 0:
                queue.append(i)

        while queue:
            node = queue.popleft()

            for dst in adj[node]:

                new_level = level[node]

                if node_type[node] == "net" and node_type[dst] == "pin_in":
                    new_level += 1

                level[dst] = max(level[dst], new_level)

                indegree[dst] -= 1

                if indegree[dst] == 0:
                    queue.append(dst)

        max_lvl = max(level) if max(level) > 0 else 1
        return [l / max_lvl for l in level]

    # ------------------------------------------------
    def build(self):

        node_list, node_type, node_index, edges = self.build_global_graph()

        fanin, fanout = self.compute_fanin_fanout(node_type, edges)
        levels = self.compute_logic_levels(node_type, edges)

        # =========================
        # NET FEATURES (UNCHANGED)
        # =========================
        net_features = []

        net_items = sorted(self.data.node_map["net"].items(), key=lambda x: x[1])

        for net_name, idx in net_items:

            gid = node_index[("net", idx)]

            f_in = math.log1p(fanin[gid])
            f_out = math.log1p(fanout[gid])
            lvl = levels[gid]

            raw_toggle = self.saif_data.get(net_name, 0.0)
            toggle = math.log1p(raw_toggle * 1e4)
            toggle = min(toggle, 5.0)

            is_pi = net_name not in self.data.net_drivers
            input_toggle = toggle if is_pi else 0.0

            net_features.append([f_in, f_out, lvl, input_toggle])

        self.data["net"].x = torch.tensor(net_features, dtype=torch.float)

        # =========================
        # CELL FEATURES (UNCHANGED)
        # =========================
        cell_features = []
        cell_items = sorted(self.data.node_map["cell"].items(), key=lambda x: x[1])

        for cell_name, idx in cell_items:

            gid = node_index[("cell", idx)]

            logic_vec = [0] * len(self.logic_family_to_id)

            if cell_name in self.data.cell_types:
                cell_type = self.data.cell_types[cell_name]
                if cell_type in self.cell_library:
                    fam = self.cell_library[cell_type]["logic_family"]
                    logic_vec = self.encode_logic_family(fam)

            f_in = math.log1p(fanin[gid])
            f_out = math.log1p(fanout[gid])
            lvl = levels[gid]

            cell_features.append(logic_vec + [f_in, f_out, lvl])

        self.data["cell"].x = torch.tensor(cell_features, dtype=torch.float)

        # =========================
        # PIN_IN FEATURES (SAFE)
        # =========================
        pin_in_features = []
        pin_in_items = sorted(self.data.node_map["pin_in"].items(), key=lambda x: x[1])

        for _, idx in pin_in_items:

            gid = node_index[("pin_in", idx)]

            pin_in_features.append([
                math.log1p(fanin[gid]),
                math.log1p(fanout[gid]),
                levels[gid]
            ])

        self.data["pin_in"].x = torch.tensor(pin_in_features, dtype=torch.float)

        # =========================
        # PIN_OUT FEATURES (SAFE)
        # =========================
        pin_out_features = []
        pin_out_items = sorted(self.data.node_map["pin_out"].items(), key=lambda x: x[1])

        for _, idx in pin_out_items:

            gid = node_index[("pin_out", idx)]

            pin_out_features.append([
                math.log1p(fanin[gid]),
                math.log1p(fanout[gid]),
                levels[gid]
            ])

        self.data["pin_out"].x = torch.tensor(pin_out_features, dtype=torch.float)

        # =========================
        # PRIMARY
        # =========================
        self.data["primary_in"].x = torch.zeros((len(self.data.node_map["primary_in"]), 1))
        self.data["primary_out"].x = torch.zeros((len(self.data.node_map["primary_out"]), 1))

        # cleanup
        del self.data.node_map
        del self.data.cell_types
        del self.data.net_drivers
        del self.data.net_sinks

        return self.data