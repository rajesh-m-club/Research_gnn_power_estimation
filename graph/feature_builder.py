import json
import math
import torch
from collections import deque, defaultdict


class FeatureBuilder:

    def __init__(self, graph, saif_data, cell_library_file, pi_nets=None):

        self.graph = graph
        self.saif_data = saif_data
        self.pi_nets = pi_nets if pi_nets else set()

        with open(cell_library_file) as f:
            self.cell_library = json.load(f)

        # build logic family mapping
        self.logic_family_to_id = self.build_logic_family_map()


    # ------------------------------------------------
    # Build logic family mapping
    # ------------------------------------------------

    def build_logic_family_map(self):

        families = set()

        for cell in self.cell_library.values():
            families.add(cell["logic_family"])

        families = sorted(list(families))

        return {fam: i for i, fam in enumerate(families)}


    # ------------------------------------------------
    # One-hot encode logic family
    # ------------------------------------------------

    def encode_logic_family(self, family):

        vec = [0] * len(self.logic_family_to_id)

        idx = self.logic_family_to_id.get(family)

        if idx is not None:
            vec[idx] = 1

        return vec


    # ------------------------------------------------
    # Compute cell fanin / fanout
    # ------------------------------------------------

    def compute_cell_fanin_fanout(self):

        num_nodes = self.graph.num_nodes

        fanin = [0] * num_nodes
        fanout = [0] * num_nodes

        edge_index = self.graph.edge_index

        for src, dst in edge_index.t():

            src = src.item()
            dst = dst.item()

            src_type = self.graph.node_types[src]
            dst_type = self.graph.node_types[dst]

            if src_type == "pin_in" and dst_type == "cell":
                fanin[dst] += 1

            if src_type == "cell" and dst_type == "pin_out":
                fanout[src] += 1

        return fanin, fanout


    # ------------------------------------------------
    # Faster logic level computation
    # ------------------------------------------------

    def compute_logic_levels(self):

        num_nodes = self.graph.num_nodes
        edge_index = self.graph.edge_index

        levels = [0] * num_nodes
        indegree = [0] * num_nodes

        adjacency = defaultdict(list)

        # build adjacency list
        for src, dst in edge_index.t():

            src = src.item()
            dst = dst.item()

            adjacency[src].append(dst)

            indegree[dst] += 1

        queue = deque()

        for i in range(num_nodes):
            if indegree[i] == 0:
                queue.append(i)

        while queue:

            node = queue.popleft()
            node_type = self.graph.node_types[node]

            for dst in adjacency[node]:

                dst_type = self.graph.node_types[dst]

                new_level = levels[node]

                # increment only when net → pin_in
                if node_type == "net" and dst_type == "pin_in":
                    new_level = levels[node] + 1

                levels[dst] = max(levels[dst], new_level)

                indegree[dst] -= 1

                if indegree[dst] == 0:
                    queue.append(dst)

        return levels


    # ------------------------------------------------
    # Build final feature matrix
    # ------------------------------------------------

    def build_features(self):

        num_nodes = self.graph.num_nodes

        cell_fanin, cell_fanout = self.compute_cell_fanin_fanout()

        logic_levels = self.compute_logic_levels()

        max_level = max(logic_levels) if max(logic_levels) > 0 else 1
        logic_levels = [lvl / max_level for lvl in logic_levels]

        features = []

        for idx in range(num_nodes):

            node_name = self.graph.idx_to_name[idx]
            node_type = self.graph.node_types[idx]

            # --------------------------------
            # Base node type
            # --------------------------------

            base_feature = self.graph.x[idx].tolist()

            # --------------------------------
            # Logic family encoding
            # --------------------------------

            logic_feature = [0] * len(self.logic_family_to_id)

            if node_type == "cell":

                cell_type = self.graph.cell_types.get(node_name)

                if cell_type in self.cell_library:

                    logic_family = self.cell_library[cell_type]["logic_family"]

                    logic_feature = self.encode_logic_family(logic_family)

            # --------------------------------
            # Fanin / fanout
            # --------------------------------

            f_in = 0
            f_out = 0

            if node_type == "cell":

                f_in = cell_fanin[idx]
                f_out = cell_fanout[idx]

            elif node_type == "net":

                f_in = len(self.graph.net_drivers.get(node_name, []))
                f_out = len(self.graph.net_sinks.get(node_name, []))

            f_in = math.log1p(f_in)
            f_out = math.log1p(f_out)

            # --------------------------------
            # Logic level
            # --------------------------------

            lvl = logic_levels[idx]

            # --------------------------------
            # Toggle feature (MASKED)
            # --------------------------------

            toggle = 0.0

            if node_type == "net":

                raw_toggle = self.saif_data.get(node_name, 0.0)

                # only PI nets reveal toggle feature
                if node_name in self.pi_nets:

                    toggle = math.log1p(raw_toggle * 1e4)
                    toggle = min(toggle, 5.0)

                else:

                    toggle = 0.0

            # --------------------------------
            # Final feature vector
            # --------------------------------

            feature = (
                base_feature +
                logic_feature +
                [
                    f_in,
                    f_out,
                    lvl,
                    toggle
                ]
            )

            features.append(feature)

        x = torch.tensor(features, dtype=torch.float)

        self.graph.x = x

        return self.graph