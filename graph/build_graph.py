import os
import torch
import math

from parser.netlist_parser import NetlistParser
from parser.saif_parser import SAIFParser
from graph.feature_builder import FeatureBuilder


class GraphBuilder:

    def __init__(self, netlist_dir, saif_dir, cell_lib_file):

        self.netlist_dir = netlist_dir
        self.saif_dir = saif_dir
        self.cell_lib_file = cell_lib_file


    # ------------------------------------------------
    # Detect PI nets
    # ------------------------------------------------

    def find_primary_input_nets(self, graph):

        pi_nets = set()

        edge_index = graph.edge_index

        for src, dst in edge_index.t():

            src = src.item()
            dst = dst.item()

            src_type = graph.node_types[src]
            dst_type = graph.node_types[dst]

            if src_type == "primary_in" and dst_type == "net":

                net_name = graph.idx_to_name[dst]
                pi_nets.add(net_name)

        return pi_nets


    # ------------------------------------------------
    # Build single graph
    # ------------------------------------------------

    def build_single_graph(self, netlist_file, saif_file):

        print(f"\nProcessing {netlist_file}")

        # ----------------------
        # Parse netlist
        # ----------------------

        net_parser = NetlistParser(netlist_file)
        graph = net_parser.parse()

        # ----------------------
        # Detect PI nets FIRST
        # ----------------------

        pi_nets = self.find_primary_input_nets(graph)

        print("Primary input nets:", len(pi_nets))

        # ----------------------
        # Parse SAIF
        # ----------------------

        saif_parser = SAIFParser(saif_file)
        saif_data = saif_parser.parse()

        print("SAIF nets:", len(saif_data))

        # ----------------------
        # Build features
        # ----------------------

        feature_builder = FeatureBuilder(
            graph,
            saif_data,
            self.cell_lib_file,
            pi_nets
        )

        graph = feature_builder.build_features()

        # ----------------------
        # Build targets
        # ----------------------

        y = []
        train_mask = []

        for idx in range(graph.num_nodes):

            node_name = graph.idx_to_name[idx]
            node_type = graph.node_types[idx]

            toggle = 0.0

            if node_type == "net":
                raw_toggle = saif_data.get(node_name, 0.0)

                toggle = math.log1p(raw_toggle * 1e4)
                toggle = min(toggle, 5.0)

            y.append(toggle)

            # train only on internal nets
            if node_type == "net" and node_name not in pi_nets:

                train_mask.append(True)

            else:

                train_mask.append(False)

        graph.y = torch.tensor(y, dtype=torch.float)
        graph.train_mask = torch.tensor(train_mask, dtype=torch.bool)

        return graph


    # ------------------------------------------------
    # Build dataset
    # ------------------------------------------------

    def build_dataset(self):

        graphs = []

        netlist_files = sorted(
            [f for f in os.listdir(self.netlist_dir) if f.endswith(".v")]
        )

        for net_file in netlist_files:

            net_path = os.path.join(self.netlist_dir, net_file)

            saif_file = net_file.replace(".v", ".saif")
            saif_path = os.path.join(self.saif_dir, saif_file)

            if not os.path.exists(saif_path):

                print(f"SAIF missing for {net_file}")
                continue

            graph = self.build_single_graph(net_path, saif_path)

            graphs.append(graph)

            print(
                f"Graph built: nodes={graph.num_nodes} edges={graph.num_edges}"
            )

        print("\nTotal graphs:", len(graphs))

        return graphs