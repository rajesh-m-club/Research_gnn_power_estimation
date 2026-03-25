import os
import torch

from parser.netlist_parser_hetero import NetlistParserHetero
from parser.saif_parser import SAIFParser
from graph.feature_builder_hetero import FeatureBuilderHetero


class GraphBuilderHetero:

    def __init__(self, netlist_dir, saif_dir, cell_lib_file):
        self.netlist_dir = netlist_dir
        self.saif_dir = saif_dir
        self.cell_lib_file = cell_lib_file

    # ------------------------------------------------
    # Build single hetero graph
    # ------------------------------------------------
    def build_single_graph(self, netlist_file, saif_file, mode="train"):

        print(f"\nProcessing {netlist_file} ({mode})")

        # ----------------------
        # Parse netlist
        # ----------------------
        parser = NetlistParserHetero(netlist_file)
        data = parser.parse()

        node_map = data.node_map

        # ----------------------
        # Parse SAIF
        # ----------------------
        saif_parser = SAIFParser(saif_file)
        saif_data = saif_parser.parse()

        # =========================
        # TARGET + MASKS
        # =========================
        net_map = node_map["net"]

        net_names = [None] * len(net_map)
        for name, idx in net_map.items():
            net_names[idx] = name

        y = []
        train_mask = []
        eval_mask = []

        missing_count = 0

        for net_name in net_names:

            has_label = net_name in saif_data

            if not has_label:
                missing_count += 1

            # ----------------------
            # TRAIN MODE
            # ----------------------
            if mode == "train":
                if not has_label:
                    raise ValueError(f"[TRAIN ERROR] Missing net in SAIF: {net_name}")

                raw_toggle = saif_data[net_name]

            # ----------------------
            # TEST MODE
            # ----------------------
            else:
                raw_toggle = saif_data.get(net_name, 0.0)

            # transform
            toggle = torch.log1p(torch.tensor(raw_toggle * 1e4))
            toggle = torch.clamp(toggle, max=5.0)

            y.append(toggle)

            # ----------------------
            # MASKS
            # ----------------------
            is_pi = net_name not in parser.net_drivers

            # TRAIN mask → used for loss
            if mode == "train":
                train_mask.append(not is_pi)
                eval_mask.append(True)

            # TEST mask
            else:
                train_mask.append(False)        # no training on test graph
                eval_mask.append(has_label)     # only PI/PO available

        if missing_count > 0:
            print(f"Missing nets in SAIF: {missing_count}")

        data["net"].y = torch.stack(y).view(-1, 1)
        data["net"].train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data["net"].eval_mask = torch.tensor(eval_mask, dtype=torch.bool)

        print("Trainable nets:", sum(train_mask), "/", len(train_mask))
        print("Eval nets:", sum(eval_mask), "/", len(eval_mask))

        # =========================
        # FEATURES
        # =========================
        feature_builder = FeatureBuilderHetero(
            data,
            saif_data,
            self.cell_lib_file
        )

        data = feature_builder.build()

        return data

    # ------------------------------------------------
    # Build dataset (CONTROLLED SPLIT)
    # ------------------------------------------------
    def build_dataset(self, test_netlist_name):

        train_set = []
        test_set = []

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

            # ----------------------
            # Decide TRAIN / TEST
            # ----------------------
            if net_file == test_netlist_name:
                data = self.build_single_graph(net_path, saif_path, mode="test")
                test_set.append(data)
            else:
                data = self.build_single_graph(net_path, saif_path, mode="train")
                train_set.append(data)

            # summary
            print("----- GRAPH SUMMARY -----")
            for ntype in data.node_types:
                print(f"{ntype}: {data[ntype].num_nodes}")

            for etype in data.edge_types:
                print(f"{etype}: {data[etype].edge_index.shape[1]}")

        print("\nTraining graphs:", len(train_set))
        print("Testing graphs :", len(test_set))

        return train_set, test_set