import re
import os
import torch
from torch_geometric.data import HeteroData


class NetlistParserHetero:

    def __init__(self, netlist_file):
        self.netlist_file = netlist_file

        # Node storage
        self.node_features = {
            "cell": [],
            "net": [],
            "pin_in": [],
            "pin_out": [],
            "primary_in": [],
            "primary_out": []
        }

        self.node_map = {
            "cell": {},
            "net": {},
            "pin_in": {},
            "pin_out": {},
            "primary_in": {},
            "primary_out": {}
        }

        # Edge storage
        self.edge_index_dict = {
            ("net", "net_to_pin_in", "pin_in"): [],
            ("pin_in", "pin_in_to_cell", "cell"): [],
            ("cell", "cell_to_pin_out", "pin_out"): [],
            ("pin_out", "pin_out_to_net", "net"): [],
            ("primary_in", "pi_to_net", "net"): [],
            ("net", "net_to_po", "primary_out"): []
        }

        # Tracking
        self.net_nodes = set()
        self.cell_types = {}
        self.net_drivers = {}
        self.net_sinks = {}

        # Node type one-hot
        self.node_types = {
            "net": [1, 0, 0, 0, 0, 0],
            "pin_in": [0, 1, 0, 0, 0, 0],
            "cell": [0, 0, 1, 0, 0, 0],
            "pin_out": [0, 0, 0, 1, 0, 0],
            "primary_in": [0, 0, 0, 0, 1, 0],
            "primary_out": [0, 0, 0, 0, 0, 1]
        }

        # Multi-output cells
        self.two_output_pattern = re.compile(r'(FA1D0|FICIND1|CMPE32D1|CMPE32D2|EDFD1)')
        self.three_output_pattern = re.compile(r'(CMPE42D1)')

    # ------------------------------------------------
    # Node creation
    # ------------------------------------------------
    def add_node(self, name, node_type, feature=None):
        if name not in self.node_map[node_type]:
            idx = len(self.node_map[node_type])
            self.node_map[node_type][name] = idx

            self.node_features[node_type].append(
                feature if feature is not None else self.node_types[node_type]
            )

            if node_type == "net":
                self.net_nodes.add(name)

        return self.node_map[node_type][name]

    # ------------------------------------------------
    # Safe port splitting
    # ------------------------------------------------
    def split_ports(self, port_str):
        ports = []
        buf = ""
        depth = 0

        for c in port_str:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1

            if c == ',' and depth == 0:
                ports.append(buf.strip())
                buf = ""
            else:
                buf += c

        if buf:
            ports.append(buf.strip())

        return ports

    # ------------------------------------------------
    # Parse cells
    # ------------------------------------------------
    def parse_cells(self, lines):

        # Improved regex (handles real netlists)
        gate_regex = re.compile(r'(\S+)\s+(\S+)\s*\((.*)\);')

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Skip non-cell lines
            if any(line.startswith(k) for k in [
                "module", "endmodule", "input", "output", "wire", "assign"
            ]):
                continue

            match = gate_regex.search(line)
            if not match:
                continue

            gate_type = match.group(1)
            inst_name = match.group(2)
            ports_str = match.group(3)

            # Create cell node
            cell_node = self.add_node(inst_name, "cell")
            self.cell_types[inst_name] = gate_type

            ports = self.split_ports(ports_str)

            # Output count detection
            if self.three_output_pattern.search(gate_type):
                output_count = 3
            elif self.two_output_pattern.search(gate_type):
                output_count = 2
            else:
                output_count = 1

            input_count = len(ports) - output_count

            for idx, p in enumerate(ports):
                port_match = re.match(r'\.([^\(\s]+)\((.+)\)$', p)
                if not port_match:
                    continue

                pin_name = port_match.group(1)
                net_name = port_match.group(2).strip()
                pin_full = inst_name + "." + pin_name

                net_node = self.add_node(net_name, "net")

                # INPUT PIN
                if idx < input_count:
                    pin_node = self.add_node(pin_full, "pin_in")

                    self.edge_index_dict[("net", "net_to_pin_in", "pin_in")].append([net_node, pin_node])
                    self.edge_index_dict[("pin_in", "pin_in_to_cell", "cell")].append([pin_node, cell_node])

                    self.net_sinks.setdefault(net_name, []).append(pin_node)

                # OUTPUT PIN
                else:
                    pin_node = self.add_node(pin_full, "pin_out")

                    self.edge_index_dict[("cell", "cell_to_pin_out", "pin_out")].append([cell_node, pin_node])
                    self.edge_index_dict[("pin_out", "pin_out_to_net", "net")].append([pin_node, net_node])

                    self.net_drivers.setdefault(net_name, []).append(pin_node)

    # ------------------------------------------------
    # Infer PI / PO
    # ------------------------------------------------
    def infer_primary_ios(self):
        for net in self.net_nodes:

            net_node = self.node_map["net"][net]

            # PI
            if net not in self.net_drivers:
                pi_name = net + "_PI"
                pi_node = self.add_node(pi_name, "primary_in")

                self.edge_index_dict[("primary_in", "pi_to_net", "net")].append(
                    [pi_node, net_node]
                )

            # PO
            if net not in self.net_sinks:
                po_name = net + "_PO"
                po_node = self.add_node(po_name, "primary_out")

                self.edge_index_dict[("net", "net_to_po", "primary_out")].append(
                    [net_node, po_node]
                )

    # ------------------------------------------------
    # Main parse
    # ------------------------------------------------
    def parse(self):
        with open(self.netlist_file, "r") as f:
            text = f.read()

        # Remove comments
        text = re.sub(r'//.*?\n', '', text)

        lines = text.split("\n")

        self.parse_cells(lines)
        self.infer_primary_ios()

        return self.build_graph()

    # ------------------------------------------------
    # Build graph (BATCH SAFE)
    # ------------------------------------------------
    def build_graph(self):
        data = HeteroData()

        # Nodes (FIXED)
        for ntype in self.node_features:
            feats = self.node_features[ntype]

            if len(feats) == 0:
                data[ntype].x = torch.zeros(
                    (0, len(self.node_types[ntype])),
                    dtype=torch.float
                )
            else:
                data[ntype].x = torch.tensor(feats, dtype=torch.float)

        # Edges
        for (src, rel, dst), edges in self.edge_index_dict.items():
            if len(edges) > 0:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[(src, rel, dst)].edge_index = edge_index

        # Metadata (SAFE)
        data.node_map = self.node_map
        data.cell_types = self.cell_types
        data.net_drivers = self.net_drivers
        data.net_sinks = self.net_sinks

        return data


# ------------------------------------------------
# Standalone test
# ------------------------------------------------
if __name__ == "__main__":

    netlist_file = "../data/netlists/netlist.v"

    parser = NetlistParserHetero(netlist_file)
    graph = parser.parse()

    print("\n===== HETERO GRAPH STATS =====")

    print("\nNode counts:")
    for ntype in graph.node_types:
        print(f"{ntype}: {graph[ntype].num_nodes}")

    print("\nEdge counts:")
    for etype in graph.edge_types:
        print(f"{etype}: {graph[etype].edge_index.shape[1]}")

    os.makedirs("../processed", exist_ok=True)
    torch.save(graph, "../processed/netlist_graph_hetero.pt")

    print("\nGraph saved successfully")