import re
import torch
from torch_geometric.data import Data

class NetlistParser:

    def __init__(self, netlist_file):
        self.netlist_file = netlist_file

        # ---------------------------
        # Node storage
        # ---------------------------
        self.node_map = {}
        self.idx_to_name = {}
        self.node_type_list = []

        self.node_features = []
        self.edges = []

        # track nets separately
        self.net_nodes = set()

        # store cell types
        self.cell_types = {}

        # connectivity tracking
        self.net_drivers = {}
        self.net_sinks = {}

        # ---------------------------
        # Node type encoding (6-class one-hot)
        # ---------------------------
        self.node_types = {
            "net": [1,0,0,0,0,0],
            "pin_in": [0,1,0,0,0,0],
            "cell": [0,0,1,0,0,0],
            "pin_out": [0,0,0,1,0,0],
            "primary_in": [0,0,0,0,1,0],
            "primary_out": [0,0,0,0,0,1]
        }

        # cells with 2 outputs
        self.two_output_pattern = re.compile(r'(FA1D0|FICIND1|CMPE32D1|CMPE32D2|EDFD1)')
        # cells with 3 outputs
        self.three_output_pattern = re.compile(r'(CMPE42D1)')

    # ------------------------------------------------
    # Node creation
    # ------------------------------------------------
    def add_node(self, name, node_type):
        if name not in self.node_map:
            idx = len(self.node_map)
            self.node_map[name] = idx
            self.idx_to_name[idx] = name
            self.node_type_list.append(node_type)
            self.node_features.append(self.node_types[node_type])
            if node_type == "net":
                self.net_nodes.add(name)
        return self.node_map[name]

    # ------------------------------------------------
    # Split ports safely (handle nested parentheses)
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
    # Parse cells and nets
    # ------------------------------------------------
    def parse_cells(self, lines):
        # Updated regex to allow / in instance names
        gate_regex = re.compile(r'(\w+)\s+([\w/]+)\s*\((.*)\);')

        for line in lines:
            line = line.strip()
            match = gate_regex.search(line)
            if not match:
                continue

            gate_type = match.group(1)
            inst_name = match.group(2)
            ports_str = match.group(3)

            if gate_type.lower() == "module":
                continue

            # create cell node
            cell_node = self.add_node(inst_name, "cell")
            self.cell_types[inst_name] = gate_type

            ports = self.split_ports(ports_str)

            # -----------------------------
            # Detect number of outputs
            # -----------------------------
            if self.three_output_pattern.search(gate_type):
                output_count = 3
            elif self.two_output_pattern.search(gate_type):
                output_count = 2
            else:
                output_count = 1

            input_count = len(ports) - output_count

            for idx, p in enumerate(ports):
                # robust regex: pin name = anything except '(' or whitespace, net = anything inside parentheses
                port_match = re.match(r'\.([^\(\s]+)\((.+)\)$', p)
                if not port_match:
                    continue

                pin_name = port_match.group(1)
                net_name = port_match.group(2).strip()  # preserves /SUM[10], i_multiplier[21], etc.
                pin_full = inst_name + "." + pin_name

                net_node = self.add_node(net_name, "net")

                # -----------------------------
                # INPUT PIN
                # -----------------------------
                if idx < input_count:
                    pin_node = self.add_node(pin_full, "pin_in")
                    self.edges.append((net_node, pin_node))
                    self.edges.append((pin_node, cell_node))
                    self.net_sinks.setdefault(net_name, []).append(pin_node)
                # -----------------------------
                # OUTPUT PIN
                # -----------------------------
                else:
                    pin_node = self.add_node(pin_full, "pin_out")
                    self.edges.append((cell_node, pin_node))
                    self.edges.append((pin_node, net_node))
                    self.net_drivers.setdefault(net_name, []).append(pin_node)

    # ------------------------------------------------
    # Infer primary IOs
    # ------------------------------------------------
    def infer_primary_ios(self):
        for net in self.net_nodes:
            net_node = self.node_map[net]
            # Primary Input
            if net not in self.net_drivers:
                pi_name = net + "_PI"
                pi_node = self.add_node(pi_name, "primary_in")
                self.edges.append((pi_node, net_node))
            # Primary Output
            if net not in self.net_sinks:
                po_name = net + "_PO"
                po_node = self.add_node(po_name, "primary_out")
                self.edges.append((net_node, po_node))

    # ------------------------------------------------
    # Main parse
    # ------------------------------------------------
    def parse(self):
        with open(self.netlist_file) as f:
            text = f.read()

        # remove comments
        text = re.sub(r'//.*?\n', '', text)
        lines = text.split("\n")

        self.parse_cells(lines)
        self.infer_primary_ios()
        return self.build_graph()

    # ------------------------------------------------
    # Build PyG graph
    # ------------------------------------------------
    def build_graph(self):
        x = torch.tensor(self.node_features, dtype=torch.float)
        edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        data = Data(x=x, edge_index=edge_index)

        # attach metadata
        data.node_map = self.node_map
        data.idx_to_name = self.idx_to_name
        data.node_types = self.node_type_list
        data.cell_types = self.cell_types
        data.net_drivers = self.net_drivers
        data.net_sinks = self.net_sinks

        return data

# ------------------------------------------------
# Standalone run
# ------------------------------------------------
if __name__ == "__main__":
    netlist_file = "../data/netlists/netlist.v"
    parser = NetlistParser(netlist_file)
    graph = parser.parse()

    print("Graph statistics")
    print("----------------")
    print("Nodes :", graph.num_nodes)
    print("Edges :", graph.num_edges)

    torch.save(graph, "../processed/netlist_graph.pt")
    print("Graph saved to processed/netlist_graph.pt")