import re
from pathlib import Path


class SAIFParser:

    def __init__(self, saif_file):

        self.saif_file = Path(saif_file)

        self.duration = None
        self.net_toggle = {}


    def parse(self):

        current_net = None

        with self.saif_file.open() as f:

            for line in f:

                # --------------------------------
                # Read simulation duration
                # --------------------------------

                if "DURATION" in line:

                    m = re.search(r"DURATION\s+([0-9.]+)", line)

                    if m:
                        self.duration = float(m.group(1))


                # --------------------------------
                # Detect net name
                # --------------------------------

                net_match = re.match(r"\s*\(([^()\s]+)\s*$", line)

                if net_match:

                    name = net_match.group(1)

                    # ignore SAIF keywords
                    if name not in ["INSTANCE", "NET"]:

                        current_net = name

                    continue


                # --------------------------------
                # Read toggle count
                # --------------------------------

                if "(TC" in line:

                    m = re.search(r"\(TC\s+(\d+)\)", line)

                    if m and current_net and self.duration:

                        TC = float(m.group(1))

                        toggle_rate = TC / self.duration

                        self.net_toggle[current_net] = toggle_rate

                        current_net = None


        return self.net_toggle