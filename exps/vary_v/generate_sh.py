import sys, os
import argparse

base = f"python exps/vary_v/run.py -t 0"
# example : -t 0 -w SHAC_GTV_Pos_Dis3.0_spd3.4_2.zip -v 1.0 -tr B

trs = ["B", "D", "8"]
vs = [0.5,1.0,1.5,2.0]


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument("--weight", "-w", type=str, default=None, )
    return parser

# generate a series of bash to run

def generate_sh(w):
    with open(f"run.sh", "w") as f:
        for tr in trs:
            for v in vs:
                cmd = f"{base} -w {w} -v {v} -tr {tr}\n"
                f.write(cmd)


args = parse_args().parse_args()
weight = args.weight
assert weight is not None, "Please provide a weight file with -w option"
generate_sh(weight)

import os
os.system("chmod +x run.sh")



