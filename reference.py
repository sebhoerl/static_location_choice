import numpy as np
import utils
import constant

def get_by_mode_and_purpose(distance = "routed"):
    data = {}

    with open("reference/stats.csv") as f:
        for line in f:
            if line.startswith("mode"): continue
            line = line.strip().split('\t')

            if distance == "routed":
                data[(line[0], line[1])] = float(line[2])
            else:
                data[(line[0], line[1])] = float(line[4])

    return data

def get_by_purpose(distance = "routed"):
    data = {}

    with open("reference/stats_purpose.csv") as f:
        for line in f:
            if line.startswith("purpose"): continue
            line = line.strip().split('\t')

            if distance == "routed":
                data[line[0]] = float(line[1])
            else:
                data[line[0]] = float(line[3])

    return data
