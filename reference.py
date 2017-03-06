import numpy as np
import utils
import constant
import pickle

def get_crowfly_distances():
    with open("reference/crowfly.p", "rb") as f:
        return pickle.load(f)

def get_by_mode_and_purpose(distance = "routed"):
    means = {}
    variances = {}

    with open("reference/stats.csv") as f:
        for line in f:
            if line.startswith("mode"): continue
            line = line.strip().split('\t')

            if distance == "routed":
                means[(line[0], line[1])] = float(line[2])
                variances[(line[0], line[1])] = float(line[3])
            else:
                means[(line[0], line[1])] = float(line[4])
                variances[(line[0], line[1])] = float(line[5])

    return means, variances

def get_by_purpose(distance = "routed"):
    means = {}
    variances = {}

    with open("reference/stats_purpose.csv") as f:
        for line in f:
            if line.startswith("purpose"): continue
            line = line.strip().split('\t')

            if distance == "routed":
                means[line[0]] = float(line[1])
                variances[line[0]] = float(line[2])
            else:
                means[line[0]] = float(line[3])
                variances[line[0]] = float(line[4])

    return means, variances
