import pyproj
from tqdm import tqdm
import numpy as np
import itertools
import pickle

purposes = ["remote_work", "leisure", "shop", "escort_kids", "escort_other"] + ["home", "work", "education"]
modes = ["car", "pt", "bike", "walk"]

purpose_map = {
    '-99' : None,
    '1': None,
    '2': None,
    '3': None,
    '4': 'shop',
    '5': 'shop',
    '6': 'remote_work',
    '7': 'remote_work',
    '8': 'leisure',
    '9': 'escort_kids',
    '10': 'escort_other',
    '11': None,
    '12': None
}

purpose_map["2"] = "work"
purpose_map["3"] = "education"
purpose_map["11"] = "home"

mode_map = {
    '-99' : None,
    '1' : None,
    '2' : 'pt', # Bahn
    '3' : 'pt', # Postauto
    '4' : 'pt', # Schiff
    '5' : 'pt', # Tram
    '6' : 'pt', # Bus
    '7' : 'pt', # Sonstig OEV
    '8' : 'pt', # Reisecar
    '9' : 'car', # Auto
    '10' : None,
    '11' : None,
    '12' : None,
    '13' : None,
    '14' : "bike", #'bike',
    '15' : "walk", #'walk',
    '16' : None,
    '17' : None
}

routed_distances = { (m, p) : [] for m, p in itertools.product(modes, purposes) }
crowfly_distances = { (m, p) : [] for m, p in itertools.product(modes, purposes) }

routed_distances_purpose = { p : [] for p in purposes }
crowfly_distances_purpose = { p : [] for p in purposes }

routed_distances_mode = { m : [] for m in modes }
crowfly_distances_mode = { m : [] for m in modes }

with open("Sebastian.csv") as f:
    for line in tqdm(f):
        line = line.replace('"', '').strip().split(';')
        mode, purpose, routed_distance, crowfly_distance = line[9], line[10], line[11], line[12]

        if line[11] == "-99" or line[12] == "-99": continue
        both = True

        if purpose in purpose_map and purpose_map[purpose] is not None:
            purpose = purpose_map[purpose]
            routed_distances_purpose[purpose].append(float(routed_distance))
            crowfly_distances_purpose[purpose].append(float(crowfly_distance))
        else:
            both = False

        if mode in mode_map and mode_map[mode] is not None:
            mode = mode_map[mode]
            routed_distances_mode[mode].append(float(routed_distance))
            crowfly_distances_mode[mode].append(float(crowfly_distance))
        else:
            both = False

        if both:
            routed_distances[(mode, purpose)].append(float(routed_distance))
            crowfly_distances[(mode, purpose)].append(float(crowfly_distance))

with open("crowfly.p", "wb+") as f:
    pickle.dump((crowfly_distances, crowfly_distances_purpose, crowfly_distances_mode), f)

with open("stats.csv", "w+") as f:
    f.write("mode\tpurpose\tmean_routed_distance\tvar_routed_distance\tmean_crowfly_distance\tvar_crowfly_distance\n")

    for m, p in itertools.product(modes, purposes):
        f.write("%s\t%s\t%f\t%f\t%f\t%f\n" % (m, p, np.mean(routed_distances[(m,p)]), np.var(routed_distances[(m,p)]), np.mean(crowfly_distances[(m,p)]), np.var(crowfly_distances[(m,p)])))

with open("stats_purpose.csv", "w+") as f:
    f.write("purpose\tmean_routed_distance\tvar_routed_distance\tmean_crowfly_distance\tvar_crowfly_distance\n")

    for p in purposes:
        f.write("%s\t%f\t%f\t%f\t%f\n" % (p, np.mean(routed_distances_purpose[p]), np.var(routed_distances_purpose[p]), np.mean(crowfly_distances_purpose[p]), np.var(crowfly_distances_purpose[p])))
