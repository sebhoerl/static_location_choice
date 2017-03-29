import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats
import gzip

with open("reference/crowfly.p", "rb") as f:
    reference = pickle.load(f)[0]

matsim = {}

with gzip.open("validation/matsim_data.gz") as f:
    for line in f:
        line = line.split(b" ")
        matsim[(str(line[0], "ascii"), str(line[1], "ascii"))] = np.array(line[2:], dtype = np.float) / 1000.0

for c in tqdm(reference.keys()):
    plt.figure()

    tf = lambda x: np.log(x)
    tf = lambda x: x

    data = reference[c]
    n = len(data)
    p = np.arange(1, n + 1) / (n + 1)
    plt.plot(tf(np.sort(data)), p, 'b--')

    beta = np.mean(data) / np.var(data)
    alpha = np.mean(data) * beta
    dist = scipy.stats.gamma(alpha, scale = 1/beta)

    plt.plot(tf(dist.ppf(p)), p, 'k--')

    mu = np.mean(data)
    s2 = np.var(data)
    dist = scipy.stats.norm(loc = mu, scale = np.sqrt(s2))

    plt.plot(tf(dist.ppf(p)), p, 'g--')

    data = matsim[c]
    n = len(data)
    p = np.arange(1, n + 1) / (n + 1)
    plt.plot(tf(np.sort(data)), p, 'r')

    plt.savefig("validation/%s_%s.png" % c)
    plt.close()
