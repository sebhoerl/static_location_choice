from facilities import FacilityReader
from population import PopulationReader, PopulationWriter
import constant, sampler, proposals, likelihoods
import reference
import pickle

import numpy as np
from tqdm import tqdm

config = dict(
    cache_path = "cache"
)

facility_ids, facility_coordinates, facility_capacities = FacilityReader(config).read("/home/sebastian/static_locchoice/consistent/ch_1/facilities.xml.gz")
facility_id_to_index = { facility_id : index for index, facility_id in enumerate(facility_ids) }

activity_types, activity_modes, activity_facilities, activity_times = PopulationReader(config).read(
    "/home/sebastian/static_locchoice/consistent/ch_1/population.xml.gz",
    facility_id_to_index
)

relevant_activity_types = ["shop", "leisure", "escort_kids", "escort_other", "remote_work"]

facility_capacities = facility_capacities.astype(np.float) * 0.01
facility_capacities = np.ceil(facility_capacities)

# Random initialization
#for t in relevant_activity_types:
#    type_indices = np.where(facility_capacities[constant.ACTIVITY_TYPES_TO_INDEX[t],:] > 0)[0]
#    type_mask = activity_types == constant.ACTIVITY_TYPES_TO_INDEX[t]
#    #activity_facilities[type_mask] = np.random.choice(type_indices, np.sum(type_mask))
#    #activity_facilities[type_mask] = type_indices[0]

proposal_distribution = proposals.RandomProposalDistribution(relevant_activity_types, activity_types, facility_capacities)
reference_means, reference_variances = reference.get_by_purpose()
reference_means, reference_variances = reference.get_by_mode_and_purpose()

distance_likelihood = likelihoods.DistanceLikelihood(relevant_activity_types, activity_facilities, activity_modes, activity_types, facility_coordinates, reference_means, reference_variances)
distance_likelihood.initialize()

min_time, max_time, bins = 0.0, 3600 * 24, 24 * 12
capacity_likelihood = likelihoods.CapacityLikelihood(config, relevant_activity_types, activity_types, activity_facilities, facility_capacities, activity_times, min_time, max_time, bins)
capacity_likelihood.initialize()

joint_likelihood = likelihoods.JointLikelihood()
joint_likelihood.add_likelihood(distance_likelihood, 1.0)
joint_likelihood.add_likelihood(capacity_likelihood, 0.0)

acceptance_count = 0
acceptance = []
likelihood = []
excess = []

distances = { c : [] for c in distance_likelihood.categories }

interval = 1000

sampler = sampler.Sampler(joint_likelihood, proposal_distribution)
for i in tqdm(range(int(1e6))):
    accepted = sampler.run_sample()
    if accepted: acceptance_count += 1

    if i % interval == 0:
        acceptance.append(acceptance_count / interval)
        acceptance_count = 0
        likelihood.append(joint_likelihood.get_likelihood())
        excess.append(capacity_likelihood.get_excess_count())

        for c in distance_likelihood.categories:
            mean_distances = distance_likelihood.get_means()
            distances[c].append( mean_distances[c] ) # - reference_means[t] )

#writer = PopulationWriter(config)
#writer.write(
#    "/home/sebastian/static_locchoice/consistent/ch_1/population.xml.gz",
#    "output/population.xml.gz",
#    activity_facilities, facility_ids)

with open("output/plotdata.p", "wb+") as f:
    pickle.dump((distances, excess, likelihood, acceptance), f)

import matplotlib.pyplot as plt

colors = ["r", "g", "b", "c", "m", "y", "k"]

if distance_likelihood.use_modes:
    for m in range(len(constant.MODES)):
        plt.figure(figsize = (12,8))
        for t, color in zip(relevant_activity_types, colors):
            t = constant.ACTIVITY_TYPES_TO_INDEX[t]
            plt.plot(distances[(m,t)], label = constant.ACTIVITY_TYPES[t], color = color)
            plt.plot([0, len(distances[(m,t)])], [reference_means[(constant.MODES[m], constant.ACTIVITY_TYPES[t])], reference_means[(constant.MODES[m], constant.ACTIVITY_TYPES[t])]], "--", color = color)
        plt.title(constant.MODES[m])
        plt.legend()
        plt.savefig("output/%s.png" % constant.MODES[m])
        plt.close()
else:
    plt.figure(figsize = (12,8))
    for t, color in zip(relevant_activity_types, colors):
        t = constant.ACTIVITY_TYPES_TO_INDEX[t]
        plt.plot(distances[t], label = constant.ACTIVITY_TYPES[t], color = color)
        plt.plot([0, len(distances[t])], [reference_means[constant.ACTIVITY_TYPES[t]], reference_means[constant.ACTIVITY_TYPES[t]]], "--", color = color)
    plt.legend()
    plt.savefig("output/distances.png")
    plt.close()

plt.figure(figsize = (12,8))
plt.plot(excess, label = "Excess Occupancy")
plt.legend()
plt.savefig("output/occupancy.png")
plt.close()

plt.figure(figsize = (12,8))
plt.subplot(2,1,1)
plt.plot(likelihood, label = "Likelihood")
plt.legend()
plt.subplot(2,1,2)
plt.plot(acceptance, label = "Acceptance")
plt.legend()
plt.savefig("output/stats.png")
plt.close()
