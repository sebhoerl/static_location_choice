from facilities import FacilityReader
from population import PopulationReader, PopulationWriter
import constant, sampler, proposals, likelihoods
import reference
import pickle
import sys
import json

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.neighbors import KDTree

import numpy as np
from tqdm import tqdm

if len(sys.argv) < 2:
    print("No config")
    exit()
else:
    config = json.load(open(sys.argv[1]))

facility_ids, facility_coordinates, facility_capacities = FacilityReader(config).read(config["source_facilities_path"])
facility_id_to_index = { facility_id : index for index, facility_id in enumerate(facility_ids) }

activity_types, activity_modes, activity_facilities, activity_end_times, activity_start_times = PopulationReader(config).read(
    config["source_population_path"],
    facility_id_to_index
)

additional_activity_types = config["additional_activity_types"]
relevant_activity_types = config["relevant_activity_types"]
relevant_modes = config["relevant_modes"]

facility_capacities = facility_capacities.astype(np.float) * config["capacity_scaling_factor"]
facility_capacities = np.ceil(facility_capacities)

# Random initialization
if config["distribution_mode"] in ("close", "mixed"):
    facility_type_masks = [facility_capacities[i] > 0 for i in range(len(constant.ACTIVITY_TYPES))]
    facility_type_indices = [np.where(m)[0] for m in facility_type_masks]
    facility_type_trees = [KDTree(facility_coordinates[facility_type_indices[t]]) for t in range(len(constant.ACTIVITY_TYPES))]

for t in tqdm(relevant_activity_types, desc = "Initialization"):
    type_indices = np.where(facility_capacities[constant.ACTIVITY_TYPES_TO_INDEX[t],:] > 0)[0]
    type_mask = activity_types == constant.ACTIVITY_TYPES_TO_INDEX[t]

    if config["distribution_mode"] == "random":
        activity_facilities[type_mask] = np.random.choice(type_indices, np.sum(type_mask))
    elif config["distribution_mode"] == "singleton":
        activity_facilities[type_mask] = type_indices[0]
    elif config["distribution_mode"] == "close":
        for i in np.where(type_mask)[0]:
            index = facility_type_trees[activity_types[i]].query(facility_coordinates[activity_facilities[i]].reshape(1, -1), k = 1, return_distance = False)
            activity_facilities[i] = facility_type_indices[constant.ACTIVITY_TYPES_TO_INDEX[t]][index]
    elif config["distribution_mode"] == "mixed":
        for m in relevant_modes:
            selector_mask = type_mask & (activity_modes == constant.MODES_TO_INDEX[m])

            if m in ("walk", "bike"):
                for i in np.where(selector_mask)[0]:
                    index = facility_type_trees[activity_types[i]].query(facility_coordinates[activity_facilities[i]].reshape(1, -1), k = 1, return_distance = False)
                    activity_facilities[i] = facility_type_indices[constant.ACTIVITY_TYPES_TO_INDEX[t]][index]
            else:
                activity_facilities[selector_mask] = np.random.choice(type_indices, np.sum(selector_mask))
    else:
        raise RuntimeError("No distribution mode selected")


if config["proposal"] == "advanced":
    census_distances = reference.get_crowfly_distances()
    proposal_distribution = proposals.DistanceSamplingProposal(config, relevant_activity_types, activity_types, activity_modes, facility_capacities, relevant_modes, facility_coordinates, activity_facilities, census_distances)
else:
    proposal_distribution = proposals.RandomProposalDistribution(relevant_activity_types, activity_types, activity_modes, facility_capacities, relevant_modes)

reference_means, reference_variances = reference.get_by_purpose()
reference_means, reference_variances = reference.get_by_mode_and_purpose()

distance_likelihood = likelihoods.DistanceLikelihood(config, relevant_activity_types + additional_activity_types, activity_facilities, activity_modes, activity_types, activity_start_times, facility_coordinates, reference_means, reference_variances, relevant_modes)
distance_likelihood.initialize()

min_time, max_time, bins = config["minimum_time"], config["maximum_time"], config["time_bins"]
capacity_likelihood = likelihoods.CapacityLikelihood(config, relevant_activity_types, activity_types, activity_facilities, facility_capacities, activity_end_times, activity_start_times, min_time, max_time, bins)
capacity_likelihood.initialize()

joint_likelihood = likelihoods.JointLikelihood()
joint_likelihood.add_likelihood(distance_likelihood, 1.0)
joint_likelihood.add_likelihood(capacity_likelihood, 1.0)

acceptance_count = 0
acceptance = []
likelihood = []
excess = []

distances = { c : [] for c in distance_likelihood.categories }

sampler = sampler.Sampler(config, joint_likelihood, proposal_distribution, activity_facilities)
for i in tqdm(range(int(config["total_iterations"])), desc = "Sampling locations"):
    accepted = sampler.run_sample()
    if accepted: acceptance_count += 1

    if config["validation_interval"] is not None and i % config["validation_interval"] == 0:
        current_likelihood = capacity_likelihood.get_likelihood()
        validation_likelihood = capacity_likelihood.compute_validation_likelihood()
        if abs(current_likelihood - validation_likelihood) > 1e-12:
            raise AssertionError((current_likelihood, validation_likelihood, abs(current_likelihood - validation_likelihood)))

    if i % config["measurement_interval"] == 0:
        acceptance.append(acceptance_count / config["measurement_interval"])
        acceptance_count = 0

        likelihood.append(joint_likelihood.get_likelihood())
        excess.append(capacity_likelihood.get_excess_count())

        for c in distance_likelihood.categories:
            mean_distances = distance_likelihood.get_means()
            distances[c].append( mean_distances[c] ) # - reference_means[t] )

    if i % config["output_interval"] == 0 and i > 0:
        with open("%s/plotdata.p" % config["output_path"], "wb+") as f:
            pickle.dump((distances, excess, likelihood, acceptance), f)

        colors = cm.rainbow(np.linspace(0,1,len(relevant_activity_types + additional_activity_types)))
        #colors = ["r", "g", "b", "c", "m", "y", "k", (0.5, 0.5, 0.5)]

        if distance_likelihood.use_modes:
            for m in relevant_modes:
                m = constant.MODES_TO_INDEX[m]
                plt.figure(figsize = (12,8))
                for t, color in zip(relevant_activity_types + additional_activity_types, colors):
                    t = constant.ACTIVITY_TYPES_TO_INDEX[t]
                    plt.plot(distances[(m,t)], label = constant.ACTIVITY_TYPES[t], color = color)
                    plt.plot([0, len(distances[(m,t)]) - 1], [reference_means[(constant.MODES[m], constant.ACTIVITY_TYPES[t])], reference_means[(constant.MODES[m], constant.ACTIVITY_TYPES[t])]], "--", color = color)
                plt.title(constant.MODES[m])
                plt.grid()
                plt.legend()
                plt.savefig("%s/%s.png" % (config["output_path"], constant.MODES[m]))
                plt.close()
        else:
            plt.figure(figsize = (12,8))
            for t, color in zip(relevant_activity_types + additional_activity_types, colors):
                t = constant.ACTIVITY_TYPES_TO_INDEX[t]
                plt.plot(distances[t], label = constant.ACTIVITY_TYPES[t], color = color)
                plt.plot([0, len(distances[t]) - 1], [reference_means[constant.ACTIVITY_TYPES[t]], reference_means[constant.ACTIVITY_TYPES[t]]], "--", color = color)
            plt.legend()
            plt.grid()
            plt.savefig("%s/distances.png" % config["output_path"])
            plt.close()

        plt.figure(figsize = (12,8))
        plt.plot(excess, label = "Excess Occupancy")
        plt.grid()
        plt.legend()
        plt.savefig("%s/occupancy.png" % config["output_path"])
        plt.close()

        plt.figure(figsize = (12,8))
        plt.subplot(2,1,1)
        plt.plot(likelihood, label = "Likelihood")
        plt.legend()
        plt.grid()
        plt.subplot(2,1,2)
        plt.plot(acceptance, label = "Acceptance")
        plt.legend()
        plt.grid()
        plt.savefig("%s/stats.png" % config["output_path"])
        plt.close()

    if config["output_population_interval"] is not None and i % config["output_population_interval"] == 0 and i > 0:
        with open("%s/population.p" % config["output_path"], "wb+") as f:
            pickle.dump(activity_facilities, f)

        writer = PopulationWriter(config)
        writer.write(
            config["source_population_path"],
            "%s/population.xml.gz" % config["output_path"],
            activity_facilities, facility_ids)
