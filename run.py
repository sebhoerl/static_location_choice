from facilities import FacilityReader
from population import PopulationReader, PopulationWriter
import constant, sampler, proposals, likelihoods
import reference

import numpy as np
from tqdm import tqdm

config = dict(
    cache_path = "cache"
)

facility_ids, facility_coordinates, facility_capacities = FacilityReader(config).read("/home/sebastian/static_locchoice/consistent/ch_1/facilities.xml.gz")
facility_id_to_index = { facility_id : index for index, facility_id in enumerate(facility_ids) }

person_ids, person_indices, activity_types, activity_modes, activity_facilities, activity_indices = PopulationReader(config).read(
    "/home/sebastian/static_locchoice/consistent/ch_1/population.xml.gz",
    facility_id_to_index
)

relevant_activity_types = ["shop", "leisure", "escort_kids", "escort_other", "remote_work"]

proposal_distribution = proposals.RandomProposalDistribution(relevant_activity_types, activity_types, facility_capacities)
references = reference.get_by_purpose()

distance_likelihood = likelihoods.DistanceLikelihood(relevant_activity_types, activity_facilities, activity_modes, activity_types, facility_coordinates, references)
distance_likelihood.initialize()

acceptance_count = 0
acceptance = []
likelihood = []

distances = { t : [] for t in relevant_activity_types }

sampler = sampler.Sampler(distance_likelihood, proposal_distribution)
for i in tqdm(range(int(2e5))):
    accepted = sampler.run_sample()
    if accepted: acceptance_count += 1

    if i % 1000 == 0:
        acceptance.append(acceptance_count / (i + 1))
        likelihood.append(distance_likelihood.get_likelihood())

        for t in relevant_activity_types:
            mean_distances = distance_likelihood.get_means()
            distances[t].append( mean_distances[constant.ACTIVITY_TYPES_TO_INDEX[t]] ) # - references[t] )

#writer = PopulationWriter(config)
#writer.write(
#    "/home/sebastian/static_locchoice/consistent/ch_1/population.xml.gz",
#    "output/population.xml.gz",
#    activity_facilities, facility_ids)

import matplotlib.pyplot as plt

plt.figure()

for t in relevant_activity_types:
    plt.plot(distances[t], label = t)

plt.legend()
plt.savefig("output/distances.png")
plt.close()

plt.figure()
plt.subplot(2,1,1)
plt.plot(likelihood, label = "Likelihood")
plt.legend()
plt.subplot(2,1,2)
plt.plot(acceptance, label = "Acceptance")
plt.legend()
plt.savefig("output/stats.png")
plt.close()
