import numpy as np
import sampler, constant, utils
from tqdm import tqdm
import scipy.special

class JointLikelihood(sampler.Likelihood):
    def __init__(self):
        self.likelihoods = []
        self.factors = []

        self.likelihood = 0.0

    def initialize(self):
        for l in self.likelihoods: l.initialize()
        self.likelihood = np.dot(self.factors, [l.get_likelihood() for l in self.likelihoods])

    def add_likelihood(self, likelihood, factor = 1.0):
        self.likelihoods.append(likelihood)
        self.factors.append(factor)

    def evaluate(self, change):
        values = np.array([l.evaluate(change) for l in self.likelihoods])

        before = np.sum(np.dot(self.factors, values[:,1]))
        after = np.sum(np.dot(self.factors, values[:,0]))

        return after, before

    def accept(self):
        for l in self.likelihoods: l.accept()
        self.likelihood = np.dot(self.factors, [l.get_likelihood() for l in self.likelihoods])

    def reject(self):
        for l in self.likelihoods: l.reject()

    def get_likelihood(self):
        return self.likelihood

class CapacityLikelihood(sampler.Likelihood):
    def __init__(self, config, relevant_activity_types, activity_types, activity_facilities, facility_capacities, activity_times, min_time, max_time, bins):
        self.relevant_activity_types = { constant.ACTIVITY_TYPES_TO_INDEX[a] : i for i, a in enumerate(relevant_activity_types)}

        self.config = config
        self.bins = bins

        self.activity_types = activity_types
        self.activity_facilities = activity_facilities
        self.facility_capacities = facility_capacities
        self.occupancy = np.zeros((len(relevant_activity_types), facility_capacities.shape[1], bins), dtype = np.int)
        self.activity_time_filter = (activity_times > min_time) & (activity_times < max_time)
        self.activity_time_bins = np.floor(((activity_times - min_time) / (max_time - min_time)) * bins).astype(np.int)
        self.activity_time_bins[self.activity_time_bins == self.bins] = self.bins - 1

        self.count_total = None
        self.count_valid = None
        self.likelihood = None
        #self.p = 0.999

        self.alpha = 5.0
        self.beta = 0.1

        self.cache = None

    def initialize(self):
        cache = None #utils.load_cache("capacity_likelihood", self.config)

        if cache is None:
            progress = tqdm(total = len(self.relevant_activity_types) * self.bins, desc = "Building occupancy matrix")
            for t, ti in self.relevant_activity_types.items():
                type_mask = self.activity_types == t

                for k in range(self.bins):
                    bin_mask = self.activity_time_bins == k

                    for facility_index in self.activity_facilities[type_mask & bin_mask]:
                        self.occupancy[ti, facility_index, k] += 1

                    progress.update()

            self.count_total = self.occupancy.shape[0] * self.occupancy.shape[1] * self.occupancy.shape[2]
            self.count_valid = 0

            progress = tqdm(total = len(self.relevant_activity_types) * self.facility_capacities.shape[1], desc = "Counting valid occupancies")

            for t in range(len(self.relevant_activity_types)):
                for f in range(self.facility_capacities.shape[1]):
                    self.count_valid += np.sum(self.occupancy[t,f,:] <= self.facility_capacities[t, f])
                    progress.update()

            #self.likelihood = self.count_valid * np.log(self.p) + (self.count_total - self.count_valid) * np.log(1.0 - self.p)
            utils.save_cache("capacity_likelihood", (self.occupancy, self.count_total, self.count_valid), self.config)
        else:
            print("Loaded occupancy matrix from cache")
            self.occupancy, self.count_total, self.count_valid = cache

        x = self.count_valid / self.count_total
        self.likelihood = (self.alpha - 1.0) * np.log(x) + (self.beta - 1.0) * np.log(1.0 - x) - scipy.special.beta(self.alpha, self.beta)

    def evaluate(self, change):
        activity_index, facility_index = change[0], change[1]

        old_capacity_limit = self.facility_capacities[self.activity_types[activity_index], self.activity_facilities[activity_index]]
        old_occupancy_count_before = self.occupancy[self.relevant_activity_types[self.activity_types[activity_index]], self.activity_facilities[activity_index], self.activity_time_bins[activity_index]]
        old_occupancy_count_after = old_occupancy_count_before - 1

        new_capacity_limit = self.facility_capacities[self.activity_types[activity_index], facility_index]
        new_occupancy_count_before = self.occupancy[self.relevant_activity_types[self.activity_types[activity_index]], facility_index, self.activity_time_bins[activity_index]]
        new_occupancy_count_after = old_occupancy_count_before + 1

        old_state_before = old_occupancy_count_before <= old_capacity_limit
        old_state_after = old_occupancy_count_after <= old_capacity_limit

        new_state_before = new_occupancy_count_before <= new_capacity_limit
        new_state_after = new_occupancy_count_after <= new_capacity_limit

        count_valid = self.count_valid

        if old_state_before and not old_state_after:
            count_valid -= 1

        if old_state_after and not old_state_before:
            count_valid += 1

        if new_state_before and not new_state_after:
            count_valid -= 1

        if new_state_after and not new_state_before:
            count_valid += 1

        x = count_valid / self.count_total
        likelihood = (self.alpha - 1.0) * np.log(x) + (self.beta - 1.0) * np.log(1.0 - x) - scipy.special.beta(self.alpha, self.beta)

        #likelihood = count_valid * np.log(self.p) + (self.count_total - count_valid) * np.log(1.0 - self.p)
        self.cache = (activity_index, facility_index, count_valid, likelihood)

        return likelihood, self.likelihood

    def accept(self):
        if self.cache is None: raise RuntimeError()
        activity_index, facility_index, count_valid, likelihood = self.cache

        self.activity_facilities[activity_index] = facility_index
        self.count_valid = count_valid
        self.likelihood = likelihood

    def reject(self):
        self.cache = None

    def get_likelihood(self):
        return self.likelihood

    def get_valid_percentage(self):
        return self.count_valid / self.count_total

class DistanceLikelihood(sampler.Likelihood):
    def __init__(self, relevant_activity_types, activity_facilities, activity_modes, activity_types, facility_coordinates, references):
        self.relevant_activity_types = [constant.ACTIVITY_TYPES_TO_INDEX[a] for a in relevant_activity_types]
        self.activity_facilities = activity_facilities
        self.facility_coordinates = facility_coordinates
        self.activity_types = activity_types

        self.references = { constant.ACTIVITY_TYPES_TO_INDEX[t] : v for t, v in references.items() }

        self.relevant_activity_mask_by_type = {
            t : ( (activity_modes != -1) & (activity_types == t) )
            for t in self.relevant_activity_types
        }

        self.relevant_activity_mask = np.zeros((len(activity_types)), dtype = np.bool)
        for t in self.relevant_activity_mask_by_type.values(): self.relevant_activity_mask |= t

        self.relevant_activity_indices = np.where(self.relevant_activity_mask)[0]
        self.relevant_activity_indices_by_type = { t : np.where(self.relevant_activity_mask_by_type[t])[0] for t in self.relevant_activity_types }

        self.relevant_activity_indices_set = set(list(self.relevant_activity_indices))

        self.distances = np.zeros((len(activity_facilities)), dtype = np.float)
        self.means = None

        self.cache = None
        self.likelihood = None

        self.sigma = { t : 1.0 / len(self.relevant_activity_indices_by_type[t]) for t in self.relevant_activity_types }

    def initialize(self):
        from_coordinates = self.facility_coordinates[self.activity_facilities[self.relevant_activity_indices - 1]]
        to_coordinates = self.facility_coordinates[self.activity_facilities[self.relevant_activity_indices]]

        self.distances[self.relevant_activity_indices] = np.sqrt(np.sum((from_coordinates - to_coordinates)**2, axis = 1)) / 1000.0
        self.means = { t : np.mean(self.distances[self.relevant_activity_mask_by_type[t]]) for t in self.relevant_activity_types }

    def evaluate(self, change):
        change_means = { t : self.means[t] for t in self.relevant_activity_types }
        change_distances = []

        change_type = self.activity_types[change[0]]
        change_coord = self.facility_coordinates[change[1]]

        front_old_distance = self.distances[change[0]]
        front_new_distance = np.sqrt(np.sum((self.facility_coordinates[self.activity_facilities[change[0] - 1]] - change_coord)**2)) / 1000.0
        change_means[change_type] += (front_new_distance - front_old_distance) / len(self.relevant_activity_indices_by_type[change_type])

        change_distances.append((change[0], front_new_distance))

        if (change[0] + 1) in self.relevant_activity_indices_set:
            back_type = self.activity_types[change[0] + 1]
            back_old_distance = self.distances[change[0] + 1]
            back_new_distance = np.sqrt(np.sum((self.facility_coordinates[self.activity_facilities[change[0] + 1]] - change_coord)**2)) / 1000.0
            change_means[back_type] += (back_new_distance - back_old_distance) / len(self.relevant_activity_indices_by_type[back_type])

            change_distances.append((change[0] + 1, back_new_distance))

        prior_likelihood = sum([-(self.means[t] - self.references[t])**2 / self.sigma[t]  for t in self.relevant_activity_types])
        posterior_likelihood = sum([-(change_means[t] - self.references[t])**2 / self.sigma[t]  for t in self.relevant_activity_types])

        self.cache = ( change[0], change[1], change_means, change_distances, posterior_likelihood )

        if self.likelihood is None: self.likelihood = prior_likelihood
        return posterior_likelihood, prior_likelihood

    def accept(self):
        activity_index, facility_index, change_means, change_distances, likelihood = self.cache

        self.likelihood = likelihood

        self.activity_facilities[activity_index] = facility_index
        self.means = change_means

        for activity_index, distance in change_distances:
            self.distances[activity_index] = distance

        self.cache = None

    def reject(self):
        self.cache = None

    def get_likelihood(self):
        return self.likelihood

    def get_means(self):
        return self.means

    def compute_plain_likelihood(self):
        distances = np.zeros((len(activity_facilities)), dtype = np.float)

        from_coordinates = self.facility_coordinates[self.activity_facilities[self.relevant_activity_indices - 1]]
        to_coordinates = self.facility_coordinates[self.activity_facilities[self.relevant_activity_indices]]

        distances[self.relevant_activity_indices] = np.sqrt(np.sum((from_coordinates - to_coordinates)**2, axis = 1)) / 1000.0
        means = { t : np.mean(distances[self.relevant_activity_mask_by_type[t]]) for t in self.relevant_activity_types }
        likelihood = sum([-(means[t] - self.references[t])**2 / self.sigma[t]  for t in self.relevant_activity_types])

        return likelihood
