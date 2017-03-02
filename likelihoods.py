import numpy as np
import sampler, constant, utils
from tqdm import tqdm
import scipy.special
import itertools

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

        self.excess_count = None
        self.likelihood = None
        #self.p = 0.999

        self.alpha = 1e-3
        self.cache = None

    def initialize(self):
        cache = utils.load_cache("capacity_likelihood", self.config)

        if cache is None:
            progress = tqdm(total = len(self.relevant_activity_types) * self.bins, desc = "Building occupancy matrix")
            for t, ti in self.relevant_activity_types.items():
                type_mask = self.activity_types == t

                for k in range(self.bins):
                    bin_mask = self.activity_time_bins == k

                    for facility_index in self.activity_facilities[type_mask & bin_mask]:
                        self.occupancy[ti, facility_index, k] += 1

                    progress.update()

            self.excess_count = 0

            progress = tqdm(total = len(self.relevant_activity_types) * self.facility_capacities.shape[1], desc = "Counting valid occupancies")

            for t in range(len(self.relevant_activity_types)):
                for f in range(self.facility_capacities.shape[1]):
                    self.excess_count += np.sum(np.maximum(self.occupancy[t,f,:] - self.facility_capacities[t, f], 0))
                    progress.update()

            utils.save_cache("capacity_likelihood", (self.occupancy, self.excess_count), self.config)
        else:
            print("Loaded occupancy matrix from cache")
            self.occupancy, self.excess_count = cache

        self.likelihood = -self.alpha * self.excess_count

    def evaluate(self, change):
        activity_index, facility_index = change[0], change[1]

        old_capacity_limit = self.facility_capacities[self.activity_types[activity_index], self.activity_facilities[activity_index]]
        old_occupancy_count_before = self.occupancy[self.relevant_activity_types[self.activity_types[activity_index]], self.activity_facilities[activity_index], self.activity_time_bins[activity_index]]
        old_occupancy_count_after = old_occupancy_count_before - 1

        new_capacity_limit = self.facility_capacities[self.activity_types[activity_index], facility_index]
        new_occupancy_count_before = self.occupancy[self.relevant_activity_types[self.activity_types[activity_index]], facility_index, self.activity_time_bins[activity_index]]
        new_occupancy_count_after = new_occupancy_count_before + 1

        old_state_before = old_occupancy_count_before <= old_capacity_limit
        old_state_after = old_occupancy_count_after <= old_capacity_limit

        new_state_before = new_occupancy_count_before <= new_capacity_limit
        new_state_after = new_occupancy_count_after <= new_capacity_limit

        excess_count = self.excess_count

        excess_count -= max(0, old_occupancy_count_before - old_capacity_limit)
        excess_count -= max(0, new_occupancy_count_before - new_capacity_limit)

        excess_count += max(0, old_occupancy_count_after - old_capacity_limit)
        excess_count += max(0, new_occupancy_count_after - new_capacity_limit)

        likelihood = -self.alpha * excess_count

        self.cache = (activity_index, facility_index, excess_count, likelihood)

        return likelihood, self.likelihood

    def accept(self):
        if self.cache is None: raise RuntimeError()
        activity_index, facility_index, excess_count, likelihood = self.cache

        self.occupancy[self.relevant_activity_types[self.activity_types[activity_index]], self.activity_facilities[activity_index]] -= 1
        self.occupancy[self.relevant_activity_types[self.activity_types[activity_index]], facility_index] += 1
        self.activity_facilities[activity_index] = facility_index
        self.excess_count = excess_count
        self.likelihood = likelihood

    def reject(self):
        self.cache = None

    def get_likelihood(self):
        return self.likelihood

    def get_valid_percentage(self):
        return self.count_valid / self.count_total

    def get_excess_count(self):
        return self.excess_count

class DistanceLikelihood(sampler.Likelihood):
    def __init__(self, relevant_activity_types, activity_facilities, activity_modes, activity_types, facility_coordinates, references):
        self.use_modes = isinstance(list(references.keys())[0], tuple)

        self.relevant_activity_types = [constant.ACTIVITY_TYPES_TO_INDEX[a] for a in relevant_activity_types]
        self.activity_facilities = activity_facilities
        self.facility_coordinates = facility_coordinates
        self.activity_types = activity_types
        self.activity_modes = activity_modes

        if self.use_modes:
            self.categories = list(itertools.product(constant.MODES_TO_INDEX.values(), self.relevant_activity_types))
            self.references = { (constant.MODES_TO_INDEX[m], constant.ACTIVITY_TYPES_TO_INDEX[t]) : v for (m,t), v in references.items() }

            self.relevant_activity_mask_by_category = {
                (m, t) : ( (activity_modes == m) & (activity_types == t) )
                for m, t in self.categories
            }
        else:
            self.categories = self.relevant_activity_types
            self.references = { constant.ACTIVITY_TYPES_TO_INDEX[t] : v for t, v in references.items() }

            self.relevant_activity_mask_by_category = {
                t : ( (activity_modes != -1) & (activity_types == t) )
                for t in self.categories
            }


        self.relevant_activity_mask = np.zeros((len(activity_types)), dtype = np.bool)
        for t in self.relevant_activity_mask_by_category.values(): self.relevant_activity_mask |= t

        self.relevant_activity_indices = np.where(self.relevant_activity_mask)[0]
        self.relevant_activity_indices_by_category = {
            c : np.where(self.relevant_activity_mask_by_category[c])[0]
            for c in self.categories
        }

        self.relevant_activity_indices_set = set(list(self.relevant_activity_indices))

        self.distances = np.zeros((len(activity_facilities)), dtype = np.float)
        self.means = None

        self.cache = None
        self.likelihood = None

        self.sigma = { c : 1.0 / len(self.relevant_activity_indices_by_category[c]) for c in self.categories }

    def initialize(self):
        from_coordinates = self.facility_coordinates[self.activity_facilities[self.relevant_activity_indices - 1]]
        to_coordinates = self.facility_coordinates[self.activity_facilities[self.relevant_activity_indices]]

        self.distances[self.relevant_activity_indices] = np.sqrt(np.sum((from_coordinates - to_coordinates)**2, axis = 1)) / 1000.0
        self.means = { c : np.mean(self.distances[self.relevant_activity_mask_by_category[c]]) for c in self.categories }

    def _get_category(self, activity_index):
        return (self.activity_modes[activity_index], self.activity_types[activity_index]) if self.use_modes else self.activity_types[activity_index]

    def evaluate(self, change):
        change_means = { c : self.means[c] for c in self.categories }
        change_distances = []

        change_coord = self.facility_coordinates[change[1]]
        change_category = self._get_category(change[0])

        front_old_distance = self.distances[change[0]]
        front_new_distance = np.sqrt(np.sum((self.facility_coordinates[self.activity_facilities[change[0] - 1]] - change_coord)**2)) / 1000.0
        change_means[change_category] += (front_new_distance - front_old_distance) / len(self.relevant_activity_mask_by_category[change_category])

        change_distances.append((change[0], front_new_distance))

        if (change[0] + 1) in self.relevant_activity_indices_set:
            back_category = self._get_category(change[0] + 1)
            back_old_distance = self.distances[change[0] + 1]
            back_new_distance = np.sqrt(np.sum((self.facility_coordinates[self.activity_facilities[change[0] + 1]] - change_coord)**2)) / 1000.0
            change_means[back_category] += (back_new_distance - back_old_distance) / len(self.relevant_activity_mask_by_category[back_category])

            change_distances.append((change[0] + 1, back_new_distance))

        prior_likelihood = sum([-(self.means[c] - self.references[c])**2 / self.sigma[c] for c in self.categories])
        posterior_likelihood = sum([-(change_means[c] - self.references[c])**2 / self.sigma[c] for c in self.categories])

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
        means = { c : np.mean(distances[self.relevant_activity_mask_by_category[c]]) for c in self.categories }
        likelihood = sum([-(means[c] - self.references[c])**2 / self.sigma[c]  for c in self.categories])

        return likelihood
