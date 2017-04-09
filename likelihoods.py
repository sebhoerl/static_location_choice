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

stime = lambda x: "%02d:%02d:%02d" % (int(x) // 3600, (int(x) % 3600) // 60, int(x) % 60)

class CapacityLikelihood(sampler.Likelihood):
    def __init__(self, config, relevant_activity_types, activity_types, activity_facilities, facility_capacities, activity_end_times, activity_start_times, min_time, max_time, bins):
        self.relevant_activity_types = { constant.ACTIVITY_TYPES_TO_INDEX[a] : i for i, a in enumerate(relevant_activity_types)}

        self.config = config
        self.bins = bins

        self.activity_types = activity_types
        self.activity_facilities = activity_facilities
        self.facility_capacities = facility_capacities
        self.occupancy = np.zeros((len(relevant_activity_types), facility_capacities.shape[1], bins), dtype = np.int)

        end_times = np.copy(activity_end_times)
        end_times[end_times < 0.0] = np.inf

        start_times = np.copy(activity_start_times)
        start_times[start_times < 0.0] = 0.0

        start_bins = np.minimum(np.maximum(np.floor(((start_times - min_time) / (max_time - min_time)) * bins), 0), self.bins - 1).astype(np.int)
        end_bins = np.minimum(np.maximum(np.maximum(np.floor(((end_times - min_time) / (max_time - min_time)) * float(bins)), start_bins + 1), 0), self.bins - 1).astype(np.int)

        self.activity_time_indices = []

        for i in tqdm(range(len(activity_end_times)), desc = "Constructing activity time indices"):
            self.activity_time_indices.append(np.arange(start_bins[i], end_bins[i]))

        self.excess_count = None
        self.likelihood = None

        self.alpha = config["capacity_likelihood_alpha"]
        self.cache = None

    def initialize(self):
        cache = None #utils.load_cache("capacity_likelihood", self.config)

        if cache is None:
            progress = tqdm(total = len(self.relevant_activity_types), desc = "Building occupancy matrix")
            for t, ti in self.relevant_activity_types.items():
                type_indices = np.where(self.activity_types == t)[0]

                for i in type_indices:
                    self.occupancy[ti, self.activity_facilities[i], self.activity_time_indices[i]] += 1

                progress.update()
            progress.close()

            self.excess_count = 0

            progress = tqdm(total = len(self.relevant_activity_types) * self.facility_capacities.shape[1], desc = "Counting valid occupancies")

            for t, ti in self.relevant_activity_types.items():
                for f in range(self.facility_capacities.shape[1]):
                    self.excess_count += np.sum(np.maximum(self.occupancy[ti,f,:] - self.facility_capacities[t, f], 0))
                    progress.update()

            progress.close()

            #utils.save_cache("capacity_likelihood", (self.occupancy, self.excess_count), self.config)
        else:
            print("Loaded occupancy matrix from cache")
            self.occupancy, self.excess_count = cache

        self.likelihood = np.log(self.alpha) - self.alpha * self.excess_count

    def evaluate(self, change):
        activity_index, facility_index = change[0], change[1]

        if facility_index == self.activity_facilities[activity_index]:
            self.cache = (activity_index, facility_index, self.excess_count, self.likelihood)
            return self.likelihood, self.likelihood

        old_capacity_limit = self.facility_capacities[self.activity_types[activity_index], self.activity_facilities[activity_index]]
        old_occupancy_counts_before = self.occupancy[self.relevant_activity_types[self.activity_types[activity_index]], self.activity_facilities[activity_index], self.activity_time_indices[activity_index]]
        old_occupancy_counts_after = old_occupancy_counts_before - 1

        new_capacity_limit = self.facility_capacities[self.activity_types[activity_index], facility_index]
        new_occupancy_counts_before = self.occupancy[self.relevant_activity_types[self.activity_types[activity_index]], facility_index, self.activity_time_indices[activity_index]]
        new_occupancy_counts_after = new_occupancy_counts_before + 1

        excess_count = float(self.excess_count)

        excess_count -= np.sum(np.maximum(old_occupancy_counts_before - old_capacity_limit, 0))
        excess_count -= np.sum(np.maximum(new_occupancy_counts_before - new_capacity_limit, 0))

        excess_count += np.sum(np.maximum(old_occupancy_counts_after - old_capacity_limit, 0))
        excess_count += np.sum(np.maximum(new_occupancy_counts_after - new_capacity_limit, 0))

        likelihood = np.log(self.alpha) - self.alpha * excess_count

        self.cache = (activity_index, facility_index, excess_count, likelihood)

        return likelihood, self.likelihood

    def accept(self):
        if self.cache is None: raise RuntimeError()
        activity_index, facility_index, excess_count, likelihood = self.cache

        self.occupancy[self.relevant_activity_types[self.activity_types[activity_index]], self.activity_facilities[activity_index], self.activity_time_indices[activity_index]] -= 1
        self.occupancy[self.relevant_activity_types[self.activity_types[activity_index]], facility_index, self.activity_time_indices[activity_index]] += 1
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

    def compute_validation_likelihood(self):
        occupancy = np.zeros((len(self.relevant_activity_types), self.facility_capacities.shape[1], self.bins), dtype = np.int)

        for t, ti in self.relevant_activity_types.items():
            type_indices = np.where(self.activity_types == t)[0]

            for i in type_indices:
                occupancy[ti, self.activity_facilities[i], self.activity_time_indices[i]] += 1

        excess_count = 0

        for t, ti in self.relevant_activity_types.items():
            for f in range(self.facility_capacities.shape[1]):
                excess_count += np.sum(np.maximum(occupancy[ti,f,:] - self.facility_capacities[t, f], 0))

        return np.log(self.alpha) - self.alpha * excess_count

class QuantileLikelihood(sampler.Likelihood):
    def __init__(self, config, relevant_activity_types, activity_facilities, activity_modes, activity_types, activity_start_times, facility_coordinates, reference_data, relevant_modes = ["car", "pt", "bike", "walk"]):
        self.relevant_activity_types = [constant.ACTIVITY_TYPES_TO_INDEX[a] for a in relevant_activity_types]
        self.relevant_modes = [constant.MODES_TO_INDEX[m] for m in relevant_modes]

        self.activity_facilities = activity_facilities
        self.facility_coordinates = facility_coordinates
        self.activity_types = np.copy(activity_types)
        self.activity_modes = activity_modes

        home = constant.ACTIVITY_TYPES_TO_INDEX["home"]

        for i in range(1, len(activity_types)):
            if activity_types[i] == home and activity_types[i-1] in self.relevant_activity_types and activity_start_times[i] > -1:
                self.activity_types[i] = activity_types[i-1]

        self.categories = list(itertools.product(self.relevant_modes, self.relevant_activity_types))

        self.relevant_activity_mask_by_category = {
            (m, t) : ( (activity_modes == m) & (activity_types == t) & (activity_start_times > -1) )
            for m, t in self.categories
        }

        self.relevant_activity_mask = np.zeros((len(activity_types)), dtype = np.bool)
        for t in self.relevant_activity_mask_by_category.values(): self.relevant_activity_mask |= t

        self.relevant_activity_indices = np.where(self.relevant_activity_mask)[0]
        self.relevant_activity_indices_by_category = {
            c : np.where(self.relevant_activity_mask_by_category[c])[0]
            for c in self.categories
        }

        self.relevant_activity_indices_set = set(list(self.relevant_activity_indices))

        self.probabilities = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) / 100
        self.reference_data = { (m,a) : np.array(reference_data[0][(constant.MODES[m], constant.ACTIVITY_TYPES[a])]) for m, a in self.categories }

        self.distances = np.zeros((len(activity_facilities)), dtype = np.float)
        self.bounds = {}
        self.quantiles = {}
        self.population_counts = {}
        self.total_population_counts = {}
        self.reference_counts = {}
        self.total_reference_counts = {}

        self.cache = None
        self.likelihood = None

    def initialize(self):
        from_coordinates = self.facility_coordinates[self.activity_facilities[self.relevant_activity_indices - 1]]
        to_coordinates = self.facility_coordinates[self.activity_facilities[self.relevant_activity_indices]]

        self.distances[self.relevant_activity_indices] = np.sqrt(np.sum((from_coordinates - to_coordinates)**2, axis = 1)) / 1000.0
        self.distances_by_category = { c : self.distances[self.relevant_activity_indices_by_category[c]] for c in self.categories }

        self.quantiles = { c : np.percentile(self.reference_data[c], self.probabilities * 100) for c in self.categories }
        self.bounds = { c : np.insert(self.quantiles[c], 0, -1) for c in self.categories}

        self.reference_counts = { c :
            np.array([np.sum((lower < self.reference_data[c]) & (self.reference_data[c] <= upper)) for lower, upper in zip(self.bounds[c][:-1], self.bounds[c][1:])])
            for c in self.categories }

        self.population_counts = { c :
            np.array([np.sum((lower < self.distances_by_category[c]) & (self.distances_by_category[c] <= upper)) for lower, upper in zip(self.bounds[c][:-1], self.bounds[c][1:])])
            for c in self.categories }

        self.total_reference_counts = {
            c: np.sum(self.reference_counts[c])
            for c in self.categories }

        self.total_population_counts = {
            c: len(self.distances_by_category[c])
            for c in self.categories }

    def evaluate(self, change):
        updated_counts = { c : np.copy(self.population_counts[c]) for c in self.categories }
        distance_updates = []

        change_coord = self.facility_coordinates[change[1]]

        if change[0] in self.relevant_activity_indices_set:
            leading_category = (self.activity_modes[change[0]], self.activity_types[change[0]])

            leading_distance_current = self.distances[change[0]]
            leading_distance_update = np.sqrt(np.sum((self.facility_coordinates[self.activity_facilities[change[0] - 1]] - change_coord)**2)) / 1000.0

            current_quantile_index = np.sum(leading_distance_current > self.quantiles[leading_category])
            updated_quantile_index = np.sum(leading_distance_update > self.quantiles[leading_category])

            if current_quantile_index < len(self.probabilities): updated_counts[leading_category][current_quantile_index] -= 1
            if updated_quantile_index < len(self.probabilities): updated_counts[leading_category][updated_quantile_index] += 1
            distance_updates.append((change[0], leading_distance_update))

        if (change[0] + 1) in self.relevant_activity_indices_set:
            following_category = (self.activity_modes[change[0] + 1], self.activity_types[change[0] + 1])

            following_distance_current = self.distances[change[0] + 1]
            following_distance_update = np.sqrt(np.sum((self.facility_coordinates[self.activity_facilities[change[0] + 1]] - change_coord)**2)) / 1000.0

            current_quantile_index = np.sum(following_distance_current > self.quantiles[following_category])
            updated_quantile_index = np.sum(following_distance_update > self.quantiles[following_category])

            if current_quantile_index < len(self.probabilities): updated_counts[following_category][current_quantile_index] -= 1
            if updated_quantile_index < len(self.probabilities): updated_counts[following_category][updated_quantile_index] += 1
            distance_updates.append((change[0] + 1, following_distance_update))

        prior_likelihood = -np.max([-np.max(np.abs(self.population_counts[c] / self.total_population_counts[c] - self.reference_counts[c] / self.total_reference_counts[c])) for c in self.categories])
        posterior_likelihood = -np.max([-np.max(np.abs(updated_counts[c] / self.total_population_counts[c] - self.reference_counts[c] / self.total_reference_counts[c])) for c in self.categories])

        self.cache = ( change[0], change[1], updated_counts, distance_updates, posterior_likelihood )
        return posterior_likelihood, prior_likelihood

    def accept(self):
        activity_index, facility_index, updated_counts, distance_updates, likelihood = self.cache

        self.likelihood = likelihood
        self.population_counts = updated_counts
        for index, distance in distance_updates: self.distances[index] = distance

        self.cache = None

    def reject(self):
        self.cache = None

    def get_likelihood(self):
        return self.likelihood

    def compute_validation_likelihood(self):
        pass

class DistanceLikelihood(sampler.Likelihood):
    def __init__(self, config, relevant_activity_types, activity_facilities, activity_modes, activity_types, activity_start_times, facility_coordinates, reference_means, reference_variances, relevant_modes = ["car", "pt", "bike", "walk"]):
        self.use_modes = isinstance(list(reference_means.keys())[0], tuple)

        self.relevant_activity_types = [constant.ACTIVITY_TYPES_TO_INDEX[a] for a in relevant_activity_types]
        self.relevant_modes = [constant.MODES_TO_INDEX[m] for m in relevant_modes]

        self.activity_facilities = activity_facilities
        self.facility_coordinates = facility_coordinates
        self.activity_types = np.copy(activity_types)
        self.activity_modes = activity_modes

        home = constant.ACTIVITY_TYPES_TO_INDEX["home"]

        for i in range(1, len(activity_types)):
            if activity_types[i] == home and activity_types[i-1] in self.relevant_activity_types and activity_start_times[i] > -1:
                self.activity_types[i] = activity_types[i-1]

        if self.use_modes:
            self.categories = list(itertools.product(self.relevant_modes, self.relevant_activity_types))
            self.references = { (constant.MODES_TO_INDEX[m], constant.ACTIVITY_TYPES_TO_INDEX[t]) : v for (m,t), v in reference_means.items() }
            self.sigma2 = { (constant.MODES_TO_INDEX[m], constant.ACTIVITY_TYPES_TO_INDEX[t]) : v for (m,t), v in reference_variances.items() }

            self.relevant_activity_mask_by_category = {
                (m, t) : ( (activity_modes == m) & (activity_types == t) & (activity_start_times > -1) )
                for m, t in self.categories
            }

            for entry in config["override_sigma"]:
                m, a, value = entry

                if m == "*" and a == "*":
                    for ai in range(len(constant.ACTIVITY_TYPES)):
                        for mi in range(len(constant.MODES)):
                            self.sigma2[(mi, ai)] = value
                            print("Setting (%s, %s) to %f" % (constant.MODES[mi], constant.ACTIVITY_TYPES[ai], value))
                elif m == "*":
                    for mi in range(len(constant.MODES)):
                        self.sigma2[(mi, constant.ACTIVITY_TYPES_TO_INDEX[a])] = value
                        print("Setting (%s, %s) to %f" % (constant.MODES[mi], a, value))
                elif a == "*":
                    for ai in range(len(constant.ACTIVITY_TYPES)):
                        self.sigma2[(constant.MODES_TO_INDEX[m],ai)] = value
                        print("Setting (%s, %s) to %f" % (m, constant.ACTIVITY_TYPES[ai], value))
                else:
                    self.sigma2[(constant.MODES_TO_INDEX[m], constant.ACTIVITY_TYPES_TO_INDEX[a])] = value
                    print("Setting (%s, %s) to %f" % (m, a, value))
        else:
            self.categories = self.relevant_activity_types
            self.references = { constant.ACTIVITY_TYPES_TO_INDEX[t] : v for t, v in reference_means.items() }
            self.sigma2 = { constant.ACTIVITY_TYPES_TO_INDEX[t] : v for t, v in reference_variances.items() }

            self.relevant_activity_mask_by_category = {
                t : ( (activity_modes != -1) & (activity_types == t) )
                for t in self.categories
            }

            for o, v in config["override_sigma"].items():
                self.sigma2[constant.ACTIVITY_TYPES_TO_INDEX[o[1]]] = v

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
        change_means[change_category] += (front_new_distance - front_old_distance) / len(self.relevant_activity_indices_by_category[change_category])

        change_distances.append((change[0], front_new_distance))

        if (change[0] + 1) in self.relevant_activity_indices_set:
            back_category = self._get_category(change[0] + 1)
            back_old_distance = self.distances[change[0] + 1]
            back_new_distance = np.sqrt(np.sum((self.facility_coordinates[self.activity_facilities[change[0] + 1]] - change_coord)**2)) / 1000.0
            change_means[back_category] += (back_new_distance - back_old_distance) / len(self.relevant_activity_indices_by_category[back_category])

            change_distances.append((change[0] + 1, back_new_distance))

        prior_likelihood = sum([-(self.means[c] - self.references[c])**2 / (2 * self.sigma2[c]) - 0.5 * np.log(2 * self.sigma2[c] * np.pi) for c in self.categories])
        posterior_likelihood = sum([-(change_means[c] - self.references[c])**2 / (2 * self.sigma2[c]) - 0.5 * np.log(2 * self.sigma2[c] * np.pi) for c in self.categories])

        self.cache = ( change[0], change[1], change_means, change_distances, posterior_likelihood )

        if self.likelihood is None: self.likelihood = prior_likelihood
        return posterior_likelihood, prior_likelihood

    def accept(self):
        activity_index, facility_index, change_means, change_distances, likelihood = self.cache

        self.likelihood = likelihood
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

    def compute_validation_likelihood(self):
        distances = np.zeros((len(self.activity_facilities)), dtype = np.float)

        from_coordinates = self.facility_coordinates[self.activity_facilities[self.relevant_activity_indices - 1]]
        to_coordinates = self.facility_coordinates[self.activity_facilities[self.relevant_activity_indices]]

        distances[self.relevant_activity_indices] = np.sqrt(np.sum((from_coordinates - to_coordinates)**2, axis = 1)) / 1000.0
        means = { c : np.mean(distances[self.relevant_activity_mask_by_category[c]]) for c in self.categories }
        likelihood = sum([-(means[c] - self.references[c])**2 / (2 * self.sigma2[c]) - 0.5 * np.log(2 * self.sigma2[c] * np.pi) for c in self.categories])

        return likelihood
