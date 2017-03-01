import numpy as np
import sampler, constant

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
