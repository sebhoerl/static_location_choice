import numpy as np
import sampler, constant
from sklearn.neighbors import KDTree

class DistanceSamplingProposal(sampler.ProposalDistribution):
    def __init__(self, config, relevant_activity_types, activity_types, activity_modes, facility_capacities, relevant_modes, facility_coordinates, activity_facilities, census_distances):
        census_distances, _, census_mode_distances = census_distances

        self.census_distances = { (constant.MODES_TO_INDEX[c[0]], constant.ACTIVITY_TYPES_TO_INDEX[c[1]]) : np.array(d, dtype=np.float) * 1e3 for c, d in census_distances.items() }
        for m, d in census_mode_distances.items(): self.census_distances[constant.MODES_TO_INDEX[m]] = np.array(d, dtype=np.float) * 1e3

        self.census_distance_counts = { k : len(d) for k, d in self.census_distances.items() }

        self.facility_coordinates = facility_coordinates
        self.activity_facilities = activity_facilities

        self.distance_based_proposal_candidate_set_size = config["distance_based_proposal_candidate_set_size"]

        self.activity_types = activity_types
        self.activity_modes = activity_modes

        self.relevant_activity_mask = np.zeros((len(activity_types)), dtype = np.bool)
        self.relevant_mode_mask = np.zeros((len(activity_types)), dtype = np.bool)

        for t in relevant_activity_types: self.relevant_activity_mask |= activity_types == constant.ACTIVITY_TYPES_TO_INDEX[t]
        for m in relevant_modes: self.relevant_mode_mask |= activity_modes == constant.MODES_TO_INDEX[m]

        self.relevant_activity_indices = np.where(self.relevant_activity_mask)[0]
        self.relevant_mode_indices = np.where(self.relevant_mode_mask)[0]
        self.relevant_indices = np.where(self.relevant_activity_mask & self.relevant_mode_mask)[0]

        self.facility_type_masks = [facility_capacities[i] > 0 for i in range(len(constant.ACTIVITY_TYPES))]
        self.facility_type_indices = [np.where(m)[0] for m in self.facility_type_masks]

        self.trees = [KDTree(self.facility_coordinates[self.facility_type_indices[t]]) for t in range(len(constant.ACTIVITY_TYPES))]

    def _get_coords(self, center1, center2, distance1, distance2):
        distance1 = max(distance1, 1.0)
        distance2 = max(distance2, 1.0)

        d = np.sqrt(np.sum((center1 - center2)**2))

        if not (d == 0 or d < abs(distance1 - distance2) or d > distance1 + distance2):
            a = 0.5 * (distance1**2 - distance2**2 + d**2) / d
            h = np.sqrt(distance1**2 - a**2)

            p2 = center1 + a * (center2 - center1) / d

            return [(
                    p2[0] + h * (center2[1] - center1[1]) / d,
                    p2[1] - h * (center2[0] - center1[0]) / d
                ), (
                    p2[0] - h * (center2[1] - center1[1]) / d,
                    p2[1] + h * (center2[0] - center1[0]) / d
                )]
        else:
            f = distance1 / (distance1 + distance2)
            return [center1 + (center2 - center1) * f]

    def sample(self):
        choice_index = np.random.randint(len(self.relevant_indices))
        activity_index = self.relevant_indices[choice_index]
        #activity_index = np.random.choice(self.relevant_indices)
        candidate_indices = self.facility_type_indices[self.activity_types[activity_index]]

        current_type = self.activity_types[activity_index]
        current_mode = self.activity_modes[activity_index]

        following_type = self.activity_types[activity_index + 1]
        following_mode = self.activity_modes[activity_index + 1]

        if following_type == constant.ACTIVITY_TYPES_TO_INDEX["home"]:
            following_type = current_type

        preceeding_coord = self.facility_coordinates[self.activity_facilities[activity_index - 1]]
        following_coord = self.facility_coordinates[self.activity_facilities[activity_index + 1]]

        preceeding_category = (current_mode, current_type)
        following_category = (following_mode, following_type)

        if not preceeding_category in self.census_distances: preceeding_category = current_mode
        if not following_category in self.census_distances: following_category = following_mode

        preceeding_choice_index = np.random.randint(self.census_distance_counts[preceeding_category])
        following_choice_index = np.random.randint(self.census_distance_counts[following_category])

        preceeding_distance = self.census_distances[preceeding_category][preceeding_choice_index]
        following_distance = self.census_distances[following_category][following_choice_index]

        #preceeding_distance = np.random.choice(self.census_distances[preceeding_category]) if preceeding_category in self.census_distances else np.random.choice(self.census_mode_distances[constant.MODES[current_mode]])
        #following_distance = np.random.choice(self.census_distances[following_category]) if following_category in self.census_distances else np.random.choice(self.census_mode_distances[constant.MODES[following_mode]])

        coords = self._get_coords(preceeding_coord, following_coord, preceeding_distance, following_distance)

        #indices = [candidate_indices[np.argmin(np.sum((self.facility_coordinates[candidate_indices] - coord)**2, axis = 1))] for coord in coords]
        #facility_index = np.random.choice(indices)

        facility_index = np.random.choice(candidate_indices[self.trees[current_type].query(coords, k = self.distance_based_proposal_candidate_set_size, return_distance = False)].flatten())
        return (activity_index, facility_index), 0.0, 0.0

class RandomProposalDistribution(sampler.ProposalDistribution):
    def __init__(self, relevant_activity_types, activity_types, activity_modes, facility_capacities, relevant_modes = ["car", "pt", "bike", "walk"]):
        self.activity_types = activity_types

        self.relevant_activity_mask = np.zeros((len(activity_types)), dtype = np.bool)
        self.relevant_mode_mask = np.zeros((len(activity_types)), dtype = np.bool)

        for t in relevant_activity_types: self.relevant_activity_mask |= activity_types == constant.ACTIVITY_TYPES_TO_INDEX[t]
        for m in relevant_modes: self.relevant_mode_mask |= activity_modes == constant.MODES_TO_INDEX[m]

        self.relevant_activity_indices = np.where(self.relevant_activity_mask)[0]
        self.relevant_mode_indices = np.where(self.relevant_mode_mask)[0]

        self.relevant_indices = np.where(self.relevant_activity_mask & self.relevant_mode_mask)[0]

        self.facility_type_masks = [facility_capacities[i] > 0 for i in range(len(constant.ACTIVITY_TYPES))]
        self.facility_type_indices = [np.where(m)[0] for m in self.facility_type_masks]

    def sample(self):
        activity_index = np.random.choice(self.relevant_indices)
        activity_type = self.activity_types[activity_index]
        facility_index = np.random.choice(self.facility_type_indices[activity_type])
        return (activity_index, facility_index), 0.0, 0.0
