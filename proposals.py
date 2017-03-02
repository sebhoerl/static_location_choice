import numpy as np
import sampler, constant

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
