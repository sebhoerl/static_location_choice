import numpy as np

class Likelihood:
    def initialize(self):
        raise NotImplementedError()

    def evaluate(self, change):
        raise NotImplementedError()

    def accept(self):
        raise NotImplementedError()

    def reject(self):
        raise NotImplementedError()

class ProposalDistribution:
    def sample(self):
        raise NotImplementedError()

class Sampler:
    def __init__(self, config, likelihood, proposal_distribution, activity_facilities, schedule):
        #self.sampling_factor = config["sampling_factor"]
        self.likelihood = likelihood
        self.proposal_distribution = proposal_distribution
        self.activity_facilities = activity_facilities
        self.schedule = schedule
        self.iteration = 0

    def run_sample(self):
        change, forward_probability, backward_probability = self.proposal_distribution.sample()

        likelihood_new, likelihood_previous = self.likelihood.evaluate(change)

        a1 = likelihood_new - likelihood_previous
        a2 = forward_probability - backward_probability
        #E = np.exp(a1 / self.schedule.get_temperature(self.iteration))

        #a = a1 + a2
        self.iteration += 1

        a = 1.0 if a1 > 0.0 else np.exp(a1 / self.schedule.get_temperature(self.iteration))

        if a > np.random.random():
            self.likelihood.accept()
            self.activity_facilities[change[0]] = change[1]
            return True
        else:
            self.likelihood.reject()
            return False
