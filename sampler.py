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
    def __init__(self, likelihood, proposal_distribution):
        self.likelihood = likelihood
        self.proposal_distribution = proposal_distribution

    def run_sample(self):
        changes, forward_probability, backward_probability = self.proposal_distribution.sample()

        likelihood_new, likelihood_previous = self.likelihood.evaluate(changes)

        a1 = likelihood_new - likelihood_previous
        a2 = forward_probability - backward_probability

        a = a1 + a2

        if a >= 0:# or np.log(np.random.random()) <= a:
            self.likelihood.accept()
            return True
        else:
            self.likelihood.reject()
            return False
