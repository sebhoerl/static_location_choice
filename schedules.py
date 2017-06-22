import numpy as np

class Schedule:
    def get_temperature(self, iteration):
        raise NotImplementedError()

class ConstantSchedule:
    def __init__(self, temperature):
        self.temperature = temperature

    def get_temperature(self, iteration):
        return self.temperature

class ExponentialSchedule:
    def __init__(self, config):
        self.T0 = config["exponential_temperature"]["initial_temperature"]
        self.T1 = config["exponential_temperature"]["final_temperature"]
        self.N1 = config["exponential_temperature"]["duration"]

        self.alpha = np.log(self.T1 / self.T0) / self.N1

    def get_temperature(self, iteration):
        return self.T0 * np.exp(self.alpha * iteration)

class LinearSchedule:
    def __init__(self, config):
        self.T0 = config["linear_temperature"]["initial_temperature"]
        self.T1 = config["linear_temperature"]["final_temperature"]
        self.N1 = config["exponential_temperature"]["duration"]

        self.alpha = (self.T1 - self.T0) / self.N1

    def get_temperature(self, iteration):
        return self.T0 + self.alpha * iteration
