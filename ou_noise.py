import numpy as np


class OrnsteinUhlenbeckNoise:
    """ Noise for Actor predictions """

    def __init__(self, action_space_size, mu=0, theta=0.5, sigma=0.2):
        self.action_space_size = action_space_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_space_size) * self.mu

    def get(self):
        self.state += self.theta * (self.mu - self.state) + self.sigma * np.random.rand(self.action_space_size)
        return self.state
