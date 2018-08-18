import numpy as np

from gym import Env, spaces
from gym.utils import seeding

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class FsaDiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None # for rendering
        self.nS = nS
        self.nA = nA
        self.nL = 3

        self.action_space = spaces.Discrete(self.nA)
        image_space = spaces.Discrete(self.nS / (self.nL - 1))
        logic_space = spaces.Discrete(self.nL)
        self.observation_space = spaces.Tuple((image_space, logic_space))

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        image = self.s // (self.nL - 1)
        logic = self.s % (self.nL - 1)
        self.lastaction = None
        return (image, logic)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction=a

        image = self.s // (self.nL - 1)
        logic = self.s % (self.nL - 1)
        full_state = (image, logic)
        return (full_state, r, d, {"prob" : p})