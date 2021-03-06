
import sys
from six import StringIO
from gym import utils
# from gym.envs.toy_text import discrete
from gym_fsa_atari.envs import fsadiscrete
import numpy as np

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

class FsaTaxiEnv(fsadiscrete.FsaDiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich
    rendering:
    - blue: passenger
    - magenta: destination
    - green: destination 2
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations

    THE GOAL: pick up passenger at blue location, drop off at magenta location,
    then go to the green location
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP,dtype='c')

        props = {"empty": 0, "dropped off": 1, "second dest": 2}

        self.locs = locs = [(0,0), (0,4), (4,0), (4,3)]

        nS = 6000
        nR = 5
        nC = 5
        maxR = nR-1
        maxC = nC-1
        isd = np.zeros(nS)
        nA = 6
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        TM = {}
        for row in range(5):
            for col in range(5):
                for passidx in range(5):
                    for destidx in range(4):
                        for destidx2 in range(4):
                            for dropoffs in range(3):
                                state = self.encode(row, col, passidx, destidx, destidx2, dropoffs)
                                prop = "empty"
                                # passidx == 4 implies passenger is in the car
                                if passidx < 4 and passidx != destidx and passidx != destidx2 and destidx != destidx2 and dropoffs < 1:
                                    isd[state] += 1
                                logic = state % (3 - 1) # the logic state; 3 is the number of logic states
                                if logic == 1:
                                    isd[state] = 0
                                for a in range(nA):
                                    # defaults
                                    newrow, newcol, newpassidx, newdropoffs = row, col, passidx, dropoffs
                                    reward = -1
                                    done = False
                                    taxiloc = (row, col)

                                    if a==0:
                                        newrow = min(row+1, maxR)
                                    elif a==1:
                                        newrow = max(row-1, 0)
                                    if a==2 and self.desc[1+row,2*col+2]==b":":
                                        newcol = min(col+1, maxC)
                                    elif a==3 and self.desc[1+row,2*col]==b":":
                                        newcol = max(col-1, 0)
                                    elif a==4: # pickup
                                        if (passidx < 4 and taxiloc == locs[passidx]):
                                            newpassidx = 4
                                        else:
                                            reward = -1
                                            # reward = -10
                                    elif a==5: # dropoff
                                        if (taxiloc == locs[destidx]) and passidx==4 and dropoffs == 0:
                                            newdropoffs = 1
                                            # reward = 20
                                            prop = "dropped off"
                                        elif (taxiloc == locs[destidx2]) and passidx==4 and dropoffs == 1:
                                            done = True
                                            newdropoffs = 2
                                            # reward = 20
                                            prop = "second dest"
                                        elif (taxiloc in locs) and passidx==4:
                                            newpassidx = locs.index(taxiloc)
                                        else:
                                            reward = -1
                                            # reward = -10
                                    newstate = self.encode(newrow, newcol, newpassidx, destidx, destidx2, newdropoffs)
                                    P[state][a].append((1.0, newstate, reward, done))
                                    TM[(state, newstate)] = prop
        isd /= isd.sum()
        fsadiscrete.FsaDiscreteEnv.__init__(self, nS, nA, P, props, TM, isd)

    def encode(self, taxirow, taxicol, passloc, destidx, destidx2, firstdropoff):
        # (5) 5, 5, 4, 4, 2
        i = taxirow
        i *= 5
        i += taxicol
        i *= 5
        i += passloc
        i *= 4
        i += destidx
        i *= 4
        i += destidx2
        i *= 3
        i += firstdropoff
        return i

    def decode(self, i):
        out = []
        out.append(i % 2)
        i = i // 3
        out.append(i % 4)
        i = i // 4
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxirow, taxicol, passidx, destidx, destidx2, firstdropoff = self.decode(self.s)
        def ul(x): return "_" if x == " " else x
        if passidx < 4:
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
            pi, pj = self.locs[passidx]
            out[1+pi][2*pj+1] = utils.colorize(out[1+pi][2*pj+1], 'blue', bold=True)
        else: # passenger in taxi
            out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'green', highlight=True)

        di, dj = self.locs[destidx]
        di2, dj2 = self.locs[destidx2]
        out[1+di][2*dj+1] = utils.colorize(out[1+di][2*dj+1], 'magenta')
        out[1 + di2][2 * dj2 + 1] = utils.colorize(out[1 + di2][2 * dj2 + 1], 'green')
        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile