import numpy as np
from gym import utils
from gym_fsa_atari.envs import mujoco_env

class ManipulatorEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.current_goal = -1
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'manipulator.xml', 2)

    def to_goal(self):
        if self.current_goal == 0:
            return self.get_body_com("ball") - self.get_body_com("target_ball")
        else:
            return self.get_body_com("ball") - self.get_body_com("target_ball_2")

    def step(self, a):
        vec = self.to_goal()
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        print("dist from goal:", reward_dist, "| current goal:", self.current_goal)
        if reward_dist > -1e-2:
            print("GOT TO GOAL")
            self.current_goal += 1
            reward = 1
        else:
            reward = 0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        if self.current_goal > 1:
            done = True
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.to_goal()
        ])