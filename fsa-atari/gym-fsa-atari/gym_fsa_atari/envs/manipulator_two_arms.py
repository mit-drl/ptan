import numpy as np
from gym import utils
from gym_fsa_atari.envs import mujoco_env
from mujoco_py.generated import const
from gym import spaces

class ManipulatorTwoArmsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.num_frames_till_end = None
        self.current_goal = -1
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'manipulator_two_arms.xml', 2)

    def to_goal(self):
        # print(self.get_body_com("ball"), self.get_site_com("box"))
        return self.get_body_com("ball") - self.get_site_com("box")

    def step(self, a):
        vec = self.to_goal()
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        print("dist from goal:", reward_dist, "| current goal:", self.current_goal)
        if reward_dist > -2e-2:
            print("GOT TO GOAL")
            self.current_goal += 1
            if self.current_goal == 1 and self.num_frames_till_end == None:
                self.num_frames_till_end = 50
            reward = 1
        else:
            reward = 0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = self.check_bounds("ball")
        done = done or self.check_bounds("box")
        if self.current_goal > 0:
            if self.num_frames_till_end > 1:
                self.num_frames_till_end -= 1
            else:
                done = True
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def check_bounds(self, body):
        body_pos = self.get_body_com(body).copy()
        body_pos[2] = 0
        body_dist_1 = np.linalg.norm(body_pos)
        body_dist_2 = np.linalg.norm(body_pos-np.array([0.4, 0, 0]))
        print(body, self.get_body_com(body))
        if body_dist_1 > 0.545 and body_dist_2 > 0.545:
            print(body + " went out of bounds!")
            return True
        return False

    def viewer_setup(self):
        self.viewer.cam.type = const.CAMERA_FIXED
        self.viewer.cam.fixedcamid = 0
        # self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        self.randomize_location_circle('ball', radius=0.54, z=0.4)
        self.randomize_location_circle('box', radius=0.54, x=0.4, y=0, z=0.4)
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

    def get_logic_state(self):
        dist = np.linalg.norm(self.get_body_com("ball")-self.get_site_com("palm_touch"))
        dist_between_thumbs = np.linalg.norm(self.get_body_com("thumbtip")-self.get_body_com("fingertip"))
        ball_in_hand = 1 if dist < 0.03 else 0
        ball_gripped = 1 if ball_in_hand and dist_between_thumbs < 0.03 else 0
        goal_A = 0 if self.current_goal < 1 else 1
        goal_B = 0 if self.current_goal < 2 else 1
        return (ball_in_hand, ball_gripped, goal_A, goal_B)

    # returns the logic's observation space
    def get_logic_observation_space(self):
        return spaces.MultiDiscrete([2, 2, 2, 2])
