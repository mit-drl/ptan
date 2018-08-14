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
        return self.get_site_com("ball_tip") - self.get_site_com("box")

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
        ball_dist1 = np.linalg.norm(self.get_body_com("ball")-self.get_site_com("palm_touch"))
        ball_dist2 = np.linalg.norm(self.get_body_com("ball") - self.get_site_com("palm_touch2"))
        box_dist1 = np.linalg.norm(self.get_body_com("box_pommel") - self.get_site_com("palm_touch"))
        box_dist2 = np.linalg.norm(self.get_body_com("box_pommel") - self.get_site_com("palm_touch2"))
        dist_between_thumbs1 = np.linalg.norm(self.get_body_com("thumbtip")-self.get_body_com("fingertip"))
        dist_between_thumbs2 = np.linalg.norm(self.get_body_com("thumbtip2") - self.get_body_com("fingertip2"))
        ball_in_hand1 = 1 if ball_dist1 < 0.03 else 0
        ball_in_hand2 = 1 if ball_dist2 < 0.03 else 0
        box_in_hand1 = 1 if box_dist1 < 0.03 else 0
        box_in_hand2 = 1 if box_dist2 < 0.03 else 0
        ball_gripped1 = 1 if ball_in_hand1 and dist_between_thumbs1 < 0.04 else 0
        ball_gripped2 = 1 if ball_in_hand2 and dist_between_thumbs2 < 0.04 else 0
        box_gripped1 = 1 if box_in_hand1 and dist_between_thumbs1 < 0.04 else 0
        box_gripped2 = 1 if box_in_hand2 and dist_between_thumbs2 < 0.04 else 0
        ball_box_dist = np.linalg.norm(self.to_goal())
        done = 1 if ball_box_dist < 2e-2 else 0
        return (ball_in_hand1, ball_in_hand2, box_in_hand1, box_in_hand2,
                ball_gripped1, ball_gripped2, box_gripped1, box_gripped2,
                done)

    # returns the logic's observation space
    def get_logic_observation_space(self):
        return spaces.MultiDiscrete([2, 2, 2, 2, 2, 2, 2, 2, 2])
