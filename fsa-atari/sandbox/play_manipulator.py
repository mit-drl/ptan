import gym
import gym_fsa_atari

env = gym.make("fsa-manipulator-v0")
env.reset()
for _ in range(3000):
    env.render()
    action = env.action_space.sample()  # take a random action
    observation, reward, done, info = env.step(action)
