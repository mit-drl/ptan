import gym
import gym_fsa_atari
import time
env = gym.make("fsa-Taxi-v2")
env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # take a random action
    observation, reward, done, info = env.step(action)
    time.sleep(0.1)