import gym
import gym_fsa_atari
import pygame
import numpy as np


def getchar():
    # Returns a single character from standard input
    import tty, termios, sys
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


env = gym.make("fsa-manipulator-v0")
keys_to_action = {
    'q': np.array([1, 0, 0, 0, 0]),
    'w': np.array([0, 1, 0, 0, 0]),
    'e': np.array([0, 0, 1, 0, 0]),
    'r': np.array([0, 0, 0, 1, 0]),
    't': np.array([0, 0, 0, 0, 1]),
    'a': np.array([-1, 0, 0, 0, 0]),
    's': np.array([0, -1, 0, 0, 0]),
    'd': np.array([0, 0, -1, 0, 0]),
    'f': np.array([0, 0, 0, -1, 0]),
    'g': np.array([0, 0, 0, 0, -1])
}
env.reset()
env.render()
observation, reward, done, info = env.step(np.array([0, 0, 0, 0, 0]))
while not done:
    ch = getchar()
    print(ch)
    if ch.strip() == '' or ch not in 'abcdefghijklmnopqrstuvwxyz':
        print('bye!')
        break
    else:
        if ch in keys_to_action:
            action = keys_to_action[ch]
        else:
            action = np.array([0, 0, 0, 0, 0])
        for _ in range(50):
            env.render()
            # action = np.array([0, 0, 0, 0, 0]) # [root, shoulder, elbow, wrist, grasp]
            observation, reward, done, info = env.step(action)
            print("reward:", reward, "done:", done)

