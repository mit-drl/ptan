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

game = "fsa-manipulator-two-arms-v0"

env = gym.make(game)
env.reset()
'''
for i in range(20000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(env.get_logic_state())
    if done:
        break
exit()
'''
if game == "fsa-manipulator-v0":
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
    do_nothing = np.array([0, 0, 0, 0, 0])
elif game == "fsa-manipulator-two-arms-v0":
    keys_to_action = {
        'q': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'w': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        'e': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        'r': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        't': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        'a': np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        's': np.array([0, -1, 0, 0, 0, 0, 0, 0, 0, 0]),
        'd': np.array([0, 0, -1, 0, 0, 0, 0, 0, 0, 0]),
        'f': np.array([0, 0, 0, -1, 0, 0, 0, 0, 0, 0]),
        'g': np.array([0, 0, 0, 0, -1, 0, 0, 0, 0, 0]),
        'y': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'u': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        'i': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        'o': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        'p': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        'h': np.array([0, 0, 0, 0, 0, -1, 0, 0, 0, 0]),
        'j': np.array([0, 0, 0, 0, 0, 0, -1, 0, 0, 0]),
        'k': np.array([0, 0, 0, 0, 0, 0, 0, -1, 0, 0]),
        'l': np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, 0]),
        ';': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
    }
    do_nothing = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

env.render()
observation, reward, done, info = env.step(do_nothing)
while not done:
    ch = getchar()
    print(ch)
    if ch.strip() == '' or ch not in 'abcdefghijklmnopqrstuvwxyz;[]\'':
        print('bye!')
        break
    else:
        if ch in keys_to_action:
            action = keys_to_action[ch]
        else:
            action = do_nothing
        for _ in range(25):
            env.render()
            # action = np.array([0, 0, 0, 0, 0]) # [root, shoulder, elbow, wrist, grasp]
            observation, reward, done, info = env.step(action)
            print("reward:", reward)
            print(env.get_logic_state())

