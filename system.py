import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human") 
for i_episode in range(100):
    obs, info = env.reset()
    terminated = False
    truncated = False
    t = 0

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        t += 1

        if terminated or truncated:
            print(f"Episode {i_episode+1} finished after {t} timesteps")

env.close()
