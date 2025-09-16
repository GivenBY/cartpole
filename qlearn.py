import gymnasium as gym
import numpy as np
import random
import math

env = gym.make('CartPole-v1', render_mode='human') 
no_buckets = (1, 1, 6, 3)
no_actions = env.action_space.n

state_value_bounds = list(zip(env.observation_space.low, env.observation_space.high))
print(state_value_bounds[3])
state_value_bounds[1] = (-0.5, 0.5)
state_value_bounds[3] = (-math.radians(50), math.radians(50))
print(state_value_bounds[3])
q_value_table = np.zeros(no_buckets + (no_actions,))

min_explore_rate = 0.1
min_learning_rate = 0.1
max_episodes = 1000
max_time_steps = 250
streak_to_end = 120
solved_time = 199
discount = 0.99
no_streaks = 0

def bucketize(state_value):
    bucket_indices = []
    for i in range(len(state_value)):
        if state_value[i] <= state_value_bounds[i][0]:
            bucket_index = 0
        elif state_value[i] >= state_value_bounds[i][1]:
            bucket_index = no_buckets[i] - 1
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (no_buckets[i]-1) * state_value_bounds[i][0] / bound_width
            scaling = (no_buckets[i]-1) / bound_width
            bucket_index = int(round(scaling * state_value[i] - offset))
        bucket_indices.append(bucket_index)
    return tuple(bucket_indices)

def select_explore_rate(episode):
    return max(min_explore_rate, min(1.0, 1.0 - math.log10((episode+1)/25)))

def select_learning_rate(episode):
    return max(min_learning_rate, min(1.0, 1.0 - math.log10((episode+1)/25)))

def select_action(state_value, explore_rate):
    if random.random() < explore_rate:
        return env.action_space.sample()
    else:
        return np.argmax(q_value_table[state_value])

for episode_no in range(max_episodes):
    explore_rate = select_explore_rate(episode_no)
    learning_rate = select_learning_rate(episode_no)

    observation, info = env.reset()
    state_value = bucketize(observation)
    previous_state_value = state_value
    done = False
    time_step = 0

    while not done:
        action = select_action(previous_state_value, explore_rate)
        observation, reward_gain, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state_value = bucketize(observation)

        best_q_value = np.max(q_value_table[state_value])
        q_value_table[previous_state_value][action] += learning_rate * (
            reward_gain + discount * best_q_value - q_value_table[previous_state_value][action]
        )

        previous_state_value = state_value
        time_step += 1

        if done or time_step >= max_time_steps:
            break

    if time_step >= solved_time:
        no_streaks += 1
    else:
        no_streaks = 0

    if no_streaks > streak_to_end:
        print(f'CartPole problem solved after {episode_no} episodes!')
        break

env.close()
print("Training completed.")
