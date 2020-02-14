import math
import random
import numpy as np 
import gym 
import matplotlib.pyplot as plt

def build_state(quantized_state):
    return int("".join(map(lambda feature: str(int(feature)), quantized_state)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

class FeatureTransformer(object):
    def __init__(self, low_limits, high_limits, num_bins=9):
        self.target_pos_x_bins = np.linspace(low_limits[0], high_limits[0], num_bins)
        self.target_pos_y_bins = np.linspace(low_limits[1], high_limits[1], num_bins)
        self.theta_1_bins = np.linspace(low_limits[2], high_limits[2], num_bins)
        self.theta_2_bins = np.linspace(low_limits[3], high_limits[3], num_bins)

    def transform(self, observation):
        target_pos_x = observation[0]
        target_pos_y = observation[1]
        theta_1 = observation[2]
        theta_2 = observation[3]
        return build_state([to_bin(target_pos_x, self.target_pos_x_bins),
                            to_bin(target_pos_y, self.target_pos_y_bins),
                            to_bin(theta_1, self.theta_1_bins),
                            to_bin(theta_2, self.theta_2_bins)])

class QLearningAgent():
    def __init__(self, alpha=0.1, gamma=0.9, eps=0.95, eps_decay=0.99, eps_min=0.01, n_states=100000, n_actions=6, low_limits=None, high_limits=None):
        self.alpha = alpha 
        self.gamma = gamma 
        self.eps = eps 
        self.eps_decay = eps_decay 
        self.eps_min = eps_min

        self.ft = FeatureTransformer(low_limits, high_limits)
        self.action_space = [i for i in range(n_actions)]
        self.Q_table = np.zeros((n_states, n_actions))

    def decrement_epsilon(self):
        self.eps = self.eps * self.eps_decay 
        if self.eps < self.eps_min:
            self.eps = self.eps_min

    def epsilon_greedy(self, state):
        state_discrete = self.ft.transform(state)
        if self.eps > random.random():
            action = np.random.choice(self.action_space)
        else :
            action = np.argmax(self.Q_table[state_discrete,:])
        return action

    def learn(self, state, action, reward, next_state, done):
        state_discrete = self.ft.transform(state)
        next_state_discrete = self.ft.transform(next_state)
        current_Q = self.Q_table[state_discrete, action]

        if not done:
            max_future_Q = np.max(self.Q_table[next_state_discrete,:])
            Q_target = reward + self.gamma * max_future_Q
        else:
            Q_target = reward

        self.Q_table[state_discrete, action] = current_Q + self.alpha * (Q_target - current_Q)

if __name__ == "__main__":
    env = gym.make('gym_robot_arm:robot-arm-v0')
    
    high_limits = env.observation_space.high
    low_limits = env.observation_space.low
    n_actions = env.action_space.n
    # n_states = env.observation_space.shape[0]
    n_states = 10000

    agent = QLearningAgent(alpha=0.0001, gamma=0.9, eps_decay=0.99, n_states=n_states, n_actions=n_actions, low_limits=low_limits, high_limits=high_limits)

    n_episodes = 3000
    n_steps = 300
    total_rewards_hist = []
    avg_reward_hist = []
    for episode in range(1, n_episodes+1):
        state = env.reset()
        total_reward = 0
        for step in range(n_steps):
            if episode > 2900:
                env.render()
            action = agent.epsilon_greedy(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            total_reward += reward
            if done:
                break
        agent.decrement_epsilon()
        total_rewards_hist.append(total_reward)
        avg_reward = np.average(total_rewards_hist[-100:])
        avg_reward_hist.append(avg_reward)
        print("Episode :", episode, "Epsilon : {:.2f}".format(agent.eps), "Total Reward : {:.2f}".format(total_reward), "Avg Reward : {:.2f}".format(avg_reward))
    env.close()
    # Plot result
    fig, ax = plt.subplots()
    t = np.arange(n_episodes)
    ax.plot(t, total_rewards_hist, label='Total Reward')
    ax.plot(t, avg_reward_hist, label='Avg Reward')
    ax.set_title("Reward VS Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.show()