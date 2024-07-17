import numpy as np


class TradingAgent:
    def reward_function(self, history):
        # The reward function is modified to penalize large losses more heavily
        # This is done by squaring the negative returns before taking the log
        # This will make the agent more risk averse
        return_val = history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
        if return_val < 1:
            return_val = return_val ** 2
        return np.log(return_val)

    def make_features(self, df):
        # Create additional features based on the provided data
        
        df["feature_close"] = df["close"].pct_change()
        # Create the feature : close[t] / open[t]
        df["feature_open"] = df["close"] / df["open"]
        # Create the feature : high[t] / close[t]
        df["feature_high"] = df["high"] / df["close"]
        # Create the feature : low[t] / close[t]
        df["feature_low"] = df["low"] / df["close"]
        # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
        df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
        df.dropna(inplace=True)
        
        # Moving averages (e.g., 10-period and 50-period moving averages)
        df["feature_ma10"] = df["close"].rolling(window=10).mean()
        df["feature_ma50"] = df["close"].rolling(window=50).mean()

        # Bollinger Bands
        df["feature_bb_upper"] = df["close"].rolling(window=20).mean() + 2 * df["close"].rolling(window=20).std()
        df["feature_bb_lower"] = df["close"].rolling(window=20).mean() - 2 * df["close"].rolling(window=20).std()
        

        # Drop rows with NaN values (due to rolling calculations)
        df.dropna(inplace=True)
        
        # Ensure that there are no look-ahead features
        # (features that use future data to compute the current value)
        
        # Return the dataframe with the newly created features
        return df

    def get_position_list(self):
        return [x / 10.0 for x in range(-10, 21)];

    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    # SARSA algorithm
    def train(self, env, episodes=2):
        for episode in range(episodes):
            done, truncated = False, False
            observation, info = env.reset()
            state = self.extract_state(observation)
            action = self.epsilon_greedy_policy(state)
            while not done and not truncated:
                next_observation, reward, done, truncated, info = env.step(action)
                next_state = self.extract_state(next_observation)
                next_action = self.epsilon_greedy_policy(next_state)
                self.update_q_table(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
                
    def update_q_table(self, state, action, reward, next_state, next_action):
        current_q_value = self.q_table.get((state, action), 0)
        next_q_value = self.q_table.get((next_state, next_action), 0)
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * next_q_value - current_q_value)
        self.q_table[(state, action)] = new_q_value

    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.get_position_list()))  # Explore
        else:
            q_values = [self.q_table.get((state, a), 0) for a in range(len(self.get_position_list()))]
            return np.argmax(q_values)  # Exploit

    def extract_state(self, observation):
        return tuple(observation)


    def get_test_position(self, observation):
        if observation[7] > observation[8]:
            return 20  # Buy
        elif observation[7] < observation[8]:
            return -20  # Sell
        else:
            return 0  # Hold

    def test(self, env):
        # DO NOT CHANGE - all changes will be ignored after upload to BRUTE!
        done, truncated = False, False
        observation, info = env.reset()
        while not done and not truncated:
            new_position = self.get_test_position(observation)
            observation, reward, done, truncated, info = env.step(new_position)
