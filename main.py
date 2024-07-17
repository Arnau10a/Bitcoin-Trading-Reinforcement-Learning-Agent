import gymnasium as gym
import gym_trading_env
from tradingagent import TradingAgent
import pandas as pd

# use this class for debugging and training the method, however, the changes to the code will NOT be usable in BRUTE,
# only the tradingagent.py is important

pd.options.mode.chained_assignment = None


agent = TradingAgent()

train = pd.read_pickle("./data/train.pkl")
train = agent.make_features(train)

env = gym.make("TradingEnv",
               name="TrainingEnvironment",
               df=train,
               reward_function=agent.reward_function,
               trading_fees=0.01 / 100,
               borrow_interest_rate=0.0003 / 100,
               portfolio_initial_value=1000,
               initial_position=0.0,
               max_episode_duration="max",
               positions=agent.get_position_list(),
               )

print("going to train")
agent.train(env)

train = pd.read_pickle("./data/test.pkl")
train = agent.make_features(train)

env = gym.make("TradingEnv",
               name="TrainingEnvironment",
               df=train,
               reward_function=agent.reward_function,
               trading_fees=0.01 / 100,
               borrow_interest_rate=0.0003 / 100,
               portfolio_initial_value=1000,
               initial_position=0.0,
               max_episode_duration="max",
               positions=agent.get_position_list(),
               )

print("going to test")
agent.test(env)
# this will return the market return and portfolio return values
print(env.unwrapped.get_metrics())

# If you want to visualize the results for better debugging, this guide might be usefult
# https://gym-trading-env.readthedocs.io/en/latest/render.html
