# Bitcoin Trading Reinforcement Learning Agent

## Project Overview
This project aims to develop a reinforcement learning agent to trade Bitcoin using historical price data. The goal is to create an agent that can learn to make profitable trading decisions. The Gymnasium library and the `gym-trading-env` environment are utilized to simulate the trading environment.

## Warning - Comparison to Real-world
While many research papers discuss applying reinforcement learning techniques to cryptocurrency trading, it's important to note that the simplified algorithm developed here is not suitable for real-world trading. Real-world markets are complex and non-stationary, requiring continuous adjustments to the algorithm. Additionally, the environment in the real world is formed by many agents trying to maximize their gain, making it unpredictable. Therefore, this code is for educational purposes only and should not be used for actual trading.

## Implementation Details
- **Environment**: Uses the Gymnasium library and `gym-trading-env`.
- **Files**:
  - `main.py`: Constructs the environment and handles training and testing.
  - `tradingagent.py`: Contains the implementation of the trading agent. Modify this file to implement the reinforcement learning algorithm and feature engineering.

### Key Methods in `tradingagent.py`
- `reward_function`: Modify this to define a better reward function. The current implementation calculates the logarithm of the percentage gain.
- `make_features`: Create numeric features for learning, such as Bollinger bands or moving averages. Ensure no look-ahead features are used.
- `get_position_list`: Customize the set of actions available to the agent, representing different positions in USD and Bitcoin.
- `get_test_position`: Define the policy for testing.
- `train`: Implement the training process. You can use any reinforcement learning algorithm to achieve the desired performance.

## Tasks
1. **State Representation**: Propose and estimate the number of states for different representations and select the best one.
2. **Reward Function**: Implement a suitable reward function that reflects trading performance accurately.
3. **Feature Engineering**: Develop features that help the learning process, avoiding any look-ahead bias.
4. **Reinforcement Learning Algorithm**: Implement an algorithm of your choice (e.g., Q-learning, SARSA).
5. **Evaluation**: Test and compare the performance on training and validation datasets.

## Goals
- Achieve a performance of 10% gain on test data.
- Achieve a performance of 50% gain on validation data.

## References
- [Gymnasium library](https://gymnasium.farama.org/index.html)
- [gym-trading-env documentation](https://gym-trading-env.readthedocs.io/en/latest/)
- [Bitcoin trading strategies paper](https://doi.org/10.1038/s41598-024-51408-w)
- [Black swan theory](https://en.wikipedia.org/wiki/Black_swan_theory)
- [Feature engineering documentation](https://gym-trading-env.readthedocs.io/en/latest/features.html)
- [Environment description](https://gym-trading-env.readthedocs.io/en/latest/environment_desc.html#action-space)

## Disclaimer
This project is for educational purposes only. The algorithm is simplified and not suitable for real-world trading. Use it at your own risk.
