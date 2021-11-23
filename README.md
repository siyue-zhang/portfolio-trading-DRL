# portfolio-trading-DRL
Automatic stock portfolio management based on reinforcement learning

The project works on S&P 500 historical split-adjusted price and volume data. RSI and MACD technical indicators are also calculated as additional observations for DRL agent. The agent receives these inputs and take actions on buy/hold/sell on each stock in the portfolio at the close of the market every day with close prices. Trading service fee is subtracted from the account for each deal. The cash in the account is not allowed to be negative. The reward given to the agent is defined by the total asset increment every day. The final objective is to maximize the total asset value after the target duration.

Dependencies:
* Stable Baselines 3
* Pytorch
* Gym
* Pandas, Matplotlib