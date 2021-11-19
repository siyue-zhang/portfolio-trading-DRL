import pandas as pd
import gym
import numpy as np
from datetime import datetime

from indicators import *
from gym import spaces
from collections import defaultdict

from tianshou.env import MultiAgentEnv



# trading fee:  min 2 USD/deal  0.01 USD/stock
def get_trading_fee(price, trading_shares):
    fee = max(0.01*np.abs(trading_shares), 2)
    return fee


class PortfolioTradingEnv(MultiAgentEnv):
    """This is a simple implementation of the Tic-Tac-Toe game, where two
    agents play against each other.
    The implementation is intended to show how to wrap an environment to
    satisfy the interface of :class:`~tianshou.env.MultiAgentEnv`.
    :param size: the size of the board (square board)
    :param win_size: how many units in a row is considered to win
    """

    def __init__(self, portfolio_stocks, initial_cash, start_day, end_day):
        super().__init__()
        assert len(portfolio_stocks) > 0, f'Portfolio should include at least one stock'
        self.portfolio_stocks = portfolio_stocks
        self.n_stocks = len(self.portfolio_stocks)
        assert initial_cash > 0, f'Initial investment should be positive, but got {initial_cash}'
        self.initial_cash = initial_cash
        self.start_day = datetime.strptime(start_day, '%Y-%m-%d') 
        self.end_day = datetime.strptime(end_day, '%Y-%m-%d')

        self.stocks_history = pd.read_csv("./prices-split-adjusted.csv")
        self.stocks_history = self.stocks_history[self.stocks_history["symbol"].isin(self.portfolio_stocks)]
        self.stocks_history.set_index(keys="date", drop=True, inplace=True)
        self.stocks_history.index = pd.to_datetime(self.stocks_history.index)
        
        self.stocks_mean = defaultdict()
        self.stocks_history_dic = dict.fromkeys(self.portfolio_stocks)
        for stock in self.portfolio_stocks:
            self.stocks_history_dic[stock] = self.stocks_history[self.stocks_history["symbol"]==stock]
            self.stocks_history_dic[stock] = self.stocks_history_dic[stock].join(RSI(stock,self.stocks_history)["RSI"], how="right")
            self.stocks_history_dic[stock] = self.stocks_history_dic[stock].join(MACD(stock,self.stocks_history)["MACD"], how="right")
            self.stocks_history_dic[stock].drop(["symbol","open","low","high"], axis=1, inplace=True)
            self.stocks_mean[stock] = np.round(np.mean(self.stocks_history_dic[stock]["close"]),2)
        print(self.stocks_history_dic["WMT"].head(30))

        # self.action_space = gym.spaces.Discrete(size * size)
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.n_stocks,))
        # Shape 5*N+1 : [Current Balance] + [prices 1-N] + [owned shares 1-N] + [Volume 1-N]
        # + [MACD 1-N] + [RSI 1-N]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (5*self.n_stocks+1,))

        self.account = dict.fromkeys(['Cash'] + self.portfolio_stocks)
        self.total_asset = None
        self.reward = None

        self.current_day = None
        self.terminal = False

    def reset(self):
        
        self.account['Cash'] = self.initial_cash
        for stock in self.portfolio_stocks:
            self.account[stock] = 0
        self.current_day = self.start_day
        self.total_asset = self.initial_cash
        self.reward = 0
        self.terminal = False

        return

    def step(self, action):
        
        # update total asset value
        summation = self.account['Cash']
        for stock in self.portfolio_stocks:
            summation += self.account[stock]*self.stocks_history_dic[stock].loc[self.current_day,"close"]
        self.total_asset = summation

        return

    def render(self):
        message = " | ".join([ f"{stock} - "+ str(round(self.account[stock],2)) for stock in self.portfolio_stocks])
        print(f"\n{str(self.current_day)[:10]}" + f" Account Status: " + f"Cash - {np.round(self.account['Cash'],2)} | " + message)
        print(f"Total Assets Value: {np.round(self.total_asset,2)}")
        return

my_pocket = PortfolioTradingEnv(["WMT","AAPL","MMM"], 100000, "2010-03-04", "2016-12-30")
my_pocket.reset()
my_pocket.step(1)
my_pocket.render()