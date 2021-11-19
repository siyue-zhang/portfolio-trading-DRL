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
    if trading_shares == 0:
        fee = 0
    else:
        fee = max(np.round(0.03*np.abs(trading_shares),2), 2)
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

        self.max_share_per_trade = 1000
        self.reward_scale = 1e-4
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
        
        self.stocks_max = defaultdict()
        self.stocks_history_dic = dict.fromkeys(self.portfolio_stocks)

        for idx, stock in enumerate(self.portfolio_stocks):
            self.stocks_history_dic[stock] = self.stocks_history[self.stocks_history["symbol"]==stock]
            self.stocks_history_dic[stock] = self.stocks_history_dic[stock].join(RSI(stock,self.stocks_history)["RSI"], how="right")
            self.stocks_history_dic[stock] = self.stocks_history_dic[stock].join(MACD(stock,self.stocks_history)["MACD"], how="right")
            self.stocks_history_dic[stock].drop(["symbol","open","low","high"], axis=1, inplace=True)
            
            if idx == 0:
                self.day_list =  self.stocks_history_dic[stock].index
           
            self.stocks_max[stock] = np.round(np.max(self.stocks_history_dic[stock]["close"]),2)
            # print("MACD MAX", max(self.stocks_history_dic[stock]["MACD"]), min(self.stocks_history_dic[stock]["MACD"]))
            # print("RSI MAX", max(self.stocks_history_dic[stock]["RSI"]), min(self.stocks_history_dic[stock]["RSI"]))

        # print(self.stocks_history_dic["WMT"].head(30))

        # self.action_space = gym.spaces.Discrete(size * size)
        self.action_space = spaces.Box(low=-1, high=1,shape = (self.n_stocks,))
        # Shape 5*N+1 : [Current Balance] + [prices 1-N] + [owned shares 1-N] + [Volume 1-N]
        # + [MACD 1-N] + [RSI 1-N]
        self.observation_space = spaces.Box(low=-1, high=1, shape = (5*self.n_stocks+1,))
        self.state_max = [self.initial_cash*10] + \
            [ 200 for i in range(self.n_stocks)] + \
            [ 2000 for i in range(self.n_stocks)] + \
            [500000000 for i in range(self.n_stocks)] + \
            [5 for i in range(self.n_stocks)] + \
            [100 for i in range(self.n_stocks)]
        self.account = dict.fromkeys(['Cash'] + self.portfolio_stocks)
        self.total_asset = None
        self.reward = None

        self.current_day = None
        self.current_prices = dict.fromkeys(self.portfolio_stocks)
        self.state = None
        self.norm_state = None
        self.terminal = False

    def reset(self):
        
        self.account['Cash'] = self.initial_cash
        for stock in self.portfolio_stocks:
            self.account[stock] = 0

        self.current_day = self.start_day
        self._update_current_price()
        
        self.total_asset = self.initial_cash
        self.reward = 0
        self.terminal = False

        self._update_state()
        print("K", self.state, len(self.state))
        self._update_norm_state()
        print("KK", self.norm_state, len(self.norm_state))

        return
    
    def _update_current_price(self):
        for stock in self.portfolio_stocks:
            self.current_prices[stock] = self.stocks_history_dic[stock].loc[self.current_day, "close"]
        return


    def _sell_stock(self, index, shares):
        stock = self.portfolio_stocks[index]
        available_shares = self.account[stock]
        shares = int(-shares)
        shares = min(available_shares, shares)

        self.account[stock] -= shares
        fee = get_trading_fee(self.current_prices[stock], shares)
        print("fee", fee)
        print(stock, "price", self.current_prices[stock])
        self.account["Cash"] += (shares*self.current_prices[stock] - fee)
        print(shares)
        print("after sell ", self.account)
        return
    
    def _buy_stock(self, index, shares):
        stock = self.portfolio_stocks[index]
        available_shares = (self.account["Cash"]-2) // (self.current_prices[stock] + 0.03)
        shares = int(shares)
        shares = min(available_shares, shares)

        self.account[stock] += shares
        fee = get_trading_fee(self.current_prices[stock], shares)
        print("fee", fee)
        print(stock, "price", self.current_prices[stock])
        self.account["Cash"] -= (shares*self.current_prices[stock] + fee)
        print(shares)
        print("after buy ", self.account)

        return

    def _update_state(self):

        self.state = [self.account["Cash"]] + \
                    [ self.current_prices[stock] for stock in self.portfolio_stocks]  + \
                    [ self.account[stock] for stock in self.portfolio_stocks] + \
                    [ self.stocks_history_dic[stock].loc[self.current_day,"volume"] for stock in self.portfolio_stocks] + \
                    [ self.stocks_history_dic[stock].loc[self.current_day,"MACD"] for stock in self.portfolio_stocks]  + \
                    [ self.stocks_history_dic[stock].loc[self.current_day,"RSI"] for stock in self.portfolio_stocks]  

        return

    def _update_norm_state(self):
        self.norm_state = self.state
        for i in range(len(self.state)):
            if i>3*self.n_stocks and i<=4*self.n_stocks:
                self.norm_state[i] = self.norm_state[i]/self.state_max[i]
            else:
                self.norm_state[i] = (self.norm_state[i]/self.state_max[i] - 0.5) / 0.5

        return

    def step(self, actions):
        

        actions = actions*self.max_share_per_trade
        argsort_actions = np.argsort(actions)
        
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

        for index in sell_index:
            # print('take sell action'.format(actions[index]))
            print("before sell",index, self.account)
            self._sell_stock(index, actions[index])

        for index in buy_index:
            # print('take buy action: {}'.format(actions[index]))
            self._buy_stock(index, actions[index])

        assert(self.account["Cash"]>=0)
        # update total asset value
        summation = self.account['Cash']
        for stock in self.portfolio_stocks:
            summation += self.account[stock]*self.stocks_history_dic[stock].loc[self.current_day,"close"]
        self.reward = (summation - self.total_asset)*self.reward_scale
        self.total_asset = summation

        # update to next date
        self.current_day = self.day_list[np.where(self.day_list == self.current_day)[0] + 1][0]
        self._update_current_price()

        if self.current_day == self.end_day:
            self.terminal = True

        self._update_state()
        print(self.state, len(self.state))
        self._update_norm_state()
        print(self.norm_state, len(self.norm_state))

        return self.norm_state, self.reward, self.terminal, {}



    def render(self):
        message = " | ".join([ f"{stock} - "+ str(round(self.account[stock],2)) for stock in self.portfolio_stocks])
        print(f"\n{str(self.current_day)[:10]}" + f" Account Status: " + f"Cash - {np.round(self.account['Cash'],2)} | " + message)
        print(f"Total Assets Value: {np.round(self.total_asset,2)}")
        return



my_pocket = PortfolioTradingEnv(["WMT","AAPL","MMM"], 100000, "2010-03-12", "2016-12-30")
my_pocket.reset()
actions = my_pocket.action_space.sample()
my_pocket.account["WMT"] = my_pocket.account["WMT"]+1000
my_pocket.account["MMM"] = my_pocket.account["MMM"]+1000
print(actions)
my_pocket.step(actions)
# my_pocket.step(actions)
my_pocket.render()
