import os
import datetime

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from portfolio_trading_env import PortfolioTradingEnv



class ShowTotalAsset(BaseCallback):
    def __init__(self, verbose=1):
        super(ShowTotalAsset, self).__init__(verbose)
        self.episode = 0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # if env.current_day == env.start_day:
        #     self.episode += 1
        #     print("one episode starts.")

        if env.current_day == last_day:
            print("one episode ends.")
            env.render()

        return True        

last_day = datetime.datetime.strptime("2016-04-11", '%Y-%m-%d') 

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

# Instantiate the env
env = PortfolioTradingEnv(["WMT","ABBV","MMM"], 100000, "2013-04-10", "2016-04-12")
# Logs will be saved in log_dir/monitor.csv
log_dir="./"
env = Monitor(env, log_dir,info_keywords=("total_reward", "total_asset", "last_account", "action_summary"))

total_timesteps = 100000
# env = make_vec_env(lambda: env, n_envs=1)
callback = ShowTotalAsset()
model = A2C('MlpPolicy', env).learn(total_timesteps)



# from stable_baselines3.common.env_checker import check_env
# It will check your custom environment and output additional warnings if needed
# check_env(env, warn=True)

# print(env.reset(), len(env.reset()))
# print(env.observation_space)


# print("--------------------TEST---------------")

# env_test = PortfolioTradingEnv(["WMT","ABBV","MMM"], 100000, "2015-12-01", "2016-12-30")
# obs = env_test.reset()
# n_steps = 20
# for step in range(n_steps):
#     action, _ = model.predict(obs, deterministic=True)
#     print("Step {}".format(step + 1))
#     print("Prices", env_test.current_prices)
#     print("Action: ", action)
#     obs, reward, done, info = env_test.step(action)
#     env_test.render()
#     if done:
#         # Note that the VecEnv resets automatically
#         # when a done signal is encountered
#         print("Total asset: ", env_test.total_asset)
#         break