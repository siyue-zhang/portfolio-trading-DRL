import os

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import plot_results

from env.portfolio_trading_env import PortfolioTradingEnv
from callback import SaveOnBestTrainingRewardCallback


def run_DRL(portfolio_stocks, initial_cash, start_day, end_day, DRL_model, timesteps = 5e4):

    # Create log dir
    log_dir = f"log/{DRL_model}"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = PortfolioTradingEnv(portfolio_stocks, initial_cash, start_day, end_day)
    n_days = env.n_days

    # Create the callback: check every episode
    callback = SaveOnBestTrainingRewardCallback(check_freq=n_days, log_dir=log_dir, model_name=DRL_model)

    env = Monitor(env, log_dir,info_keywords=("total_asset", "last_account", "action_summary"))

    if DRL_model == "TD3":
        # Add some action noise for exploration
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        # Because we use parameter noise, we should use a MlpPolicy with layer normalization
        model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0, seed=10)

    if DRL_model == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, seed=10)

    # Train the agent
    model.learn(total_timesteps=int(timesteps), callback=callback)

    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, f"{DRL_model} Trading")
    plt.show()


if __name__=='__main__':

    run_DRL(["WMT","AAPL","MMM"], 100000, "2012-01-01", "2012-12-30", "PPO", 1e5)