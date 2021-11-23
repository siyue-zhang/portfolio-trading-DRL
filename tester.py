
from collections import defaultdict
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import TD3, PPO

from env.portfolio_trading_env import PortfolioTradingEnv

def draw_train_log(DRL_model):

    log = pd.read_csv(f"./log/{DRL_model}/monitor.csv", skiprows=1)

    x = range(1, log.index[-1]+2)

    plt.figure(figsize=(14,5))
    plt.plot(x, log["total_asset"].values, color="blue")
    plt.title("Total asset value after each episode")
    plt.show()


def test_trained_model(portfolio_stocks, initial_cash, start_day, end_day, DRL_model):

    print("--------------------TEST-----------------")
    env_test = PortfolioTradingEnv(portfolio_stocks, initial_cash, start_day, end_day, verbose=True)

    if DRL_model == "TD3":
        model = TD3.load("./log/TD3/TD3_best_model")

    if DRL_model == "PPO":
        model = PPO.load("./log/PPO/PPO_best_model")

    obs = env_test.reset()
    total_asset_log = []
    equal_weighted_index = [1]
    prices_log = defaultdict(lambda: [])

    n_steps = env_test.n_days
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)

        pre_prices = env_test.current_prices.copy()
        print(f"Day {step+1}: {env_test.current_prices}")
        obs, reward, done, info = env_test.step(action)

        indexes = []
        for stock in portfolio_stocks:
            indexes.append(env_test.current_prices[stock]/pre_prices[stock])
            prices_log[stock].append(env_test.current_prices[stock])
        equal_weighted_index.append(equal_weighted_index[-1]*np.mean(indexes))

        total_asset_log.append(info["total_asset"])
        env_test.render()
        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Total asset: ", env_test.total_asset)
            break
    print(f"After Test: \n {env_test.action_summary}")

    np.savetxt(f"./log/{DRL_model}/{DRL_model}_asset_train_2012_test_{start_day}.csv", total_asset_log)

    del equal_weighted_index[0]

    x = range(1, len(total_asset_log)+1)
    fig, ax1 = plt.subplots()
    ax1.plot(x, total_asset_log, color="blue", label="total asset",)
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(x, equal_weighted_index, color="green", label="price index")
    ax2.legend(loc="lower left")
    plt.show()

    # plt.plot(x, prices_log[portfolio_stocks[0]])
    # plt.plot(x, prices_log[portfolio_stocks[1]])
    # plt.plot(x, prices_log[portfolio_stocks[2]])
    # plt.show()



if __name__=='__main__':

    # draw_train_log("TD3")

    test_trained_model(["WMT","AAPL","MMM"], 100000, "2012-01-01", "2012-12-30", "PPO")