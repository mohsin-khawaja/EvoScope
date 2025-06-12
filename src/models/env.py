import numpy as np

class TradingEnv:
    def __init__(self, df, transaction_cost=0.001):
        self.df = df.reset_index(drop=True)
        self.tc = transaction_cost
        self.reset()

    def reset(self):
        self.t = 0
        self.position = 0
        self.cash = 1.0
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.t]
        return np.array([
            row['AAPL_Close'],
            row['BTC_Close'],
            row['sentiment'],
            self.position
        ], dtype=float)

    def step(self, action):
        price = self.df['AAPL_Close'].iloc[self.t]
        prev_val = self.position * price
        cost = abs(action - self.position) * self.tc
        self.position = action
        self.t += 1
        new_price = self.df['AAPL_Close'].iloc[self.t]
        cur_val = self.position * new_price
        reward = cur_val - prev_val - cost
        done = self.t >= len(self.df) - 1
        return self._get_obs(), reward, done, {} 