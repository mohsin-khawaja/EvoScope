import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """Custom Trading Environment compatible with Gymnasium"""
    
    def __init__(self, df, transaction_cost=0.001):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.tc = transaction_cost
        
        # Define action and observation space
        # Action: continuous position from -1 (short) to 1 (long)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation: [AAPL_Close, BTC_Close, sentiment, position]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1]), 
            high=np.array([1000, 100000, 1, 1]), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.position = 0.0
        self.cash = 1.0
        self.initial_portfolio_value = self.cash
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        if self.t >= len(self.df):
            self.t = len(self.df) - 1
            
        row = self.df.iloc[self.t]
        return np.array([
            row['AAPL_Close'],
            row['BTC_Close'], 
            row['sentiment'],
            self.position
        ], dtype=np.float32)

    def step(self, action):
        # Ensure action is a scalar
        if isinstance(action, np.ndarray):
            action = action[0]
        
        price = self.df['AAPL_Close'].iloc[self.t]
        prev_val = self.position * price
        
        # Calculate transaction cost
        cost = abs(action - self.position) * self.tc
        
        # Update position
        self.position = np.clip(action, -1, 1)
        
        # Move to next time step
        self.t += 1
        
        # Check if episode is done
        done = self.t >= len(self.df) - 1
        truncated = False
        
        if not done:
            new_price = self.df['AAPL_Close'].iloc[self.t]
            cur_val = self.position * new_price
            reward = cur_val - prev_val - cost
        else:
            reward = 0
            
        obs = self._get_obs()
        info = {'portfolio_value': self.position * price + self.cash}
        
        return obs, reward, done, truncated, info 