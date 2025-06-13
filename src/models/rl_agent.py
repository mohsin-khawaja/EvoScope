import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Optional
import logging

class TradingEnvironment:
    """
    Trading environment for RL agent
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1% transaction cost
        max_position: float = 1.0  # Maximum position size (100% of portfolio)
    ):
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.balance = self.initial_balance
        self.position = 0.0  # Current position (-1 to 1, negative = short)
        self.portfolio_value = self.initial_balance
        self.trade_history = []
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self) -> dict:
        """Get current state"""
        return {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'step_count': self.step_count
        }
    
    def step(self, action: int, current_price: float, lstm_features: np.ndarray) -> Tuple[dict, float, bool]:
        """
        Execute trading action
        
        Args:
            action: 0=hold, 1=buy, 2=sell
            current_price: Current asset price
            lstm_features: Features from LSTM model
            
        Returns:
            state: New state
            reward: Reward for the action
            done: Whether episode is finished
        """
        old_portfolio_value = self.portfolio_value
        
        # Execute action
        if action == 1:  # Buy
            self._execute_buy(current_price)
        elif action == 2:  # Sell
            self._execute_sell(current_price)
        # action == 0 is hold (do nothing)
        
        # Update portfolio value
        self.portfolio_value = self.balance + (self.position * current_price)
        
        # Calculate reward
        reward = self._calculate_reward(old_portfolio_value, current_price)
        
        # Check if done
        done = self.portfolio_value <= 0.1 * self.initial_balance or self.step_count >= 1000
        
        self.step_count += 1
        
        return self._get_state(), reward, done
    
    def _execute_buy(self, price: float):
        """Execute buy order"""
        if self.position < self.max_position:
            # Calculate how much we can buy
            available_cash = self.balance * 0.95  # Keep 5% as buffer
            shares_to_buy = min(
                available_cash / price,
                (self.max_position - self.position) * self.initial_balance / price
            )
            
            if shares_to_buy > 0:
                cost = shares_to_buy * price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.position += shares_to_buy * price / self.initial_balance
                    self.trade_history.append(('BUY', shares_to_buy, price, self.step_count))
    
    def _execute_sell(self, price: float):
        """Execute sell order"""
        if self.position > -self.max_position:
            # Calculate how much we can sell
            shares_to_sell = min(
                abs(self.position) * self.initial_balance / price,
                (self.max_position + self.position) * self.initial_balance / price
            )
            
            if shares_to_sell > 0:
                proceeds = shares_to_sell * price * (1 - self.transaction_cost)
                self.balance += proceeds
                self.position -= shares_to_sell * price / self.initial_balance
                self.trade_history.append(('SELL', shares_to_sell, price, self.step_count))
    
    def _calculate_reward(self, old_portfolio_value: float, current_price: float) -> float:
        """Calculate reward for the action"""
        # Portfolio return
        portfolio_return = (self.portfolio_value - old_portfolio_value) / old_portfolio_value
        
        # Risk-adjusted return (Sharpe-like)
        base_reward = portfolio_return * 100  # Scale up
        
        # Penalty for excessive trading
        trading_penalty = -0.01 if len(self.trade_history) > 0 and \
                         self.trade_history[-1][3] == self.step_count else 0
        
        # Bonus for maintaining reasonable position
        position_bonus = 0.001 if abs(self.position) <= 0.8 else -0.005
        
        return base_reward + trading_penalty + position_bonus


class DQNAgent(nn.Module):
    """
    Deep Q-Network agent for trading
    """
    
    def __init__(
        self,
        state_size: int = 131,  # 128 LSTM features + 3 portfolio features
        action_size: int = 3,   # hold, buy, sell
        hidden_size: int = 256,
        learning_rate: float = 0.001
    ):
        super(DQNAgent, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Neural network
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Training parameters
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.forward(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self) -> Optional[float]:
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.forward(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
    
    def save_model(self, filepath: str):
        """Save model weights"""
        torch.save(self.state_dict(), filepath)
    
    def load_model(self, filepath: str):
        """Load model weights"""
        self.load_state_dict(torch.load(filepath))


class TradingAgent:
    """
    Complete trading agent combining LSTM and RL
    """
    
    def __init__(self, lstm_model, dqn_agent, environment):
        self.lstm_model = lstm_model
        self.dqn_agent = dqn_agent
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
    def prepare_state(self, lstm_features: np.ndarray, env_state: dict) -> np.ndarray:
        """Combine LSTM features with environment state"""
        portfolio_features = np.array([
            env_state['balance'] / self.environment.initial_balance,
            env_state['position'],
            env_state['portfolio_value'] / self.environment.initial_balance
        ])
        return np.concatenate([lstm_features.flatten(), portfolio_features])
    
    def train_episode(self, price_data: np.ndarray, lstm_features: np.ndarray) -> dict:
        """Train agent for one episode"""
        state = self.environment.reset()
        total_reward = 0
        losses = []
        
        for i in range(len(price_data) - 1):
            # Prepare state
            current_state = self.prepare_state(lstm_features[i], state)
            
            # Choose action
            action = self.dqn_agent.act(current_state, training=True)
            
            # Execute action
            next_state, reward, done = self.environment.step(
                action, price_data[i], lstm_features[i]
            )
            
            # Prepare next state
            next_state_vector = self.prepare_state(lstm_features[i+1], next_state)
            
            # Remember experience
            self.dqn_agent.remember(current_state, action, reward, next_state_vector, done)
            
            # Train
            loss = self.dqn_agent.replay()
            if loss is not None:
                losses.append(loss)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        return {
            'total_reward': total_reward,
            'final_portfolio_value': state['portfolio_value'],
            'num_trades': len(self.environment.trade_history),
            'avg_loss': np.mean(losses) if losses else 0,
            'epsilon': self.dqn_agent.epsilon
        }
    
    def predict_action(self, lstm_features: np.ndarray, env_state: dict) -> Tuple[int, str]:
        """Predict best action for current state"""
        state = self.prepare_state(lstm_features, env_state)
        action = self.dqn_agent.act(state, training=False)
        
        action_names = ['HOLD', 'BUY', 'SELL']
        return action, action_names[action] 