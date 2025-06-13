import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, List
import os
from datetime import datetime
import json

from models.lstm_model import LSTMPricePredictor, create_lstm_model
from models.rl_agent import TradingAgent, DQNAgent, TradingEnvironment
from data.fetch_data import get_stock_data
from features.build_features import build_dataset


class LSTMTrainer:
    """
    LSTM model trainer for price prediction
    """
    
    def __init__(self, model: LSTMPricePredictor, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(self, data: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.values)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict Close price (first column)
        
        return np.array(X), np.array(y)
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
              epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001) -> Dict:
        """
        Train the LSTM model
        """
        # Prepare training data
        X_train, y_train = self.prepare_data(train_data, self.model.sequence_length)
        X_val, y_val = self.prepare_data(val_data, self.model.sequence_length)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        self.logger.info(f"Starting LSTM training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                predictions, _ = self.model(batch_X)
                loss = criterion(predictions.squeeze(), batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch_X = X_val[i:i+batch_size]
                    batch_y = y_val[i:i+batch_size]
                    
                    predictions, _ = self.model(batch_X)
                    loss = criterion(predictions.squeeze(), batch_y)
                    val_loss += loss.item()
            
            train_loss /= (len(X_train) // batch_size)
            val_loss /= (len(X_val) // batch_size)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'models/best_lstm_model.pth')
            
            if epoch % 10 == 0:
                self.logger.info(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate the trained model
        """
        X_test, y_test = self.prepare_data(test_data, self.model.sequence_length)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions, features = self.model(X_test)
            mse = nn.MSELoss()(predictions.squeeze(), y_test)
            mae = nn.L1Loss()(predictions.squeeze(), y_test)
        
        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'rmse': np.sqrt(mse.item())
        }


class RLTrainer:
    """
    Reinforcement Learning trainer for trading agent
    """
    
    def __init__(self, lstm_model: LSTMPricePredictor, initial_balance: float = 10000.0):
        self.lstm_model = lstm_model
        self.environment = TradingEnvironment(initial_balance=initial_balance)
        self.dqn_agent = DQNAgent()
        self.trading_agent = TradingAgent(lstm_model, self.dqn_agent, self.environment)
        self.logger = logging.getLogger(__name__)
        
    def train(self, price_data: pd.DataFrame, episodes: int = 1000) -> Dict:
        """
        Train the RL agent
        """
        # Prepare features using LSTM
        features_data = build_dataset(price_data)
        
        # Extract LSTM features for all time steps
        lstm_features_list = []
        sequence_length = self.lstm_model.sequence_length
        
        for i in range(sequence_length, len(features_data)):
            sequence = features_data.iloc[i-sequence_length:i].values
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            with torch.no_grad():
                features = self.lstm_model.extract_features(sequence_tensor)
                lstm_features_list.append(features.numpy().flatten())
        
        lstm_features = np.array(lstm_features_list)
        prices = price_data['Close'].iloc[sequence_length:].values
        
        # Training metrics
        episode_rewards = []
        episode_losses = []
        portfolio_values = []
        
        self.logger.info(f"Starting RL training for {episodes} episodes...")
        
        for episode in range(episodes):
            # Train one episode
            results = self.trading_agent.train_episode(prices, lstm_features)
            
            episode_rewards.append(results['total_reward'])
            episode_losses.append(results['avg_loss'])
            portfolio_values.append(results['final_portfolio_value'])
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_portfolio = np.mean(portfolio_values[-100:])
                self.logger.info(
                    f'Episode {episode}/{episodes}, '
                    f'Avg Reward: {avg_reward:.2f}, '
                    f'Avg Portfolio: ${avg_portfolio:.2f}, '
                    f'Epsilon: {results["epsilon"]:.3f}'
                )
            
            # Save best model
            if episode > 100 and results['final_portfolio_value'] == max(portfolio_values):
                self.dqn_agent.save_model('models/best_dqn_model.pth')
        
        return {
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'portfolio_values': portfolio_values,
            'final_epsilon': self.dqn_agent.epsilon
        }
    
    def evaluate(self, test_data: pd.DataFrame, num_episodes: int = 10) -> Dict:
        """
        Evaluate the trained RL agent
        """
        # Load best model
        try:
            self.dqn_agent.load_model('models/best_dqn_model.pth')
        except:
            self.logger.warning("Could not load best model, using current model")
        
        # Prepare test data
        features_data = build_dataset(test_data)
        lstm_features_list = []
        sequence_length = self.lstm_model.sequence_length
        
        for i in range(sequence_length, len(features_data)):
            sequence = features_data.iloc[i-sequence_length:i].values
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            with torch.no_grad():
                features = self.lstm_model.extract_features(sequence_tensor)
                lstm_features_list.append(features.numpy().flatten())
        
        lstm_features = np.array(lstm_features_list)
        prices = test_data['Close'].iloc[sequence_length:].values
        
        # Run evaluation episodes
        portfolio_values = []
        total_returns = []
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            total_reward = 0
            
            for i in range(len(prices) - 1):
                current_state = self.trading_agent.prepare_state(lstm_features[i], state)
                action = self.dqn_agent.act(current_state, training=False)
                
                next_state, reward, done = self.environment.step(
                    action, prices[i], lstm_features[i]
                )
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            portfolio_values.append(state['portfolio_value'])
            total_returns.append(
                (state['portfolio_value'] - self.environment.initial_balance) / 
                self.environment.initial_balance * 100
            )
        
        return {
            'avg_portfolio_value': np.mean(portfolio_values),
            'avg_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'max_return': np.max(total_returns),
            'min_return': np.min(total_returns),
            'sharpe_ratio': np.mean(total_returns) / np.std(total_returns) if np.std(total_returns) > 0 else 0
        }


def plot_training_results(lstm_results: Dict, rl_results: Dict, save_path: str = 'training_results.png'):
    """
    Plot training results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # LSTM training losses
    axes[0, 0].plot(lstm_results['train_losses'], label='Train Loss')
    axes[0, 0].plot(lstm_results['val_losses'], label='Validation Loss')
    axes[0, 0].set_title('LSTM Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # RL episode rewards
    axes[0, 1].plot(rl_results['episode_rewards'])
    axes[0, 1].set_title('RL Episode Rewards')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Reward')
    axes[0, 1].grid(True)
    
    # RL portfolio values
    axes[1, 0].plot(rl_results['portfolio_values'])
    axes[1, 0].set_title('RL Portfolio Values')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Portfolio Value ($)')
    axes[1, 0].grid(True)
    
    # RL training losses
    valid_losses = [loss for loss in rl_results['episode_losses'] if loss > 0]
    if valid_losses:
        axes[1, 1].plot(valid_losses)
        axes[1, 1].set_title('RL Training Losses')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def main():
    """
    Main training function
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Configuration
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'lstm_config': {
            'input_size': 10,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'sequence_length': 60
        },
        'training_config': {
            'lstm_epochs': 100,
            'rl_episodes': 1000,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    }
    
    logger.info("Starting model training pipeline...")
    
    # Load and prepare data
    all_data = []
    for symbol in config['symbols']:
        data = get_stock_data(symbol, period='2y')
        features = build_dataset(data)
        all_data.append(features)
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Split data
    train_size = int(0.7 * len(combined_data))
    val_size = int(0.15 * len(combined_data))
    
    train_data = combined_data[:train_size]
    val_data = combined_data[train_size:train_size + val_size]
    test_data = combined_data[train_size + val_size:]
    
    logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Train LSTM model
    logger.info("Training LSTM model...")
    lstm_model = create_lstm_model(config['lstm_config'])
    lstm_trainer = LSTMTrainer(lstm_model)
    
    lstm_results = lstm_trainer.train(
        train_data, val_data,
        epochs=config['training_config']['lstm_epochs'],
        batch_size=config['training_config']['batch_size'],
        learning_rate=config['training_config']['learning_rate']
    )
    
    # Evaluate LSTM
    lstm_eval = lstm_trainer.evaluate(test_data)
    logger.info(f"LSTM Evaluation - MSE: {lstm_eval['mse']:.6f}, MAE: {lstm_eval['mae']:.6f}")
    
    # Train RL agent
    logger.info("Training RL agent...")
    rl_trainer = RLTrainer(lstm_model)
    
    # Use a subset of data for RL training (it's computationally intensive)
    rl_train_data = get_stock_data('AAPL', period='1y')  # Use AAPL for RL training
    
    rl_results = rl_trainer.train(
        rl_train_data,
        episodes=config['training_config']['rl_episodes']
    )
    
    # Evaluate RL agent
    rl_test_data = get_stock_data('AAPL', period='3mo')  # Recent 3 months for testing
    rl_eval = rl_trainer.evaluate(rl_test_data)
    logger.info(f"RL Evaluation - Avg Return: {rl_eval['avg_return']:.2f}%, Sharpe: {rl_eval['sharpe_ratio']:.3f}")
    
    # Save results
    results = {
        'config': config,
        'lstm_results': lstm_results,
        'lstm_evaluation': lstm_eval,
        'rl_results': rl_results,
        'rl_evaluation': rl_eval,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Plot results
    plot_training_results(lstm_results, rl_results)
    
    logger.info("Training completed successfully!")
    logger.info(f"Models saved in 'models/' directory")
    logger.info(f"Results saved in 'training_results.json'")


if __name__ == "__main__":
    main() 