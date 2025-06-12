# src/models/train_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

from ..features.build_features import build_dataset
from .env import TradingEnv

def train_lstm_policy():
    """Original LSTM policy training function"""
    df = build_dataset()
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Calculate returns and create labels
    df['returns'] = df['AAPL_Close'].pct_change()
    df['future_returns'] = df['returns'].shift(-1)
    
    # Remove NaN values
    df = df.dropna()
    
    # Create features and labels with same length
    X = df[['AAPL_Close','BTC_Close','sentiment']].values
    y = (df['future_returns'] > 0).astype(int).values
    
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # TODO: reshape for LSTM and build simple LSTM + dense policy head
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    print(model.evaluate(X_test, y_test))


def train_rl_agent():
    """RL agent training using PPO"""
    try:
        from stable_baselines3 import PPO
        
        # Load dataset and create environment
        df = build_dataset()
        env = TradingEnv(df)
        
        # Create PPO model
        model = PPO("MlpPolicy", env, verbose=1)
        
        # Train the model
        print("Starting RL training...")
        model.learn(total_timesteps=10000)
        
        # Save the model
        model.save("ppo_trader")
        print("Model saved as 'ppo_trader'")
        
        # Basic evaluation
        obs, info = env.reset()
        total_reward = 0
        for _ in range(100):
            action, _ = model.predict(obs)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
            total_reward += reward
            if done:
                break
        
        print(f"Evaluation total reward: {total_reward}")
        
    except ImportError:
        print("stable-baselines3 not installed. Running pseudocode version...")
        
        # Pseudocode version without stable-baselines3
        df = build_dataset()
        env = TradingEnv(df)
        
        print("=== RL Training Pseudocode ===")
        print("1. Initialize PPO agent with MLP policy")
        print("2. For each episode:")
        print("   - Reset environment")
        print("   - Collect trajectory using current policy")  
        print("   - Calculate returns and advantages")
        print("   - Update policy using PPO loss")
        print("3. Save trained model")
        print("4. Evaluate performance")
        
        # Simple random policy evaluation for demonstration
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:
            # Random action between -1 and 1 (position)
            action = np.random.uniform(-1, 1)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
            total_reward += reward
            steps += 1
            if done:
                break
                
        print(f"Random policy total reward over {steps} steps: {total_reward:.4f}")


if __name__ == "__main__":
    print("Training LSTM policy...")
    train_lstm_policy()
    
    print("\nTraining RL agent...")
    train_rl_agent()
