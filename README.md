# rl-lstm-ai-trading-agent

An agentic RL + LSTM trading framework for stocks and crypto.  
- **Data**: OHLCV from Yahoo Finance & Binance, plus news sentiment  
- **Agent**: LSTM-based policy network (PPO/DQN)  
- **Evaluation**: Walk-forward backtests, P&L / Sharpe metrics  

## Structure
```
rl-lstm-ai-trading-agent/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   │   └── fetch_data.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   └── train_model.py
│   └── utils/
│       └── config.py
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
