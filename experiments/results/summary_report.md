# Experimental Results Summary
Generated on: 2025-06-13 00:40:55
Total experiments conducted: 167

## LSTM Experiments
- Total LSTM experiments: 81
- Best configuration:
  - Hidden size: 256.0
  - Layers: 1.0
  - Sequence length: 60.0
  - Dropout: 0.2
  - Test accuracy: 0.650

## DQN Experiments
- Total DQN experiments: 81
- Best configuration:
  - Learning rate: 0.001
  - Epsilon: 0.3
  - Batch size: 128.0
  - Memory size: 10000.0
  - Sharpe ratio: 0.859

## Baseline Comparisons
- buy_hold: 0.080 return, 0.579 Sharpe
- random: -0.014 return, 0.076 Sharpe
- technical_analysis: 0.055 return, 0.396 Sharpe
- lstm_only: 0.059 return, 0.497 Sharpe
- rl_only: 0.037 return, 0.304 Sharpe
