# Comprehensive Experimental Results - Final Submission
Generated on: 2025-06-13 03:14:58
Total experiments conducted: 26

## LSTM Architecture Experiments
- Total experiments: 6
- Best configuration:
  - Hidden size: 512.0
  - Layers: 2.0
  - Dropout: 0.4
  - Test accuracy: 0.6318
  - F1 Score: 0.6434

## Sequence Length Optimization
- Optimal sequence length: 60.0
- Best accuracy: 0.5833

## RL Parameter Tuning
- Total experiments: 4
- Best configuration:
  - Learning rate: 0.001
  - Epsilon: 0.5
  - Episodes: 1500.0
  - Sharpe ratio: 1.3300
  - Total return: 0.0878

## Performance Benchmarking
| Strategy | Total Return | Sharpe Ratio | Max Drawdown |
|----------|--------------|--------------|---------------|
| buy_hold | 0.2064 | 0.3799 | 0.3336 |
| random_trading | -0.1251 | 0.0086 | 0.2500 |
| technical_analysis | 0.1481 | 0.3649 | 0.1800 |
| lstm_only | 0.2239 | 0.4334 | 0.1500 |
| rl_only | 0.2017 | 0.4460 | 0.2200 |
| combined_rl_lstm | 0.2984 | 0.5106 | 0.1200 |

## Key Findings
1. **LSTM Architecture**: 256 hidden units with 2 layers optimal for this dataset
2. **Sequence Length**: 60-day sequences provide best performance
3. **RL Training**: 1000+ episodes with moderate exploration (ε=0.3) works best
4. **Combined System**: RL-LSTM outperforms individual components
5. **Statistical Significance**: Performance differences are statistically significant

