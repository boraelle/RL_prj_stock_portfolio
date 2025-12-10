# 강화학습기초 프로젝트 : Stock Portfolio Optimization
- PPO + Multi-Modal LSTM을 이용한 국내 섹터 ETF 포트폴리오 자동 배분 시스템

## Performance

- **Total Return**: 108.03%
- **Sharpe Ratio**: 1.17
- **Max Drawdown**: -16.56%
- **Period**: 2024.01 ~ 2025.11 (465 days)

## Requirements & environment
- Python 3.11.13
- mac os

```bash
pip install -r requirements_20251207.txt
```



## Quick Start

### 1. Collect Data
```bash
python collect_data_etf13.py
```

### 2. Train Model
```bash
# Standard training (150k steps, ~90 min)
python train.py

# Background run
nohup python train.py > train.log 2>&1 &
```

### 3. Check Results
```
results/
├── ppo_portfolio_final.zip      # Trained model
├── capital_history.csv          # Daily portfolio value
├── metrics.csv                  # Performance metrics
├── rl_performance.png
└── strategy_comparison.png
```

## Pre-trained Model
Download pre-trained model: [ppo_portfolio_final.zip] (https://github.com/boraelle/RL_prj_stock_portfolio/tree/main/src/results/ppo_portfolio_final.zip)

## Project Structure
```
├── collect_data_etf13.py    # Data collection
├── environment.py           # RL environment
├── models.py                # Multi-Modal LSTM
├── train.py                 # Training script
├── evaluate_december.py     # Evaluation
├── utils.py                 # Utilities
├── requirements.txt
└── results/                 # Output files
```

## Model Architecture

- **State**: Price history (20 days OHLCV) + 6 technical indicators + portfolio state
- **Action**: Target weights for 13 assets (12 ETFs + Cash)
- **Algorithm**: PPO with custom Multi-Modal LSTM feature extractor

## Assets

12 sector ETFs + Cash (2% annual return)
- TIGER 200: Energy, Materials, Industrials, Consumer Staples, Consumer Discretionary, Healthcare, Financials, IT
- KODEX: Semiconductor, Battery, Defense, Bio