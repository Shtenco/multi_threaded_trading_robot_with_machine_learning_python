# Multi-Threaded Trading Robot with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MetaTrader5](https://img.shields.io/badge/MetaTrader-5-green.svg)](https://www.metatrader5.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An intelligent trading system combining machine learning and parallel computing for automated Forex trading. Hybrid Python + MetaTrader 5 architecture where Python handles analytics and model training, while MT5 executes trading operations.

**Current accuracy on examination dataset: 64-65%** with average profit significantly exceeding average loss.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Technical Details](#technical-details)
- [Usage](#usage)
- [Results](#results)
- [Future Development](#future-development)
- [Author](#author)
- [License](#license)

## 🎯 Overview

This project implements a sophisticated multi-threaded algorithmic trading system that leverages:

- **Machine Learning**: XGBoost with two-level ensembling (Gradient Boosting + Bagging)
- **Parallel Computing**: Multiple currency pairs processed simultaneously in separate threads
- **Advanced Risk Management**: Portfolio-level risk allocation with dynamic position sizing
- **Data Augmentation**: 5x data multiplication through noise, scaling, shifting, and inversion
- **Feature Engineering**: 25+ technical indicators plus synthetic feature generation

The system is designed to trade multiple currency pairs simultaneously while maintaining strict portfolio risk limits and adapting to changing market conditions through intelligent clustering of market regimes.

## ⚡ Key Features

### Machine Learning Pipeline

- **XGBoost Classifier** with hyperparameter optimization via GridSearchCV
- **Gaussian Mixture Models** for market regime clustering (6 components)
- **Recursive Feature Elimination** (RFECV) for automatic feature selection
- **BaggingClassifier** wrapper for meta-ensembling
- **5-fold Cross-Validation** for robust model evaluation

### Data Processing

#### Technical Indicators (25+)
- **Momentum**: RSI, MACD, Momentum, ROC, TRIX
- **Volatility**: ATR, Bollinger Bands, Keltner Channels
- **Oscillators**: Stochastic, Williams %R, CCI, MFI
- **Trend**: Aroon, Efficiency Ratio, Fractal Analysis
- **Volume**: Volume SMA Ratio, Price-Volume Trend, Chaikin Oscillator
- **Cyclical**: Hour Sin/Cos, Day of Week Sin/Cos

#### Data Augmentation Strategy
1. **Noise Addition**: Gaussian noise (σ = 0.01) to simulate microstructure
2. **Time Shifting**: +1 hour offset to capture temporal patterns
3. **Scaling**: Random scaling (0.9-1.1) for different volatility regimes
4. **Inversion**: Price inversion to learn mirror patterns

### Risk Management

- **Portfolio Risk Limit**: $1000 total across all positions
- **Dynamic Position Sizing**: ATR-based lot calculation per instrument
- **Fixed Stop Loss**: 300 pips with 800 pips take profit (1:2.67 risk/reward)
- **Spread Filter**: Maximum 35 pips spread allowed
- **Position Limits**: Maximum 6 concurrent positions

### Multi-Threading Architecture

- **Parallel Processing**: 7 currency pairs in separate threads
- **Thread-Safe Logging**: Queue-based logging system
- **Position Synchronization**: Automatic position flag updates
- **Graceful Shutdown**: Coordinated thread termination

## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     MAIN THREAD                              │
│  • Portfolio Initialization                                  │
│  • Position Size Calculation                                 │
│  • Thread Coordination                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼───────┐ ┌────▼─────┐ ┌─────▼──────┐
│  EURUSD THREAD│ │GBPUSD THD│ │ AUDUSD THD │  ... (7 threads)
│               │ │          │ │            │
│ 1. Data Load │ │          │ │            │
│ 2. Augment   │ │          │ │            │
│ 3. Label     │ │          │ │            │
│ 4. Cluster   │ │          │ │            │
│ 5. Feature   │ │          │ │            │
│ 6. Train     │ │          │ │            │
│ 7. Trade ↓   │ │          │ │            │
└───────┬───────┘ └────┬─────┘ └─────┬──────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │      GLOBAL RESOURCES        │
        │  • POSITION_SIZES (Dict)     │
        │  • SYMBOL_TRADES (Dict)      │
        │  • log_queue (Queue)         │
        │  • all_symbols_done (Flag)   │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │     MetaTrader 5 API         │
        │  • Quote Retrieval           │
        │  • Order Execution           │
        │  • Position Monitoring       │
        └──────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- MetaTrader 5 Terminal
- Windows OS (MT5 requirement)
- Active MT5 account (demo or live)

### Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21.0
pandas>=1.3.0
MetaTrader5>=5.0.4
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0
```

### Configuration

1. **Install MetaTrader 5** and open demo/live account

2. **Set terminal path** in code:
```python
TERMINAL_PATH: str = r"C:\Program Files\RoboForex MT5 Terminal\terminal64.exe"
```

3. **Configure risk parameters**:
```python
TOTAL_PORTFOLIO_RISK: float = 1000.0  # Total portfolio risk (USD)
MAX_OPEN_TRADES: int = 6              # Maximum positions
```

4. **Select currency pairs**:
```python
symbols = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "EURGBP"]
```

## 📊 Technical Details

### Data Processing Pipeline
```python
# 1. Load historical data (2021-2024)
raw_data = retrieve_data(symbol)  # ~26,000 hourly bars

# 2. Create 25+ technical indicators
# RSI, MACD, ATR, Bollinger Bands, Stochastic, etc.

# 3. Data augmentation (5x increase)
augmented_data = augment_data(raw_data)  # ~130,000 examples

# 4. Labeling with real trading conditions
labeled_data = label_data(data, symbol)  # SL 300 pips, TP 800 pips

# 5. Feature clustering (GMM)
clustered_data = cluster_features_by_gmm(data, n_components=6)

# 6. Synthetic feature generation
generated_data = generate_new_features(data, num_features=10)

# 7. Feature selection (RFECV)
selected_data = feature_engineering(data, n_features_to_select=15)

# 8. XGBoost + Bagging training
model = train_xgboost_classifier(train_data, num_boost_rounds=1000)
```

### XGBoost Model Parameters
```python
xgb.XGBClassifier(
    objective='binary:logistic',
    max_depth=5,              # Prevent overfitting
    learning_rate=0.2,        # Learning rate
    n_estimators=300,         # Number of trees
    subsample=0.01,           # 1% data per tree (aggressive regularization)
    colsample_bytree=0.1,     # 10% features per tree
    reg_alpha=1,              # L1 regularization
    reg_lambda=1              # L2 regularization
)
```

### Position Size Calculation
```python
# Per instrument:
risk_per_instrument = TOTAL_PORTFOLIO_RISK / len(symbols)  # $142.86 for 7 pairs

# Risk per standard lot
risk_per_lot = 300 pips * tick_value

# Position size
volume = risk_per_instrument / risk_per_lot

# Normalize to broker requirements
normalized_volume = round(volume / lot_step) * lot_step
```

## 🚀 Usage

### Starting the System
```bash
python trading_robot.py
```

### Monitoring

The system outputs detailed logs to console:
```
=== Symbol Analysis EURUSD ===
Point: 1e-05
Trade tick value: 1.0
Risk per 1 lot at SL 300 pips: 300.00 USD
Calculated position size: 0.476190 lot
Normalized size: 0.48 lot
FINAL RISK: 144.00 USD

Accuracy for symbol EURUSD: 64.23%
Position opened EURUSD: Buy, lot=0.48, spread=2.1 pips
```

### Stopping the System

The system automatically terminates when `all_symbols_done = True` flag is set by any thread, or via Ctrl+C.

## 📈 Results

### Current Performance

- **Examination dataset accuracy**: 64-65%
- **Average profitable trade**: Significantly > average losing trade
- **Several days of trading results**: Positive (see screenshots in article)
- **Profitability simulations**: Encouraging at given win rate

### Trading Example
```
Symbol: EURUSD
Entry: 1.0850
Stop Loss: 1.0820 (300 pips)
Take Profit: 1.0930 (800 pips)
Volume: 0.48 lot
Risk: $144
Potential Profit: $384
Risk/Reward: 1:2.67
```

## 🔮 Future Development

### Near-Term Improvements

1. **Full MQL5 Migration**
   - Custom ML libraries in MQL5
   - Eliminate Python-MT5 API latency
   - 10-50x performance improvement

2. **Graphical Interface**
   - Real-time dashboard monitoring
   - Dynamic parameter management
   - Alert and notification system

3. **Enhanced Risk Management**
   - Real-time correlation analysis
   - Monte Carlo Value-at-Risk
   - Automatic hedging system

### Ambitious Goals

4. **GAN for Synthetic Data**
   - Generate 25-year price series
   - Preserve market statistical properties
   - Solve data scarcity problem

5. **Ensemble of 100 Models**
   - Specialization on market regimes
   - Meta-classifier for regimes
   - Unprecedented prediction accuracy

6. **Google Cloud Platform Integration**
   - Massively parallel training
   - 1000+ core virtual machines
   - Auto-scaling resources

## 👨‍💻 Author

**Yevgeniy Koshtenko**

- 🌍 Location: Kostanay, Kazakhstan
- 💼 Experience: 10 years in algorithmic trading (since 2016)
- 📚 Research: 100+ published articles on ML and trading
- 💻 Codebase: 680,000 lines MQL5 + 1,200,000 lines Python
- 🔬 Expertise: Python, MQL5, ML, Quantum Computing, Data Engineering

### Project Portfolio

- High-frequency arbitrage system (HFT)
- Computer vision for chart analysis (CNN)
- Quantum computing on IBM Quantum
- Central bank data mining + ML
- Biological neural networks for forecasting

### Connect

- 📧 Email: [koshtenco@gmail.com]
- 🔗 GitHub: [Shtenco]
- 📝 MQL5: [[your-mql5-profile](https://www.mql5.com/ru/users/koshtenko)]

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## ⚠️ Disclaimer

This trading robot is for educational purposes only. Trading in financial markets involves high risk of capital loss. The author is not responsible for any financial losses resulting from the use of this software. Always conduct thorough testing on demo accounts before using real funds.

## 🙏 Acknowledgments

- MetaQuotes for MetaTrader 5 platform
- MQL5 community for support and inspiration
- Developers of scikit-learn, XGBoost and other libraries

---

**⭐ If this project was useful to you, please star it on GitHub!**
