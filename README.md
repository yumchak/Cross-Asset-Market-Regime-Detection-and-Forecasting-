# Cross-Asset-Market-Regime-Detection-and-Forecasting

## What This Project Does

This project builds an end-to-end quantitative data science pipeline 
that identifies hidden structural patterns in complex time series data 
using unsupervised machine learning, then tests whether those patterns 
improve the accuracy of a supervised classification model.

The application is financial markets, but the core methodology — finding 
hidden regime shifts in data without labelled examples — is directly 
transferable to any domain where an organisation faces structural changes 
in demand, behaviour, or resource needs.


---

## Research Question

Can a Gaussian Hidden Markov Model identify latent market regimes from 
cross-asset financial data without any labelled training data, and does 
conditioning an XGBoost direction classifier on the current regime improve 
walk-forward accuracy versus a baseline model?

---

## Data

Five daily time series from Yahoo Finance via yfinance, January 2010 
to present:

- SPY — US equity market (S&P 500 ETF)
- TLT — Long-duration US Treasury bonds
- GLD — Gold (safe haven proxy)
- VIX — CBOE Volatility Index (market fear gauge)
- HYG — High-yield corporate bonds (credit risk proxy)

---

## Methods

**Feature Engineering**
Four rolling features from daily closing prices: SPY 20-day volatility,
SPY 20-day momentum, VIX level, HYG 20-day return.
Normalised with StandardScaler before model fitting.

**Gaussian HMM (unsupervised)**
Fitted to normalised feature matrix with 2 hidden states via 
Expectation-Maximisation. Assigns a regime label to every trading 
day without any labelled training data.

**Statistical Validation**
T-test confirms regimes do not differ in average return (p = 0.127).
Levene test confirms regimes differ significantly in volatility 
(p ≈ 0.000, 2.33x ratio).

**XGBoost Walk-Forward Classification**
Binary classifier predicting next-week SPY return direction.
Train on 3 years, predict next 6 months, roll forward across 26 folds.
Model A uses 4 raw features. Model B adds the HMM regime label as a 
5th feature. Evaluated on accuracy and F1 score.

---

## Key Results

### Regime Detection

| Metric | Calm Regime | Stressed Regime |
|--------|-------------|-----------------|
| Trading days | 2,691 (65%) | 1,429 (35%) |
| Average VIX | 15.1 | 24.8 |
| Daily return std | 0.0068 | 0.0158 |
| Volatility ratio | — | 2.33x higher |

The model correctly identifies known stress periods including the 2011 
European debt crisis, 2015-16 China growth scare, March 2020 Covid 
crash, and 2022 Fed rate hike bear market — without being given any 
information about real-world events.

### Forecasting Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Model A (no regime) | 56.7% | 0.683 |
| Model B (with regime) | 56.9% | 0.684 |
| Improvement | +0.18pp | +0.001 |
| Naive baseline | 61.4% | — |

### Feature Importance (Model B)

| Feature | Importance | Rank |
|---------|------------|------|
| Regime (HMM label) | 0.256 | 1st |
| VIX Level | 0.199 | 2nd |
| SPY Volatility | 0.185 | 3rd |
| SPY Momentum | 0.183 | 4th |
| HYG Return | 0.177 | 5th |

The regime label ranked 1st out of 5 features despite the modest 
accuracy improvement, suggesting it captures a compressed summary 
of market conditions that individual features cannot fully replicate.
---

## Visualisations

### 1. SPY Price with HMM-Detected Regimes
![Regime Chart](images/regime_chart.png)

### 2. Return Distributions by Regime
![Return Distributions](images/return_distributions.png)

### 3. XGBoost Feature Importance
![Feature Importance](images/feature_importance.png)

### 4. Walk-Forward Model Comparison
![Model Comparison](images/model_comparison.png)

---

## Interpretation

The modest accuracy improvement (+0.18pp) is consistent with the 
efficient market hypothesis. In liquid markets, large and persistent 
forecasting edges are quickly arbitraged away. The more meaningful 
finding is that:

1. The HMM successfully discovers real market structure without labels
2. The regime label is the most informative single feature available
3. The regimes are statistically validated with extremely high confidence

This project is not claiming to predict markets profitably. It is 
demonstrating a rigorous end-to-end pipeline for unsupervised 
structural discovery followed by supervised evaluation — a methodology 
with broad applicability.

---
## Limitations

- Only 5 assets and 4 features used
- 2-state HMM may oversimplify market dynamics
- No transaction costs or slippage modelled
- Walk-forward accuracy below naive baseline (61.4%)
- HMM regime labels sensitive to random seed without fixing 
  random_state

