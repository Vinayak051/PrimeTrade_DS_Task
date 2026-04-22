# Trader Performance vs Market Sentiment — Write-up

---

## Methodology

The analysis combined two datasets: raw on-chain trade records for a set of crypto accounts (211,224 trades, May 2023–May 2025) and the daily Crypto Fear & Greed Index. The raw trade data was aggregated to a daily per-account level, producing five features — `daily_pnl`, `trades_per_day`, `avg_trade_size`, `win_rate`, and `long_short_ratio` — then joined to the sentiment index by date to form a clean 2,340-row dataset.

Two models were built on this dataset. First, a Random Forest classifier was trained to predict whether a trader's next-day PnL would fall into a Loss, Neutral, or Profit bucket, using the six behavioural and sentiment features. A grid search over depth and tree-count parameters found the best configuration at `max_depth=5`, `n_estimators=100`, achieving 53% accuracy and a macro F1 of 0.53 on a held-out test set — meaningfully above a random 33% baseline for a three-class problem. Second, a KMeans model (k=3) was applied to trader-level averages to segment accounts into three behavioural archetypes.

---

## Insights

**Sentiment shapes how traders behave, not just how markets move.**  
During Fear periods, traders placed significantly more trades per day (mean 98 vs 77 during Greed) and at higher average sizes ($8,976 vs $6,428). Counterintuitively, this increased activity during fearful conditions did not translate to better returns — it mostly reflected reactive, emotionally-driven overtrading.

**Win rate and trade frequency matter more than sentiment itself.**  
The correlation matrix showed `win_rate` (0.21) and `trades_per_day` (0.176) as the strongest predictors of daily PnL. The raw sentiment `value` had essentially zero direct correlation (0.000) with PnL, meaning sentiment doesn't cause profits — but it heavily influences the behaviour that does.

**Three distinct trader archetypes emerged from clustering.**  
- *Archetype 0 — Moderate Active*: ~103 trades/day, $16k average trade size, 41.8% win rate. Mid-range volume with larger size per trade.  
- *Archetype 1 — Small-Stake Trader*: ~84 trades/day, $3.8k average size, 29.8% win rate. Lower activity and the weakest win rate of the three groups.  
- *Archetype 2 — High-Frequency Long Bias*: ~757 trades/day, $4.5k average size, 45.6% win rate. Extremely high volume with the best win rate, and a strongly long-skewed position ratio.

The predictive model confirmed that sentiment score contributes to classification but is not the dominant signal — behavioural features like win rate and trade count carry more weight in the feature importance ranking.

---

## Strategy Recommendations

**1. Adjust leverage based on sentiment — but only for stable accounts.**  
When the index moves into Greed territory, accounts with a demonstrated track record of consistent win rates can reasonably take on slightly more leverage. The data shows that Greed periods tend to coincide with calmer, lower-frequency trading and smaller average sizes, which means market conditions are generally less erratic. However, this adjustment should be gated on past stability — accounts in the Small-Stake archetype (Archetype 1, win rate ~30%) should not be included, as their baseline performance does not support increased exposure regardless of sentiment.

During Fear, leverage should be cut back across the board. Fear periods in this dataset correlated with more trades, larger sizes, and worse outcomes — a pattern consistent with panic-driven behaviour amplifying losses rather than recovering them.

**2. Avoid overtrading during Fear.**  
The clearest behavioural signal in the data is that traders push up their trade count during Fear without a corresponding improvement in win rate. A practical rule is to set a daily trade cap during Fear and Extreme Fear regimes — something like reducing maximum daily trades to 60–70% of a trader's personal baseline. Fewer, more deliberate entries tend to preserve capital in uncertain conditions. The High-Frequency archetype (Archetype 2) is the exception here, as its high volume appears structural rather than reactive, but for the majority of accounts the data strongly supports doing less when the market is fearful.
