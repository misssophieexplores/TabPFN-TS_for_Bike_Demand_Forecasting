# Aggregation Methodology: % Change Across Cities/Horizons

**Rule:** compute the ratio (degraded ÷ clean) on the smallest unit —
one city, one horizon — then aggregate those per-unit ratios with a
**geometric mean**.

Previous discussion on order of operations is a known issue in the forecasting literature, referred to as
**"ratio of means vs. mean of ratios"**.

## Why geometric mean

| | Clean MASE | Degraded MASE | ratio |
|---|---|---|---|
| City A | 1.0 | 2.0 | 2.0 |
| City B | 2.0 | 1.0 | 0.5 |

City A's error doubled, City B's halved, which is proportionally equal with
opposite effects.

- Arithmetic mean of the ratios: (2.0 + 0.5)/2 = 1.25 → implies a net
  25% increase, i.e. reads as "the model on average overestimates".
- Geometric mean of the ratios: √(2.0 × 0.5) = 1.0 → reflects that the
  two effects offset exactly.

## Citation

Davydenko, A. & Fildes, R. (2013). *Measuring forecasting accuracy: The case of judgmental adjustments to SKU-level demand forecasts.* International Journal of Forecasting, 29(3), 510–522.
- Worked example with ratios r₁ = 0.5, r₂ = 2.0 showing the same
  divergence between arithmetic and geometric aggregation.
- Proposes AvgRelMAE: geometric mean of per-series ratios.
- Builds on Armstrong & Collopy (1992), which tested error measures on
  data from the original M-competition (Makridakis et al., 1982).
