# Kirk Spread Option — Static Vega Hedge Comparison

Compares two static vega hedging strategies for a Margrabe (zero-strike) spread option on two correlated assets.

## Problem

A spread option `max(S₁ − S₂, 0)` has vega exposure to both legs simultaneously. A static hedge must work across a wide range of future spot scenarios, not just at inception. The notebook asks: **is an ATM single-option hedge or an OTM strangle a better vega hedge?**

## Strategies

| | Strategy A | Strategy B |
|---|---|---|
| S₁ hedge | ATM call @ S₁ | Strangle: call @ S₁·α, put @ S₁/α |
| S₂ hedge | ATM put @ S₂ | Strangle: call @ S₂·α, put @ S₂/α |
| Delta | Futures on each leg | Futures on each leg |

The strangle in B is log-symmetric: `√(K_call · K_put) = S`, so the geometric mean strike equals the forward.

## Method

Hedge ratios are found by **OLS across a 35×35 spot grid** `(S₁, S₂) ∈ [18, 35]²`, simultaneously minimising residual delta, gamma, and vega (each normalised to contribute equally):

```
min_h  ‖ Greeks_spread + h · Greeks_instrument + nF · Greeks_future ‖²
```

## Outputs

- **Residual vega heatmaps** — 2×2 grid showing where each strategy leaves unhedged vega across spot scenarios
- **Cost breakdown** — per-instrument table (ratio, unit price, weighted cost) side by side for A and B
- **Scoring table** — RMSE, max error, net cost, cost efficiency, and % of vega risk removed vs unhedged baseline
- **Alpha sensitivity curve** — total RMSE vs OTM factor α, showing the optimal strangle width

## Key finding (default parameters)

B (strangle) wins on 6 of 7 metrics. The decisive advantage is **17% lower worst-case vega error**. It costs ~2× more than A because OTM options are cheaper per unit, so the hedge collects less premium. However B delivers more hedge per dollar spent (efficiency: 1.70 vs 3.64).

## Parameters

All parameters are in the second cell and easy to change:

```python
S1_0, S2_0 = 25.0, 25.0   # spot prices
sigma1, sigma2 = 0.30, 0.30
rho    = 0.90              # correlation
T      = 1.0               # maturity (years)
alpha  = 1.30              # strangle OTM factor (B only)
```

## Requirements

```
numpy
scipy
pandas
matplotlib
```

```bash
pip install numpy scipy pandas matplotlib
jupyter notebook kirk_spread_hedge.ipynb
```
