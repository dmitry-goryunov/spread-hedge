import numpy as np
from scipy.stats import norm
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Kirk Spread Hedge", layout="wide")
st.title("Kirk Spread Option — Static Vega Hedge Comparison")

# ── Sidebar parameters ─────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")
    S1_0   = st.number_input("S₁ spot",   value=25.0, step=1.0)
    S2_0   = st.number_input("S₂ spot",   value=25.0, step=1.0)
    sigma1 = st.slider("σ₁", 0.05, 1.0, 0.30, 0.01)
    sigma2 = st.slider("σ₂", 0.05, 1.0, 0.30, 0.01)
    rho    = st.slider("ρ (correlation)", -0.99, 0.99, 0.90, 0.01)
    T      = st.slider("T (years)", 0.1, 3.0, 1.0, 0.1)
    alpha  = st.slider("α (strangle OTM factor)", 1.01, 2.5, 1.30, 0.01)
    N_G    = st.slider("Grid size", 15, 50, 35, 5)

# ── Pricing & Greeks ───────────────────────────────────────────────
_N, _n = norm.cdf, norm.pdf


def _d1d2(S, K_, sig):
    d1 = (np.log(S / K_) + 0.5 * sig**2 * T) / (sig * sqrt(T))
    return d1, d1 - sig * sqrt(T)


def bs_call(S, K_, sig):
    d1, d2 = _d1d2(S, K_, sig)
    return dict(
        price=S * _N(d1) - K_ * _N(d2),
        delta=_N(d1),
        gamma=_n(d1) / (S * sig * sqrt(T)),
        vega =S * _n(d1) * sqrt(T),
    )


def bs_put(S, K_, sig):
    d1, d2 = _d1d2(S, K_, sig)
    return dict(
        price=K_ * _N(-d2) - S * _N(-d1),
        delta=_N(d1) - 1.0,
        gamma=_n(d1) / (S * sig * sqrt(T)),
        vega =S * _n(d1) * sqrt(T),
    )


def kirk(S1, S2):
    sig_k = sqrt(sigma1**2 - 2*rho*sigma1*sigma2 + sigma2**2)
    sqT   = sqrt(T)
    d1    = (np.log(S1 / S2) + 0.5 * sig_k**2 * T) / (sig_k * sqT)
    d2    = d1 - sig_k * sqT
    vk    = S1 * _n(d1) * sqT
    return dict(
        price   =S1 * _N(d1) - S2 * _N(d2),
        delta1  = _N(d1),
        delta2  =-_N(d2),
        gamma11 =_n(d1) / (S1 * sig_k * sqT),
        gamma22 =S1 * _n(d1) / (S2**2 * sig_k * sqT),
        vega1   =vk * (sigma1 - rho * sigma2) / sig_k,
        vega2   =vk * (sigma2 - rho * sigma1) / sig_k,
    )


def _ols_pair(sp_d, sp_g, sp_v, inst_d, inst_g, inst_v):
    n    = len(sp_d)
    sc_d = max(float(np.abs(sp_d).max()), 1e-10)
    sc_g = max(float(np.abs(sp_g).max()), 1e-10)
    sc_v = max(float(np.abs(sp_v).max()), 1e-10)
    A = np.zeros((3 * n, 2))
    b = np.zeros(3 * n)
    A[:n,    0] = inst_d / sc_d;  A[:n, 1] = 1.0 / sc_d;  b[:n]    = -sp_d / sc_d
    A[n:2*n, 0] = inst_g / sc_g;                           b[n:2*n] = -sp_g / sc_g
    A[2*n:,  0] = inst_v / sc_v;                           b[2*n:]  = -sp_v / sc_v
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(c[0]), float(c[1])


# ── Grid & Greeks ──────────────────────────────────────────────────
s_lo = min(S1_0, S2_0) * 0.70
s_hi = max(S1_0, S2_0) * 1.40
s1g  = np.linspace(s_lo, s_hi, N_G)
s2g  = np.linspace(s_lo, s_hi, N_G)
s1pf, s2pf = np.meshgrid(s1g, s2g, indexing="ij")
s1f, s2f   = s1pf.ravel(), s2pf.ravel()

sp   = kirk(s1f, s2f)
K1c, K1p = S1_0 * alpha, S1_0 / alpha
K2c, K2p = S2_0 * alpha, S2_0 / alpha

# Strategy A
c1a = bs_call(s1f, S1_0, sigma1)
p2a = bs_put( s2f, S2_0, sigma2)
h1a, nF1a = _ols_pair(sp["delta1"], sp["gamma11"], sp["vega1"], c1a["delta"], c1a["gamma"], c1a["vega"])
h2a, nF2a = _ols_pair(sp["delta2"], sp["gamma22"], sp["vega2"], p2a["delta"], p2a["gamma"], p2a["vega"])
V1a = sp["vega1"] + h1a * c1a["vega"]
V2a = sp["vega2"] + h2a * p2a["vega"]

# Strategy B
c1b = bs_call(s1f, K1c, sigma1);  p1b = bs_put(s1f, K1p, sigma1)
c2b = bs_call(s2f, K2c, sigma2);  p2b = bs_put(s2f, K2p, sigma2)
str1 = {k: c1b[k] + p1b[k] for k in ("delta", "gamma", "vega")}
str2 = {k: c2b[k] + p2b[k] for k in ("delta", "gamma", "vega")}
h1b, nF1b = _ols_pair(sp["delta1"], sp["gamma11"], sp["vega1"], str1["delta"], str1["gamma"], str1["vega"])
h2b, nF2b = _ols_pair(sp["delta2"], sp["gamma22"], sp["vega2"], str2["delta"], str2["gamma"], str2["vega"])
V1b = sp["vega1"] + h1b * str1["vega"]
V2b = sp["vega2"] + h2b * str2["vega"]

# ── Costs ──────────────────────────────────────────────────────────
sp0   = kirk(S1_0, S2_0)
pr_sp = float(np.squeeze(sp0["price"]))

pr_c1a = float(np.squeeze(bs_call(S1_0, S1_0, sigma1)["price"]))
pr_p2a = float(np.squeeze(bs_put( S2_0, S2_0, sigma2)["price"]))
net_A  = pr_sp + h1a * pr_c1a + h2a * pr_p2a

pr_c1b = float(np.squeeze(bs_call(S1_0, K1c, sigma1)["price"]))
pr_p1b = float(np.squeeze(bs_put( S1_0, K1p, sigma1)["price"]))
pr_c2b = float(np.squeeze(bs_call(S2_0, K2c, sigma2)["price"]))
pr_p2b = float(np.squeeze(bs_put( S2_0, K2p, sigma2)["price"]))
net_B  = pr_sp + h1b * (pr_c1b + pr_p1b) + h2b * (pr_c2b + pr_p2b)

# ── Scoring ────────────────────────────────────────────────────────
V1_raw, V2_raw = sp["vega1"], sp["vega2"]
rmse_raw = float(np.sqrt(np.mean(V1_raw**2))) + float(np.sqrt(np.mean(V2_raw**2)))


def _score(V1, V2, net_cost, label):
    r1  = float(np.sqrt(np.mean(V1**2)))
    r2  = float(np.sqrt(np.mean(V2**2)))
    tot = r1 + r2
    return pd.Series({
        "rmse_ν1":   r1,
        "rmse_ν2":   r2,
        "rmse_total": tot,
        "max_ν1":    float(np.abs(V1).max()),
        "max_ν2":    float(np.abs(V2).max()),
        "net_cost":  net_cost,
        "efficiency": tot / max(abs(net_cost), 1e-8),
        "eff_%":     (1 - tot / rmse_raw) * 100,
    }, name=label)


scores = pd.DataFrame([
    _score(V1a, V2a, net_A, f"A  ATM@{S1_0:.0f}"),
    _score(V1b, V2b, net_B, f"B  α={alpha:.2f}"),
])
scores.index.name = "Strategy"

# ══════════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════════

# ── Row 1: hedge ratios + score table ─────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Hedge Ratios")
    hr = pd.DataFrame({
        "h (option)": [h1a, h2a, h1b, h2b],
        "nF (futures)": [nF1a, nF2a, nF1b, nF2b],
    }, index=[
        f"A  S₁ call@{S1_0:.0f}", f"A  S₂ put@{S2_0:.0f}",
        f"B  S₁ strangle@{K1p:.1f}/{K1c:.1f}",
        f"B  S₂ strangle@{K2p:.1f}/{K2c:.1f}",
    ])
    st.dataframe(hr.style.format("{:+.4f}"))

with col2:
    st.subheader("Hedge Quality")
    winner = scores.idxmin()
    st.dataframe(scores.style.format("{:+.4f}").highlight_min(axis=0, color="#d4edda"))

# ── Row 2: cost breakdown ──────────────────────────────────────────
st.subheader("Cost Breakdown by Instrument")
nan = float("nan")

def _crow(label, ra, ua, rb, ub):
    return {
        "instrument": label,
        "A  ratio":   ra  if ra  is not None else nan,
        "A  unit px": ua  if ua  is not None else nan,
        "A  cost":    ra * ua if ra is not None else nan,
        "B  ratio":   rb  if rb  is not None else nan,
        "B  unit px": ub  if ub  is not None else nan,
        "B  cost":    rb * ub if rb is not None else nan,
    }

cost_rows = [
    _crow(f"S₁ call  A:ATM@{S1_0:.0f}  B:OTM@{K1c:.2f}", h1a, pr_c1a, h1b, pr_c1b),
    _crow(f"S₁ put   A:—         B:OTM@{K1p:.2f}",        None, None,   h1b, pr_p1b),
    _crow(f"S₂ call  A:—         B:OTM@{K2c:.2f}",        None, None,   h2b, pr_c2b),
    _crow(f"S₂ put   A:ATM@{S2_0:.0f}  B:OTM@{K2p:.2f}", h2a, pr_p2a,  h2b, pr_p2b),
    {"instrument": "hedge subtotal",
     "A  ratio": nan, "A  unit px": nan, "A  cost": h1a*pr_c1a + h2a*pr_p2a,
     "B  ratio": nan, "B  unit px": nan, "B  cost": h1b*(pr_c1b+pr_p1b) + h2b*(pr_c2b+pr_p2b)},
    {"instrument": "spread option",
     "A  ratio": nan, "A  unit px": nan, "A  cost": pr_sp,
     "B  ratio": nan, "B  unit px": nan, "B  cost": pr_sp},
    {"instrument": "NET COST",
     "A  ratio": nan, "A  unit px": nan, "A  cost": net_A,
     "B  ratio": nan, "B  unit px": nan, "B  cost": net_B},
]
ct = pd.DataFrame(cost_rows).set_index("instrument")
st.dataframe(ct.style.format(lambda x: f"{x:+.4f}" if pd.notna(x) else "—"))

# ── Row 3: heatmaps ────────────────────────────────────────────────
st.subheader("Residual Vega Heatmaps")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle(
    f"Residual Vega  σ₁={sigma1}  σ₂={sigma2}  ρ={rho}  T={T}  α={alpha:.2f}",
    fontsize=11,
)

def _hm(ax, data, title, vlim):
    im = ax.imshow(
        data.reshape(N_G, N_G), origin="lower",
        extent=[s_lo, s_hi, s_lo, s_hi], aspect="auto",
        vmin=-vlim, vmax=vlim, cmap="RdBu_r",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("S₁"); ax.set_ylabel("S₂")
    plt.colorbar(im, ax=ax, shrink=0.85)

v1_lim = max(np.abs(V1a).max(), np.abs(V1b).max())
v2_lim = max(np.abs(V2a).max(), np.abs(V2b).max())

_hm(axes[0, 0], V1a, f"A — Residual ν₁  (ATM call@{S1_0:.0f})", v1_lim)
_hm(axes[0, 1], V1b, f"B — Residual ν₁  (strangle {K1p:.1f}/{K1c:.1f})", v1_lim)
_hm(axes[1, 0], V2a, f"A — Residual ν₂  (ATM put@{S2_0:.0f})", v2_lim)
_hm(axes[1, 1], V2b, f"B — Residual ν₂  (strangle {K2p:.1f}/{K2c:.1f})", v2_lim)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Row 4: alpha sensitivity ───────────────────────────────────────
st.subheader("Strangle OTM Sensitivity (α sweep)")

alphas_range = np.linspace(1.02, 2.5, 80)
rmse_B_sweep = []

for a in alphas_range:
    kc1, kp1 = S1_0 * a, S1_0 / a
    kc2, kp2 = S2_0 * a, S2_0 / a
    cb1 = bs_call(s1f, kc1, sigma1); pb1 = bs_put(s1f, kp1, sigma1)
    cb2 = bs_call(s2f, kc2, sigma2); pb2 = bs_put(s2f, kp2, sigma2)
    s1_ = {k: cb1[k] + pb1[k] for k in ("delta", "gamma", "vega")}
    s2_ = {k: cb2[k] + pb2[k] for k in ("delta", "gamma", "vega")}
    hh1, _ = _ols_pair(sp["delta1"], sp["gamma11"], sp["vega1"], s1_["delta"], s1_["gamma"], s1_["vega"])
    hh2, _ = _ols_pair(sp["delta2"], sp["gamma22"], sp["vega2"], s2_["delta"], s2_["gamma"], s2_["vega"])
    r1 = float(np.sqrt(np.mean((sp["vega1"] + hh1 * s1_["vega"])**2)))
    r2 = float(np.sqrt(np.mean((sp["vega2"] + hh2 * s2_["vega"])**2)))
    rmse_B_sweep.append(r1 + r2)

rmse_A_ref = float(scores.loc[f"A  ATM@{S1_0:.0f}", "rmse_total"])
best_idx   = int(np.argmin(rmse_B_sweep))
best_alpha = alphas_range[best_idx]

fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(alphas_range, rmse_B_sweep, color="tab:orange", label="Strategy B (strangle)")
ax2.axhline(rmse_A_ref, color="tab:blue", linestyle="--",
            label=f"Strategy A (ATM, RMSE={rmse_A_ref:.4f})")
ax2.axvline(alpha, color="gray", linestyle=":", label=f"current α={alpha:.2f}")
ax2.axvline(best_alpha, color="tab:orange", linestyle=":",
            label=f"optimal α={best_alpha:.2f}  (RMSE={rmse_B_sweep[best_idx]:.4f})")
ax2.set_xlabel("α  (call @ S·α,  put @ S/α)")
ax2.set_ylabel("RMSE total  (ν₁ + ν₂)")
ax2.set_title("Hedge quality vs strangle width")
ax2.legend(fontsize=9)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.caption(
    f"Optimal α={best_alpha:.3f}  "
    f"→  call@{S1_0*best_alpha:.2f} / put@{S1_0/best_alpha:.2f}  "
    f"(RMSE={rmse_B_sweep[best_idx]:.4f}  vs  A={rmse_A_ref:.4f})"
)
