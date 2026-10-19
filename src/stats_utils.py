"""Statistical helpers shared by the RQ scripts (one implementation, used everywhere).

References: Wilson (1927) score interval; Pearson χ² with a Monte Carlo permutation
p-value; Cramér (1946) V interpreted with Cohen (1988) df-adjusted benchmarks;
ε² for Kruskal–Wallis (Tomczak & Tomczak, 2014) with Cohen (1988) η² benchmarks;
Cliff (1993) δ with Romano et al. (2006) thresholds."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportion_confint

from config import SEED

N_PERM = 10_000
N_BOOT = 2000


def wilson(k, n):
    """95% Wilson score interval for a proportion, in percent: (p, lo, hi)."""
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return 100 * k / n, 100 * lo, 100 * hi


def gini(counts):
    """Gini coefficient of a count distribution (0 = even, 1 = concentrated in one unit)."""
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n < 2 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def chi2_stat(obs):
    exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
    return float(((obs - exp) ** 2 / exp).sum()), exp


def permutation_chi2(groups, cats, n_perm=N_PERM, seed=SEED):
    """Pearson χ² with a Monte Carlo permutation p-value (valid with sparse cells):
    group labels are shuffled n_perm times, keeping both margins fixed.
    Also returns Cramér's V and adjusted standardised residuals."""
    g_codes, g_levels = pd.factorize(groups, sort=True)
    c_codes, c_levels = pd.factorize(cats, sort=True)
    r, c = len(g_levels), len(c_levels)
    table = lambda g: np.bincount(g * c + c_codes, minlength=r * c).reshape(r, c)
    obs = table(g_codes)
    stat, exp = chi2_stat(obs)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        hits += chi2_stat(table(rng.permutation(g_codes)))[0] >= stat - 1e-9
    p = (hits + 1) / (n_perm + 1)
    n = obs.sum()
    df_star = min(r, c) - 1
    v = np.sqrt(stat / (n * df_star))
    rs, cs = obs.sum(1, keepdims=True) / n, obs.sum(0, keepdims=True) / n
    adj = (obs - exp) / np.sqrt(exp * (1 - rs) * (1 - cs))
    return dict(
        stat=stat,
        df=(r - 1) * (c - 1),
        p=p,
        v=v,
        df_star=df_star,
        n=int(n),
        pct_expected_lt5=float((exp < 5).mean() * 100),
        min_expected=float(exp.min()),
        residuals=pd.DataFrame(adj, index=g_levels, columns=c_levels),
        observed=pd.DataFrame(obs, index=g_levels, columns=c_levels),
    )


def v_label(v, df_star):
    """Cohen (1988): w benchmarks 0.1/0.3/0.5, divided by sqrt(df*) for Cramér's V."""
    s = np.sqrt(df_star)
    return "negligible" if v < 0.1 / s else "small" if v < 0.3 / s else "medium" if v < 0.5 / s else "large"


def fmt_ci(p, lo, hi):
    return f"{p:.1f} [{lo:.1f}, {hi:.1f}]"


def eps2(H, n):
    """ε² for Kruskal–Wallis (Tomczak & Tomczak, 2014): H(n+1)/(n²−1)."""
    return H * (n + 1) / (n**2 - 1)


def eps2_label(e):
    """Cohen (1988) η² benchmarks 0.01 / 0.06 / 0.14."""
    return "negligible" if e < 0.01 else "small" if e < 0.06 else "medium" if e < 0.14 else "large"


def cliffs_delta(x, y):
    """δ = P(X > Y) − P(X < Y), computed from the Mann–Whitney U statistic of x."""
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2 * u / (len(x) * len(y)) - 1


def delta_label(dl):
    a = abs(dl)  # Romano et al. (2006)
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


def median_ci_exact(x, conf=0.95):
    """Distribution-free CI for the median from order statistics (binomial method;
    Conover, Practical Nonparametric Statistics). Deterministic; coverage >= conf.
    Returns (lo, hi, achieved_coverage)."""
    from scipy.stats import binom

    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    alpha = 1 - conf
    k = int(binom.ppf(alpha / 2, n, 0.5))  # P(B <= k-1) <= alpha/2 for the largest such k
    while k > 0 and binom.cdf(k - 1, n, 0.5) > alpha / 2:
        k -= 1
    k = max(k, 1)
    coverage = 1 - 2 * binom.cdf(k - 1, n, 0.5)
    return x[k - 1], x[n - k], coverage


def boot_median_ci(x, rng, n_boot=N_BOOT):
    x = np.asarray(x)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return np.percentile(meds, [2.5, 97.5])
