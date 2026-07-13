import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def p_adic_valuation(x, p=2):
    """
    Compute the p-adic valuation of a number: v_p(x) = max{k: p^k | x}
    """
    if x == 0:
        return 0
    x = int(round(abs(x) * 1000))  # scale to integer
    if x == 0:
        return 0
    v = 0
    while x % p == 0:
        v += 1
        x //= p
    return v

def de_rham_witt_complex(series, p=2, tick_size=0.01):
    """
    Construct de Rham-Witt complex from a discrete price grid.
    Returns the torsion invariant (microstructure noise measure).
    """
    if len(series) < 5:
        return 0
    # Discretise to price grid
    prices = np.cumsum(series)
    # Quantize to tick size
    quantized = np.round(prices / tick_size) * tick_size
    # Compute p-adic valuations of price differences
    differences = np.diff(quantized)
    valuations = np.array([p_adic_valuation(d, p) for d in differences])
    # Torsion invariant = sum of valuations (microstructure noise)
    torsion = np.sum(valuations)
    # Also compute the variance of valuations (higher variance = more microstructure)
    var_torsion = np.var(valuations) if len(valuations) > 0 else 0
    # Score = torsion + var_torsion (higher = more microstructure)
    score = torsion + var_torsion
    return score

def crystalline_score(returns, macro_df, p=2, tick_size=0.01):
    """
    Compute per-ETF crystalline cohomology score.
    Higher score = more microstructure structure / noise.
    """
    if len(returns) < 10 or macro_df is None or len(macro_df) < 10:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 10:
        return 0.0
    # Compute p-adic cohomology invariant
    torsion = de_rham_witt_complex(returns, p, tick_size)
    # Macro factor modulates the score
    macro_factor = compute_composite_macro_factor(macro_df)[-1]
    score = torsion * (1 + macro_factor * 0.5)
    return float(score)
