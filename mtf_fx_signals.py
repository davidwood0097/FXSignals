
import pandas as pd
import numpy as np

# ---- Core helpers ----
def sma(series: pd.Series, length: int) -> pd.Series:
    length = int(max(1, length))
    return series.rolling(length, min_periods=1).mean()

def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series,
                  k_len: int, k_smooth: int, d_len: int):
    k_len = int(max(1, k_len))
    k_smooth = int(max(1, k_smooth))
    d_len = int(max(1, d_len))

    ll = low.rolling(k_len, min_periods=1).min()
    hh = high.rolling(k_len, min_periods=1).max()
    # Avoid division by zero
    rng = (hh - ll).replace(0, np.nan)
    k_raw = 100.0 * (close - ll) / rng
    k_raw = k_raw.fillna(method="ffill").fillna(50.0)

    k = k_raw.rolling(k_smooth, min_periods=1).mean()
    d = k.rolling(d_len, min_periods=1).mean()
    return k, d

def momentum_dir(delta_series: pd.Series, eps: float = 0.10) -> pd.Series:
    # Classify last-step momentum by epsilon threshold
    def lab(x):
        if pd.isna(x): return "flat"
        if x > eps: return "up"
        if x < -eps: return "down"
        return "flat"
    return delta_series.apply(lab)

def zone_from_d(d: pd.Series, lower: float = 20.0, upper: float = 80.0) -> pd.Series:
    def z(v):
        if pd.isna(v): return "neutral"
        if v <= lower: return "buy zone"
        if v >= upper: return "sell zone"
        return "neutral"
    return d.apply(z)

def trend_from_price_vs_sma(close: pd.Series, ma: pd.Series, tol: float = 0.0) -> pd.Series:
    # compare close to SMA with optional tolerance
    def t(c, m):
        if pd.isna(c) or pd.isna(m): return "uncertain"
        up = m * (1.0 + tol)
        dn = m * (1.0 - tol)
        if c > up: return "bullish"
        if c < dn: return "bearish"
        return "uncertain"
    return pd.Series(np.vectorize(t)(close.values, ma.values), index=close.index)

def forecast_from_trends(st_trend: pd.Series, lt_trend: pd.Series) -> pd.Series:
    # Simple combination: agree -> that direction, else uncertain
    def f(st, lt):
        if st == "bullish" and lt == "bullish": return "bullish"
        if st == "bearish" and lt == "bearish": return "bearish"
        return "uncertain"
    return pd.Series([f(st_trend.iloc[i], lt_trend.iloc[i]) for i in range(min(len(st_trend), len(lt_trend)))])

# ---- Optional: single-symbol compute for uploaded CSVs ----
def compute_signals(df: pd.DataFrame, st_tf: str = "1H", lt_tf: str = "1D",
                    st_sma_len: int = 7, lt_sma_len: int = 100,
                    st_k_len: int = 14, st_k_smooth: int = 3, st_d_len: int = 3,
                    lt_k_len: int = 56, lt_k_smooth: int = 3, lt_d_len: int = 3,
                    lower_thr: float = 20.0, upper_thr: float = 80.0,
                    confirm_htf: bool = True) -> pd.DataFrame:
    # expects df with index datetime and columns: open, high, low, close
    d = df.copy().sort_index()
    # Resample helpers
    def resample(dfin: pd.DataFrame, rule: str) -> pd.DataFrame:
        o = dfin['open'].resample(rule).first()
        h = dfin['high'].resample(rule).max()
        l = dfin['low'].resample(rule).min()
        c = dfin['close'].resample(rule).last()
        out = pd.concat([o,h,l,c], axis=1)
        out.columns = ['open','high','low','close']
        return out.dropna(how="any")

    st_df = resample(d, st_tf)
    lt_df = resample(d, lt_tf)

    # ST
    st_close, st_high, st_low = st_df["close"], st_df["high"], st_df["low"]
    st_sma_ser = sma(st_close, st_sma_len)
    _, st_dv = stochastic_kd(st_high, st_low, st_close, st_k_len, st_k_smooth, st_d_len)
    st_momo_dir = momentum_dir(st_dv - st_dv.shift(1))
    st_zone = zone_from_d(st_dv, lower_thr, upper_thr)
    st_trend = trend_from_price_vs_sma(st_close, st_sma_ser)

    # LT
    lt_close, lt_high, lt_low = lt_df["close"], lt_df["high"], lt_df["low"]
    lt_sma_ser = sma(lt_close, lt_sma_len)
    _, lt_dv = stochastic_kd(lt_high, lt_low, lt_close, lt_k_len, lt_k_smooth, lt_d_len)
    lt_momo_dir = momentum_dir(lt_dv - lt_dv.shift(1))
    lt_zone = zone_from_d(lt_dv, lower_thr, upper_thr)
    lt_trend = trend_from_price_vs_sma(lt_close, lt_sma_ser)

    # align indexes
    ix = st_df.index.intersection(lt_df.index)
    out = pd.DataFrame(index=ix)
    out["st_momo_dir"] = st_momo_dir.reindex(ix)
    out["st_d"] = st_dv.reindex(ix)
    out["st_zone"] = st_zone.reindex(ix)
    out["st_trend"] = st_trend.reindex(ix)
    out["lt_momo_dir"] = lt_momo_dir.reindex(ix)
    out["lt_d"] = lt_dv.reindex(ix)
    out["lt_zone"] = lt_zone.reindex(ix)
    out["lt_trend"] = lt_trend.reindex(ix)

    # forecast
    def f(st, lt):
        if st == "bullish" and lt == "bullish": return "bullish"
        if st == "bearish" and lt == "bearish": return "bearish"
        return "uncertain"
    out["forecast"] = [f(st, lt) for st, lt in zip(out["st_trend"], out["lt_trend"])]
    return out
