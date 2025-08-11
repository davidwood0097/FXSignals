
# streamlit_app_v8_0.py
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import streamlit as st
import logging
import warnings

# Silence noisy loggers that print 404s for missing Yahoo symbols
for name in ("yfinance","yfinance.base","yfinance.data","yfinance.tz","urllib3","urllib3.connectionpool"):
    logging.getLogger(name).setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except Exception:
    yf = None

from mtf_fx_signals import (
    sma, stochastic_kd, momentum_dir, zone_from_d,
    trend_from_price_vs_sma, forecast_from_trends
)

st.set_page_config(page_title="Signals Dashboard (FX / Indices / Metals / Crypto)", page_icon="📊", layout="wide")

# ===== Sidebar =====
st.sidebar.header("Data & Refresh")
data_mode = st.sidebar.selectbox("Data source", ["Yahoo Finance (live)", "Upload CSV (single symbol)"])

default_fx = "AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD, CADCHF, CADJPY, CHFJPY, EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURMXN, EURNOK, EURNZD, EURSEK, EURUSD, EURZAR, GBPAUD, GBPCAD, GBPCHF, GBPNOK, GBPNZD, GBPJPY, GBPSEK, GBPUSD, NZDCAD, NZDCHF, NZDJPY, NZDUSD, USDCAD, USDCHF, USDCNH, USDJPY, USDMXN, USDNOK, USDSEK, USDZAR"
preset_indices = "^GSPC, ^DJI, ^IXIC, ^GDAXI, ^FTSE, ^N225, ^AXJO"
preset_metals  = "XAUUSD, XAGUSD, XPTUSD, XPDUSD"
preset_crypto  = "BTC-USD, ETH-USD, SOL-USD, XRP-USD"

if data_mode == "Yahoo Finance (live)":
    pairs_fx = st.sidebar.text_area("FX pairs (auto '=X')", value=default_fx, height=140)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Add more markets (optional)")
    use_indices = st.sidebar.checkbox("Include indices", value=True)
    use_metals  = st.sidebar.checkbox("Include metals", value=True)
    use_crypto  = st.sidebar.checkbox("Include crypto", value=True)
    idx_input   = st.sidebar.text_input("Indices (Yahoo tickers)", value=preset_indices if use_indices else "")
    met_input   = st.sidebar.text_input("Metals (FX-style or Yahoo tickers)", value=preset_metals if use_metals else "")
    cry_input   = st.sidebar.text_input("Crypto (Yahoo tickers)", value=preset_crypto if use_crypto else "")

    st_days     = st.sidebar.slider("1H lookback (days)", 5, 180, 30)
    lt_days     = st.sidebar.slider("1D lookback (days)", 120, 1500, 400)
    req_delay   = st.sidebar.slider("Request pacing (sec)", 0.0, 1.0, 0.30, 0.05, help="Small delay helps avoid Yahoo throttling")
    refresh_sec = st.sidebar.slider("Auto-refresh (sec)", 0, 600, 60)
    only_closed = st.sidebar.checkbox("Use only closed bars", value=True)
else:
    uploaded    = st.sidebar.file_uploader("Upload OHLC CSV", type=["csv"])
    refresh_sec = 0
    only_closed = True
    req_delay   = 0.0

st.sidebar.header("Signal Parameters")
st_sma_len  = st.sidebar.number_input("ST SMA Len (1H)", 1, 500, 7)
st_k_len    = st.sidebar.number_input("ST %K Len (1H)", 1, 500, 14)
st_k_smooth = st.sidebar.number_input("ST %K Smooth", 1, 50, 3)
st_d_len    = st.sidebar.number_input("ST %D Len (1H)", 1, 50, 3)

lt_sma_len  = st.sidebar.number_input("LT SMA Len (1D)", 1, 2000, 100)
lt_k_len    = st.sidebar.number_input("LT %K Len (1D)", 1, 1000, 56)
lt_k_smooth = st.sidebar.number_input("LT %K Smooth", 1, 50, 3)
lt_d_len    = st.sidebar.number_input("LT %D Len (1D)", 1, 50, 3)

lower_thr   = st.sidebar.slider("Buy Zone <=", 0.0, 100.0, 20.0, 0.5)
upper_thr   = st.sidebar.slider("Sell Zone >=", 0.0, 100.0, 80.0, 0.5)
flat_eps    = st.sidebar.number_input("Momentum flat tolerance (delta)", 0.0, 5.0, 0.10, 0.01)
show_numbers= st.sidebar.checkbox("Show numeric momentum (append)", value=False)

st.title("📊 ST (1H) + LT (1D) Signals — Multi-Market")
st.caption("Adds LT fallbacks, tall tables (no inner scroll), and friendly index names. Metals still use futures-first.")

# ===== Utilities =====
def split_list(s: str):
    return [x.strip() for x in (s or "").replace("\n", ",").split(",") if x.strip()]

def is_fx_pair(token: str) -> bool:
    t = token.replace("/", "").replace(" ", "").upper()
    return len(t) == 6 and t.isalpha()

# Metals fallbacks: FUTURES FIRST, then spot aliases
METALS_FALLBACKS = {
    "XAUUSD": ["GC=F", "XAUUSD=X", "XAU=X"],
    "XAGUSD": ["SI=F", "XAGUSD=X", "XAG=X"],
    "XPTUSD": ["PL=F", "XPTUSD=X"],
    "XPDUSD": ["PA=F", "XPDUSD=X"],
}

def normalize_with_fallbacks(token: str):
    t = token.strip()
    if not t:
        return []
    upper = t.replace("/", "").replace(" ", "").upper()
    if upper in ("GC=F","SI=F","PL=F","PA=F","XAUUSD=X","XAGUSD=X","XPTUSD=X","XPDUSD=X","XAU=X","XAG=X"):
        return [upper]
    if upper in METALS_FALLBACKS:
        return METALS_FALLBACKS[upper][:]
    if is_fx_pair(upper):
        return [upper + "=X"]
    return [t]

def pretty_index_name(instr: str) -> str:
    u = instr.upper()
    alias = {
        "^GSPC": "S&P500", "GSPC": "S&P500",
        "^DJI": "US30 Dow Jones", "DJI": "US30 Dow Jones",
        "^IXIC": "NASDAQ", "IXIC": "NASDAQ",
        "^GDAXI": "DAX GER30", "GDAXI": "DAX GER30",
        "^FTSE": "FTSE UK100", "FTSE": "FTSE UK100",
        "^N225": "NIKKEI", "N225": "NIKKEI",
        "^AXJO": "ASX200", "AXJO": "ASX200",
    }
    return alias.get(u, instr)

def utc_now():
    return datetime.now(timezone.utc)

@st.cache_data(show_spinner=False, ttl=120)
def dl_download(ticker, period, interval):
    try:
        return yf.download(ticker, period=period, interval=interval,
                           auto_adjust=False, group_by="column",
                           progress=False, prepost=False, threads=False)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=120)
def dl_download_range(ticker, start, end, interval):
    try:
        return yf.download(ticker, start=start, end=end, interval=interval,
                           auto_adjust=False, group_by="column",
                           progress=False, prepost=False, threads=False)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=120)
def dl_history(ticker, period, interval):
    try:
        return yf.Ticker(ticker).history(period=period, interval=interval,
                                         auto_adjust=False, prepost=False)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=120)
def dl_history_range(ticker, start, end, interval):
    try:
        return yf.Ticker(ticker).history(start=start, end=end, interval=interval,
                                         auto_adjust=False, prepost=False)
    except Exception:
        return None

def tidy(df):
    if df is None or len(getattr(df, "index", [])) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    cols = ["Open","High","Low","Close"]
    if all(c in df.columns for c in cols):
        out = df[cols].copy()
        out.columns = ["open","high","low","close"]
    else:
        cols2 = ["open","high","low","close"]
        if not all(c in df.columns for c in cols2):
            return pd.DataFrame()
        out = df[cols2].copy()
    out.index.name = "time"
    out = out[~out.index.duplicated(keep="last")].dropna(how="any")
    return out

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    out = pd.concat([o,h,l,c], axis=1)
    out.columns = ['open','high','low','close']
    out = out.dropna(how='any')
    out.index.name = 'time'
    return out

def fetch_one(tk, interval, period_days):
    end = utc_now()
    start = end - timedelta(days=period_days)
    pstr = f"{period_days}d"
    # download(period)
    df = tidy(dl_download(tk, pstr, interval))
    if not df.empty: return df, f"download {pstr} {interval}"
    # history(period)
    df = tidy(dl_history(tk, pstr, interval))
    if not df.empty: return df, f"history {pstr} {interval}"
    # download(start/end)
    df = tidy(dl_download_range(tk, start, end, interval))
    if not df.empty: return df, f"download {start.date()}→{end.date()} {interval}"
    # history(start/end)
    df = tidy(dl_history_range(tk, start, end, interval))
    if not df.empty: return df, f"history {start.date()}→{end.date()} {interval}"
    return pd.DataFrame(), "none"

def fetch_intraday_candidates(cands, period_days, intervals):
    for tk in cands:
        for itv in intervals:
            df, how = fetch_one(tk, itv, period_days)
            if not df.empty:
                return df, tk, how, itv
    return pd.DataFrame(), None, "none", None

def fetch_daily_candidates(cands, period_days):
    for tk in cands:
        df, how = fetch_one(tk, "1d", period_days)
        if not df.empty:
            return df, tk, how
    return pd.DataFrame(), None, "none"

def drop_last_if_open(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.iloc[:-1] if len(df) >= 2 else df

def colorize(value):
    if isinstance(value, (int, float)) and not pd.isna(value):
        if value > 0:   return f"🟢 {value:+.2f}%"
        if value < 0:   return f"🔴 {value:+.2f}%"
        return f"🔵 {value:+.2f}%"
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, str):
        v = value.strip().lower()
        return {"bullish":"🟢 Bullish","bearish":"🔴 Bearish","uncertain":"🟡 Uncertain",
                "up":"🟢 UP","down":"🔴 DOWN","flat":"🔵 FLAT",
                "buy zone":"🟢 Buy Zone","sell zone":"🔴 Sell Zone","neutral":"🔵 Neutral"}.get(v, value)
    return str(value)

def last_common_valid(*series_list: pd.Series):
    if not series_list:
        return None
    valid = series_list[0].dropna().index
    for s in series_list[1:]:
        valid = valid.intersection(s.dropna().index)
    if len(valid) == 0:
        return None
    return valid[-1]

def compute_lt_metrics(lt_df: pd.DataFrame, sma_len: int, k_len: int, k_smooth: int, d_len: int,
                       lower: float, upper: float, eps: float):
    lt_close, lt_high, lt_low = lt_df["close"], lt_df["high"], lt_df["low"]
    lt_sma_ser = sma(lt_close, sma_len)
    _, lt_d = stochastic_kd(lt_high, lt_low, lt_close, k_len, k_smooth, d_len)
    lt_momo_dir = momentum_dir(lt_d - lt_d.shift(1), eps)
    lt_zone = zone_from_d(lt_d, lower, upper)
    lt_trend = trend_from_price_vs_sma(lt_close, lt_sma_ser)
    lt_t = last_common_valid(lt_sma_ser, lt_d, lt_zone, lt_trend)
    if lt_t is None:
        return None
    lt_last_d = float(lt_d.loc[lt_t]) if not pd.isna(lt_d.loc[lt_t]) else np.nan
    return {
        "lt_momo_dir": lt_momo_dir.loc[lt_t],
        "lt_d": lt_last_d,
        "lt_zone": lt_zone.loc[lt_t],
        "lt_trend": lt_trend.loc[lt_t],
    }

# ===== Connectivity quick test =====
st.markdown("### Connectivity quick test")
if st.button("Run test (EURUSD, ^GSPC, BTC-USD, GC=F)"):
    test_syms = {"EURUSD=X":"60m", "^GSPC":"60m", "BTC-USD":"60m", "GC=F":"60m"}
    lines = []
    for tk, itv in test_syms.items():
        df, how = fetch_one(tk, itv, 10)
        lines.append(f"{tk}: rows={0 if df is None else len(df)} via {how}")
        time.sleep(0.15)
    st.info("\n".join(lines))

# ===== Compute flows =====
def compute_one(symbol_label: str, candidates: list[str]):
    # Intraday: try 1h/60m then 30m/15m and resample
    st_df, st_used_tk, st_how, st_itv = fetch_intraday_candidates(candidates, st_days, ["1h","60m","30m","15m"])
    if st_df.empty:
        raise ValueError("No data returned (1H)")
    if st_itv in ("30m","15m"):
        st_df = resample_ohlc(st_df, "60T")

    # Daily
    lt_df, lt_used_tk, lt_how = fetch_daily_candidates(candidates, lt_days)
    if lt_df.empty:
        raise ValueError("No data returned (1D)")

    if only_closed:
        st_df = drop_last_if_open(st_df)
        lt_df = drop_last_if_open(lt_df)

    # ST calcs
    st_close, st_high, st_low = st_df["close"], st_df["high"], st_df["low"]
    st_sma_ser = sma(st_close, st_sma_len)
    _, st_d = stochastic_kd(st_high, st_low, st_close, st_k_len, st_k_smooth, st_d_len)
    st_momo_dir = momentum_dir(st_d - st_d.shift(1), flat_eps)
    st_zone = zone_from_d(st_d, lower_thr, upper_thr)
    st_trend = trend_from_price_vs_sma(st_close, st_sma_ser)
    st_t = last_common_valid(st_sma_ser, st_d, st_zone, st_trend)
    if st_t is None:
        raise ValueError("Insufficient 1H history for ST calculations")
    st_last_d = float(st_d.loc[st_t]) if not pd.isna(st_d.loc[st_t]) else np.nan
    st_last = {
        "st_momo_dir": st_momo_dir.loc[st_t],
        "st_d": st_last_d,
        "st_zone": st_zone.loc[st_t],
        "st_trend": st_trend.loc[st_t],
    }
    last_price = float(st_close.iloc[-1])

    # LT calcs with fallback for short history (e.g., USDCNH)
    lt_metrics = compute_lt_metrics(lt_df, lt_sma_len, lt_k_len, lt_k_smooth, lt_d_len,
                                    lower_thr, upper_thr, flat_eps)
    lt_used_note = "LT normal"
    if lt_metrics is None:
        n = len(lt_df)
        # fallback lengths bounded to available history
        fb_sma = min(lt_sma_len, max(20, n // 4))
        fb_k   = min(lt_k_len, max(10, n // 4))
        fb_d   = min(lt_d_len, 3)
        fb_ks  = min(lt_k_smooth, 3)
        lt_metrics = compute_lt_metrics(lt_df, fb_sma, fb_k, fb_ks, fb_d,
                                        lower_thr, upper_thr, flat_eps)
        lt_used_note = f"LT fallback (sma={fb_sma}, k={fb_k})" if lt_metrics is not None else "LT unavailable"
    if lt_metrics is None:
        # Graceful degrade
        lt_metrics = {
            "lt_momo_dir": "uncertain",
            "lt_d": np.nan,
            "lt_zone": "neutral",
            "lt_trend": "uncertain"
        }

    forecast = forecast_from_trends(
        pd.Series([st_last["st_trend"]], index=[0]),
        pd.Series([lt_metrics["lt_trend"]], index=[0])
    ).iloc[0]

    # Daily % change from LT close series
    lt_close = lt_df["close"]
    dchg = np.nan
    if len(lt_close) >= 2 and not pd.isna(lt_close.iloc[-1]) and not pd.isna(lt_close.iloc[-2]):
        dchg = float((lt_close.iloc[-1] / lt_close.iloc[-2] - 1.0) * 100.0)

    display_label = pretty_index_name(symbol_label)

    return {
        "Instrument": display_label,
        "Last Price": round(last_price, 5),
        "Daily Change": dchg,
        "Forecast": forecast,
        "ST Momentum": f"{st_last['st_momo_dir']} ({st_last['st_d']:.1f})" if show_numbers else f"{st_last['st_momo_dir']}",
        "LT Momentum": f"{lt_metrics['lt_momo_dir']} ({lt_metrics['lt_d']:.1f})" if show_numbers else f"{lt_metrics['lt_momo_dir']}",
        "ST Alert": st_last["st_zone"],
        "LT Alert": lt_metrics["lt_zone"],
        "diag": f"ST {st_used_tk} via {st_how} | LT {lt_used_tk} via {lt_how} | {lt_used_note}",
    }

def build_tokens():
    fx_list  = split_list(pairs_fx) if data_mode == "Yahoo Finance (live)" else []
    idx_list = split_list(idx_input) if data_mode == "Yahoo Finance (live)" and use_indices else []
    met_list = split_list(met_input) if data_mode == "Yahoo Finance (live)" and use_metals else []
    cry_list = split_list(cry_input) if data_mode == "Yahoo Finance (live)" and use_crypto else []
    return fx_list + idx_list + met_list + cry_list

def calc_table_height(df: pd.DataFrame, base: int = 44, pad: int = 48, max_h: int = 3000) -> int:
    if df is None or df.empty:
        return 350
    # Height = row_height * (rows + header) + padding, capped
    est = base * (len(df) + 1) + pad
    return min(max(est, 350), max_h)

# ===== Main run =====
rows_main, rows_diag = [], []
if data_mode == "Yahoo Finance (live)":
    tokens = build_tokens()
    pairs = [(tok, normalize_with_fallbacks(tok)) for tok in tokens if tok.strip()]
    progress = st.progress(0) if len(pairs) > 1 else None
    for i, (label, cands) in enumerate(pairs, start=1):
        try:
            res = compute_one(label.upper(), cands)
            rows_main.append({
                "Instrument": res["Instrument"],
                "Last Price": res["Last Price"],
                "Daily Change": colorize(res["Daily Change"]),
                "Forecast": colorize(res["Forecast"]),
                "ST Momentum": colorize(res["ST Momentum"]),
                "LT Momentum": colorize(res["LT Momentum"]),
                "ST Alert": colorize(res["ST Alert"]),
                "LT Alert": colorize(res["LT Alert"]),
            })
            rows_diag.append({"Instrument": res["Instrument"], "Data used": res["diag"]})
        except Exception as e:
            st.warning(f"{label}: {e}")
        finally:
            if progress is not None:
                progress.progress(i/len(pairs))
            if req_delay and req_delay > 0:
                time.sleep(req_delay)
    if progress is not None:
        progress.empty()
else:
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        dt_col = None
        for c in df_up.columns:
            if c.lower() in ("time","datetime","date"):
                dt_col = c; break
        if dt_col is None:
            st.error("CSV must include a time/datetime column.")
        else:
            df_up[dt_col] = pd.to_datetime(df_up[dt_col], utc=True, errors="coerce")
            df_up = df_up.set_index(dt_col)
            df_up.columns = [c.lower() for c in df_up.columns]
            from mtf_fx_signals import compute_signals
            out = compute_signals(df_up, st_tf="1H", lt_tf="1D",
                                  st_sma_len=st_sma_len, lt_sma_len=lt_sma_len,
                                  st_k_len=st_k_len, st_k_smooth=st_k_smooth, st_d_len=st_d_len,
                                  lt_k_len=lt_k_len, lt_k_smooth=lt_k_smooth, lt_d_len=lt_d_len,
                                  lower_thr=lower_thr, upper_thr=upper_thr, confirm_htf=True)
            last = out.iloc[-1]
            c = colorize
            rows_main.append({
                "Instrument": "UPLOADED_SYMBOL",
                "Last Price": float(df_up["close"].iloc[-1]),
                "Daily Change": c(np.nan),
                "Forecast": c(last["forecast"]),
                "ST Momentum": c(f"{last['st_momo_dir']} ({last['st_d']:.1f})" if show_numbers else last['st_momo_dir']),
                "LT Momentum": c(f"{last['lt_momo_dir']} ({last['lt_d']:.1f})" if show_numbers else last['lt_momo_dir']),
                "ST Alert": c(last["st_zone"]),
                "LT Alert": c(last["lt_zone"]),
            })

df = pd.DataFrame.from_records(rows_main) if rows_main else pd.DataFrame()
df_diag = pd.DataFrame.from_records(rows_diag) if rows_diag else pd.DataFrame()

# Exact column order
order = ["Instrument","Last Price","Daily Change","Forecast","ST Momentum","LT Momentum","ST Alert","LT Alert"]
if not df.empty:
    df = df[[c for c in order if c in df.columns]]

st.subheader("Signals")
height = calc_table_height(df)
st.dataframe(df, use_container_width=True, height=height)

if not df_diag.empty:
    with st.expander("Diagnostics (what source worked per symbol)"):
        st.dataframe(df_diag, use_container_width=True)

if data_mode == "Yahoo Finance (live)" and refresh_sec and refresh_sec > 0:
    st.caption(f"Auto-refreshing in {int(refresh_sec)} seconds...")
    time.sleep(int(refresh_sec))
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()
