
# streamlit_app_v8_2.py
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import streamlit as st
import logging
import warnings

# Silence noisy loggers
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

# ===== Standard defaults =====
DEFAULTS = {
    "st_days": 30,
    "lt_days": 400,
    "req_delay": 0.30,
    "only_closed": True,
    "st_sma_len": 7,
    "st_k_len": 14,
    "st_k_smooth": 3,
    "st_d_len": 3,
    "lt_sma_len": 100,
    "lt_k_len": 56,
    "lt_k_smooth": 3,
    "lt_d_len": 3,
    "lower_thr": 20.0,
    "upper_thr": 80.0,
    "flat_eps": 0.10,
    "show_numbers": False,
    "refresh_min": 10
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===== Sidebar =====
st.sidebar.header("Data & Refresh")

if st.sidebar.button("Restore Standard Defaults", type="secondary"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.success("Defaults restored.")
    st.rerun()

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

    st_days     = st.sidebar.slider("1H lookback (days)", 5, 180, st.session_state["st_days"], key="st_days")
    lt_days     = st.sidebar.slider("1D lookback (days)", 120, 1500, st.session_state["lt_days"], key="lt_days")
    req_delay   = st.sidebar.slider("Request pacing (sec)", 0.0, 1.0, st.session_state["req_delay"], 0.05, help="Small delay helps avoid Yahoo throttling", key="req_delay")
    refresh_min = st.sidebar.slider("Auto-refresh (minutes)", 0, 60, st.session_state["refresh_min"], key="refresh_min")
    only_closed = st.sidebar.checkbox("Use only closed bars", value=st.session_state["only_closed"], key="only_closed")
else:
    uploaded    = st.sidebar.file_uploader("Upload OHLC CSV", type=["csv"])
    refresh_min = st.session_state["refresh_min"]
    only_closed = True
    req_delay   = st.session_state["req_delay"]
    st_days     = st.session_state["st_days"]
    lt_days     = st.session_state["lt_days"]

st.sidebar.header("Signal Parameters")
st_sma_len  = st.sidebar.number_input("ST SMA Len (1H)", 1, 500, st.session_state["st_sma_len"], key="st_sma_len")
st_k_len    = st.sidebar.number_input("ST %K Len (1H)", 1, 500, st.session_state["st_k_len"], key="st_k_len")
st_k_smooth = st.sidebar.number_input("ST %K Smooth", 1, 50, st.session_state["st_k_smooth"], key="st_k_smooth")
st_d_len    = st.sidebar.number_input("ST %D Len (1H)", 1, 50, st.session_state["st_d_len"], key="st_d_len")

lt_sma_len  = st.sidebar.number_input("LT SMA Len (1D)", 1, 2000, st.session_state["lt_sma_len"], key="lt_sma_len")
lt_k_len    = st.sidebar.number_input("LT %K Len (1D)", 1, 1000, st.session_state["lt_k_len"], key="lt_k_len")
lt_k_smooth = st.sidebar.number_input("LT %K Smooth", 1, 50, st.session_state["lt_k_smooth"], key="lt_k_smooth")
lt_d_len    = st.sidebar.number_input("LT %D Len (1D)", 1, 50, st.session_state["lt_d_len"], key="lt_d_len")

lower_thr   = st.sidebar.slider("Buy Zone <=", 0.0, 100.0, float(st.session_state["lower_thr"]), 0.5, key="lower_thr")
upper_thr   = st.sidebar.slider("Sell Zone >=", 0.0, 100.0, float(st.session_state["upper_thr"]), 0.5, key="upper_thr")
flat_eps    = st.sidebar.number_input("Momentum flat tolerance (delta)", 0.0, 5.0, float(st.session_state["flat_eps"]), 0.01, key="flat_eps")
show_numbers= st.sidebar.checkbox("Show numeric momentum (append)", value=bool(st.session_state["show_numbers"]), key="show_numbers")

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

def build_tokens(pairs_fx, idx_input, met_input, cry_input, use_indices, use_metals, use_crypto, data_mode):
    fx_list  = split_list(pairs_fx) if data_mode == "Yahoo Finance (live)" else []
    idx_list = split_list(idx_input) if data_mode == "Yahoo Finance (live)" and use_indices else []
    met_list = split_list(met_input) if data_mode == "Yahoo Finance (live)" and use_metals else []
    cry_list = split_list(cry_input) if data_mode == "Yahoo Finance (live)" and use_crypto else []
    return fx_list + idx_list + met_list + cry_list

def calc_table_height(df: pd.DataFrame, base: int = 44, pad: int = 48, max_h: int = 3000) -> int:
    if df is None or df.empty:
        return 350
    est = base * (len(df) + 1) + pad
    return min(max(est, 350), max_h)

st.title("📊 ST (1H) + LT (1D) Signals — Multi-Market")
st.caption("Now with colored ST/LT Alert cells when aligned to Forecast, plus minute-based auto-refresh.")

# ===== Main run =====
rows_main, rows_diag = [], []
if data_mode == "Yahoo Finance (live)":
    tokens = build_tokens(pairs_fx, idx_input, met_input, cry_input, use_indices, use_metals, use_crypto, data_mode)
    pairs = [(tok, normalize_with_fallbacks(tok)) for tok in tokens if tok.strip()]
    progress = st.progress(0) if len(pairs) > 1 else None
    for i, (label, cands) in enumerate(pairs, start=1):
        try:
            # Intraday
            st_df, st_used_tk, st_how, st_itv = fetch_intraday_candidates(cands, st_days, ["1h","60m","30m","15m"])
            if st_df.empty:
                raise ValueError("No data returned (1H)")
            if st_itv in ("30m","15m"):
                st_df = resample_ohlc(st_df, "60T")
            # Daily
            lt_df, lt_used_tk, lt_how = fetch_daily_candidates(cands, lt_days)
            if lt_df.empty:
                raise ValueError("No data returned (1D)")
            if only_closed:
                st_df = drop_last_if_open(st_df)
                lt_df = drop_last_if_open(lt_df)

            # ST
            st_close, st_high, st_low = st_df["close"], st_df["high"], st_df["low"]
            st_sma_ser = sma(st_close, st_sma_len)
            _, st_dv = stochastic_kd(st_high, st_low, st_close, st_k_len, st_k_smooth, st_d_len)
            st_momo_dir = momentum_dir(st_dv - st_dv.shift(1), flat_eps)
            st_zone = zone_from_d(st_dv, lower_thr, upper_thr)
            st_trend = trend_from_price_vs_sma(st_close, st_sma_ser)
            # align
            def last_valid(*series_list: pd.Series):
                valid = series_list[0].dropna().index
                for s in series_list[1:]:
                    valid = valid.intersection(s.dropna().index)
                return valid[-1] if len(valid) else None
            st_t = last_valid(st_sma_ser, st_dv, st_zone, st_trend)
            if st_t is None:
                raise ValueError("Insufficient 1H history for ST calculations")
            st_last_d = float(st_dv.loc[st_t]) if not pd.isna(st_dv.loc[st_t]) else np.nan
            st_last = {
                "st_momo_dir": st_momo_dir.loc[st_t],
                "st_d": st_last_d,
                "st_zone": st_zone.loc[st_t],
                "st_trend": st_trend.loc[st_t],
            }
            last_price = float(st_close.iloc[-1])

            # LT with fallback
            lt_metrics = compute_lt_metrics(lt_df, lt_sma_len, lt_k_len, lt_k_smooth, lt_d_len,
                                            lower_thr, upper_thr, flat_eps)
            lt_used_note = "LT normal"
            if lt_metrics is None:
                n = len(lt_df)
                fb_sma = min(lt_sma_len, max(20, n // 4))
                fb_k   = min(lt_k_len, max(10, n // 4))
                fb_d   = min(lt_d_len, 3)
                fb_ks  = min(lt_k_smooth, 3)
                lt_metrics = compute_lt_metrics(lt_df, fb_sma, fb_k, fb_ks, fb_d,
                                                lower_thr, upper_thr, flat_eps)
                lt_used_note = f"LT fallback (sma={fb_sma}, k={fb_k})" if lt_metrics is not None else "LT unavailable"
            if lt_metrics is None:
                lt_metrics = {"lt_momo_dir":"uncertain","lt_d":np.nan,"lt_zone":"neutral","lt_trend":"uncertain"}

            # Forecast & daily change
            forecast = forecast_from_trends(
                pd.Series([st_last["st_trend"]], index=[0]),
                pd.Series([lt_metrics["lt_trend"]], index=[0])
            ).iloc[0]

            lt_close = lt_df["close"]
            dchg = np.nan
            if len(lt_close) >= 2 and not pd.isna(lt_close.iloc[-1]) and not pd.isna(lt_close.iloc[-2]):
                dchg = float((lt_close.iloc[-1] / lt_close.iloc[-2] - 1.0) * 100.0)

            display_label = pretty_index_name(label.upper())

            # Build row (use emojis for display)
            def emoj_fc(f):
                return "🟢 Bullish" if f=="bullish" else ("🔴 Bearish" if f=="bearish" else "🟡 Uncertain")
            def emoj_dir(d):
                return "🟢 UP" if d=="up" else ("🔴 DOWN" if d=="down" else "🔵 FLAT")
            def emoj_zone(z):
                return "🟢 Buy Zone" if z=="buy zone" else ("🔴 Sell Zone" if z=="sell zone" else "🔵 Neutral")

            rows_main.append({
                "Instrument": display_label,
                "Last Price": round(last_price, 5),
                "Daily Change": dchg,
                "Forecast": emoj_fc(forecast),
                "ST Momentum": emoj_dir(st_last["st_momo_dir"]),
                "LT Momentum": emoj_dir(lt_metrics["lt_momo_dir"]),
                "ST Alert": emoj_zone(st_last["st_zone"]),
                "LT Alert": emoj_zone(lt_metrics["lt_zone"]),
            })
            rows_diag.append({"Instrument": display_label, "Data used": f"ST {st_used_tk} via {st_how} | LT {lt_used_tk} via {lt_how} | {lt_used_note}"})
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
            def emojify_dir(d):
                return "🟢 UP" if d=="up" else ("🔴 DOWN" if d=="down" else "🔵 FLAT")
            def emojify_zone(z):
                return "🟢 Buy Zone" if z=="buy zone" else ("🔴 Sell Zone" if z=="sell zone" else "🔵 Neutral")
            def emojify_fc(f):
                return "🟢 Bullish" if f=="bullish" else ("🔴 Bearish" if f=="bearish" else "🟡 Uncertain")
            rows_main.append({
                "Instrument": "UPLOADED_SYMBOL",
                "Last Price": float(df_up["close"].iloc[-1]),
                "Daily Change": np.nan,
                "Forecast": emojify_fc(last["forecast"]),
                "ST Momentum": emojify_dir(last["st_momo_dir"]),
                "LT Momentum": emojify_dir(last["lt_momo_dir"]),
                "ST Alert": emojify_zone(last["st_zone"]),
                "LT Alert": emojify_zone(last["lt_zone"]),
            })

# Build DataFrame
df = pd.DataFrame.from_records(rows_main) if rows_main else pd.DataFrame()
order = ["Instrument","Last Price","Daily Change","Forecast","ST Momentum","LT Momentum","ST Alert","LT Alert"]
if not df.empty:
    df = df[[c for c in order if c in df.columns]]
    # Color daily change text
    df["Daily Change"] = df["Daily Change"].apply(lambda v: "🟢 %+0.2f%%" % v if isinstance(v,(int,float)) and v>0
                                                  else ("🔴 %+0.2f%%" % v if isinstance(v,(int,float)) and v<0
                                                        else ("🔵 %+0.2f%%" % v if isinstance(v,(int,float)) else "—")))

# Styling: shade ST/LT Alert if aligned with Forecast
def highlight_alignment(dataframe: pd.DataFrame):
    styles = pd.DataFrame('', index=dataframe.index, columns=dataframe.columns)
    for idx, row in dataframe.iterrows():
        f = str(row.get("Forecast", ""))
        stA = str(row.get("ST Alert", ""))
        ltA = str(row.get("LT Alert", ""))
        if "Bullish" in f:
            if "Buy" in stA: styles.loc[idx, "ST Alert"] = 'background-color: rgba(0, 200, 0, 0.18)'
            if "Buy" in ltA: styles.loc[idx, "LT Alert"] = 'background-color: rgba(0, 200, 0, 0.18)'
        elif "Bearish" in f:
            if "Sell" in stA: styles.loc[idx, "ST Alert"] = 'background-color: rgba(255, 0, 0, 0.18)'
            if "Sell" in ltA: styles.loc[idx, "LT Alert"] = 'background-color: rgba(255, 0, 0, 0.18)'
    return styles

st.subheader("Signals")
if df.empty:
    st.info("No rows to display yet.")
else:
    styler = df.style.apply(highlight_alignment, axis=None)
    st.dataframe(styler, use_container_width=True, height=calc_table_height(df))

# Diagnostics
if rows_diag:
    with st.expander("Diagnostics (what source worked per symbol)"):
        st.dataframe(pd.DataFrame(rows_diag), use_container_width=True)

# Auto-refresh (minutes)
if data_mode == "Yahoo Finance (live)" and refresh_min and refresh_min > 0:
    st.caption(f"Auto-refreshing in {int(refresh_min)} minute(s)...")
    time.sleep(int(refresh_min*60))
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()
