from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .universe import get_universe_df
from .valuation import dcf
from ._retry import RateLimiter, with_retry

_CHUNK = 500


def _bulk_download_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Fetch 2y of daily close prices for all tickers in batched yf.download() calls.
    Returns {ticker: DataFrame(Close)} — missing/failed tickers are omitted.
    """
    result = {}
    chunks = [tickers[i:i + _CHUNK] for i in range(0, len(tickers), _CHUNK)]
    print(f"Bulk downloading prices ({len(tickers)} tickers in {len(chunks)} batches)...")

    for chunk in tqdm(chunks, desc="price batches"):
        try:
            hist = yf.download(chunk, period="2y", auto_adjust=True,
                               progress=False, threads=True)
            if hist.empty:
                continue

            close = hist["Close"] if isinstance(hist.columns, pd.MultiIndex) else hist[["Close"]]

            for ticker in chunk:
                if ticker in close.columns:
                    series = close[ticker].dropna()
                    if not series.empty:
                        result[ticker] = series.to_frame(name="Close")
        except Exception as e:
            print(f"\nBulk download failed for a batch: {e}")

    return result


def _nearest_close(hist: pd.DataFrame, target: pd.Timestamp) -> float | None:
    if hist.empty:
        return None

    idx = hist.index
    if idx.tz is not None:
        idx_cmp = idx.tz_convert("UTC")
        tgt = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    else:
        idx_cmp = idx
        tgt = target.replace(tzinfo=None)

    candidates = idx_cmp[idx_cmp >= tgt]
    if candidates.empty:
        return None

    val = hist["Close"].iloc[idx_cmp.get_loc(candidates[0])]
    return None if pd.isna(val) else float(val)


def backtest_ticker(ticker: str, discount_rate: float = 0.10,
                    price_hist: pd.DataFrame | None = None) -> dict:
    result = {"ticker": ticker.upper()}
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        cf = stock.cashflow

        currency = info.get("currency", "")
        gbp = currency == "GBp"

        # Need at least 3 periods: [0]=most recent (exclude), [1]=year-ago FCF, [2+]=growth history
        if "Free Cash Flow" not in cf.index or len(cf.columns) < 3:
            result["error"] = "Insufficient FCF history"
            return result

        fcf_series = cf.loc["Free Cash Flow"]

        fcf_then = fcf_series.iloc[1]
        if pd.isna(fcf_then) or fcf_then <= 0:
            result["error"] = "Negative or missing FCF at backtest date"
            return result

        shares = info.get("sharesOutstanding")
        if not shares:
            result["error"] = "No share count available"
            return result

        fcf_per_share_then = fcf_then / shares

        # Growth rate as of a year ago: CAGR using reports [1] onwards (excluding most recent)
        hist_fcf = fcf_series.iloc[1:].dropna().sort_index()
        growth = None
        if len(hist_fcf) >= 2:
            oldest, newest = hist_fcf.iloc[0], hist_fcf.iloc[-1]
            if oldest > 0:
                years = len(hist_fcf) - 1
                growth = (newest / oldest) ** (1 / years) - 1

        if growth is None:
            result["error"] = "Insufficient FCF history for growth rate"
            return result

        growth = min(growth, 0.25)
        iv_then = dcf(fcf_per_share_then, growth, discount_rate)

        # Use pre-fetched price history if available, otherwise fetch individually
        hist = price_hist if price_hist is not None else stock.history(period="2y")
        if hist is None or hist.empty:
            result["error"] = "No price history"
            return result

        one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
        price_then_raw = _nearest_close(hist, one_year_ago)
        price_now_raw = _nearest_close(hist, pd.Timestamp.now() - pd.DateOffset(days=5))

        if price_then_raw is None or price_now_raw is None:
            result["error"] = "Could not find required prices in history"
            return result

        divisor = 100 if gbp else 1
        price_then = price_then_raw / divisor
        price_now = price_now_raw / divisor

        pct_vs_iv = ((price_then - iv_then) / iv_then) * 100
        if pct_vs_iv < -20:
            rating_then = "Undervalued"
        elif pct_vs_iv > 20:
            rating_then = "Overvalued"
        else:
            rating_then = "Fairly valued"

        lump_sum_return = ((price_now - price_then) / price_then) * 100

        monthly_prices = []
        for m in range(1, 13):
            target = pd.Timestamp.now() - pd.DateOffset(months=m)
            p = _nearest_close(hist, target)
            if p is not None:
                monthly_prices.append(p / divisor)

        dca_return = None
        if monthly_prices:
            n = len(monthly_prices)
            total_units = sum(1 / p for p in monthly_prices)
            dca_return = round(((total_units * price_now) - n) / n * 100, 1)

        result.update({
            "price_then": round(price_then, 4),
            "iv_then": round(iv_then, 2),
            "rating_then": rating_then,
            "pct_vs_iv_then": round(pct_vs_iv, 1),
            "price_now": round(price_now, 4),
            "lump_sum_return_pct": round(lump_sum_return, 1),
            "dca_return_pct": dca_return,
            "growth_used": round(growth, 4),
            "currency": "GBP" if gbp else currency,
            "error": None,
        })

    except Exception as e:
        result["error"] = str(e)

    return result


def backtest_index(
    index: str,
    discount_rate: float = 0.10,
    sectors: list[str] | None = None,
    industries: list[str] | None = None,
    workers: int = 5,
    pause: float = 1.0,
) -> pd.DataFrame:
    """
    Backtest DCF valuation across an index over the past year.

    Price history is fetched in bulk via yf.download() to minimise HTTP requests.
    Fundamental requests (info, cashflow) are rate-limited by `pause` seconds between
    calls across all workers.

    Parameters
    ----------
    index         : 'sp500', 'ftse100', 'ftse250', 'nasdaq', 'ftse_aim'
    discount_rate : DCF discount rate (default 0.10)
    sectors       : optional sector pre-filter (case-insensitive)
    industries    : optional industry pre-filter (NASDAQ only, case-insensitive)
    workers       : concurrent requests (default 5)
    pause         : minimum seconds between fundamental requests (default 1.0)
                    Increase to 2.0+ for large unfiltered indices like NASDAQ

    Returns
    -------
    pd.DataFrame sorted by lump_sum_return_pct descending.
    Filter by rating_then == 'Undervalued' to evaluate the DCF strategy.

    Example
    -------
    >>> from stonking.backtest import backtest_index
    >>> df = backtest_index('nasdaq', industries=['Aerospace'], pause=1.0)
    >>> df[df['rating_then'] == 'Undervalued'].sort_values('lump_sum_return_pct', ascending=False)
    """
    universe = get_universe_df(index)

    if sectors and "sector" in universe.columns:
        universe = universe[universe["sector"].str.lower().isin([s.lower() for s in sectors])]
    if industries and "industry" in universe.columns:
        universe = universe[universe["industry"].str.lower().isin([i.lower() for i in industries])]

    tickers = universe["ticker"].tolist()

    price_data = _bulk_download_prices(tickers)
    hits = len(price_data)
    print(f"Price data: {hits}/{len(tickers)} tickers. "
          f"Fetching fundamentals at ≤{1/pause:.1f} req/s...")

    limiter = RateLimiter(pause)

    def fetch(ticker):
        ph = price_data.get(ticker)
        try:
            return with_retry(backtest_ticker, ticker, discount_rate, ph,
                              rate_limiter=limiter)
        except Exception as e:
            return {"ticker": ticker, "error": f"Failed after retries: {e}"}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, t): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(tickers), desc="fundamentals"):
            results.append(future.result())

    df = pd.DataFrame(results)
    ordered = [
        "ticker", "currency",
        "price_then", "iv_then", "rating_then", "pct_vs_iv_then",
        "price_now", "lump_sum_return_pct", "dca_return_pct",
        "growth_used", "error",
    ]
    cols = [c for c in ordered if c in df.columns]
    return df[cols].sort_values("lump_sum_return_pct", ascending=False, na_position="last")
