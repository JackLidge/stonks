from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from .universe import get_universe_df
from .valuation import assess


def scan_index(
    index: str,
    discount_rate: float = 0.10,
    sectors: list[str] | None = None,
    industries: list[str] | None = None,
    workers: int = 10,
) -> pd.DataFrame:
    """
    Run DCF/P-S valuation across an index and return results as a DataFrame.

    Parameters
    ----------
    index        : 'sp500', 'ftse100', 'ftse250', 'nasdaq', 'ftse_aim'
    discount_rate: DCF discount rate (default 0.10)
    sectors      : optional list of sectors to filter on (case-insensitive)
    industries   : optional list of industries to filter on (case-insensitive)
    workers      : concurrent yfinance requests (default 10)

    Returns
    -------
    pd.DataFrame with one row per ticker, sortable by intrinsic_value, verdict etc.

    Example
    -------
    >>> from stonking.scan import scan_index
    >>> df = scan_index('ftse100', sectors=['Technology'])
    >>> df.sort_values('intrinsic_value', ascending=False)
    """
    universe = get_universe_df(index)

    if sectors and "sector" in universe.columns:
        mask = universe["sector"].str.lower().isin([s.lower() for s in sectors])
        universe = universe[mask]

    if industries and "industry" in universe.columns:
        mask = universe["industry"].str.lower().isin([i.lower() for i in industries])
        universe = universe[mask]

    tickers = universe["ticker"].tolist()
    print(f"Scanning {len(tickers)} tickers from {index}...")

    def fetch(ticker):
        try:
            return assess(ticker, discount_rate)
        except Exception as e:
            return {"ticker": ticker, "verdict": f"Error: {e}", "method": None}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, t): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(tickers), desc=index):
            results.append(future.result())

    df = pd.DataFrame(results)

    ordered = [
        "ticker", "method", "price", "currency", "intrinsic_value",
        "rating", "pct_diff",
        "ps_ratio", "ps_benchmark", "ps_benchmark_label",
        "growth", "growth_source", "discount_rate", "verdict",
    ]
    cols = [c for c in ordered if c in df.columns] + \
           [c for c in df.columns if c not in ordered]
    return df[cols]
