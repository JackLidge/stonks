import io
import requests
import pandas as pd

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _wiki_df(url: str, table_index: int, ticker_col: str,
             sector_col: str | None = None, suffix: str = "") -> pd.DataFrame:
    html = requests.get(url, headers=_HEADERS, timeout=15).text
    df = pd.read_html(io.StringIO(html))[table_index]
    out = pd.DataFrame()
    out["ticker"] = df[ticker_col].dropna().apply(lambda t: f"{t}{suffix}")
    if sector_col:
        out["sector"] = df[sector_col]
    return out


def get_sp500_df() -> pd.DataFrame:
    return _wiki_df(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        table_index=0,
        ticker_col="Symbol",
        sector_col="GICS Sector",
    )


def get_ftse100_df() -> pd.DataFrame:
    sector_col = "FTSE industry classification benchmark sector[39]"
    return _wiki_df(
        "https://en.wikipedia.org/wiki/FTSE_100_Index",
        table_index=6,
        ticker_col="Ticker",
        sector_col=sector_col,
        suffix=".L",
    )


def get_ftse250_df() -> pd.DataFrame:
    sector_col = "FTSE Industry Classification Benchmark sector[12]"
    return _wiki_df(
        "https://en.wikipedia.org/wiki/FTSE_250_Index",
        table_index=3,
        ticker_col="Ticker",
        sector_col=sector_col,
        suffix=".L",
    )


def get_nasdaq_df() -> pd.DataFrame:
    resp = requests.get(
        "https://api.nasdaq.com/api/screener/stocks",
        params={"tableonly": "true", "limit": 10000, "download": "true"},
        headers=_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()["data"]["rows"]
    df = pd.DataFrame(rows)[["symbol", "sector", "industry"]]
    return df.rename(columns={"symbol": "ticker"}).dropna(subset=["ticker"])


def get_ftse_aim_df() -> pd.DataFrame:
    resp = requests.get(
        "https://stockanalysis.com/list/london-stock-exchange-aim/",
        headers=_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    df = pd.read_html(io.StringIO(resp.text))[0]
    out = pd.DataFrame()
    out["ticker"] = df["Symbol"].dropna().apply(lambda t: f"{t}.L")
    return out


_INDEX_DF_FNS = {
    "sp500": get_sp500_df,
    "ftse100": get_ftse100_df,
    "ftse250": get_ftse250_df,
    "nasdaq": get_nasdaq_df,
    "ftse_aim": get_ftse_aim_df,
}


def get_universe_df(index: str) -> pd.DataFrame:
    key = index.lower().replace("-", "_")
    if key not in _INDEX_DF_FNS:
        raise ValueError(f"Unknown index '{index}'. Choose from: {', '.join(_INDEX_DF_FNS)}")
    return _INDEX_DF_FNS[key]()


def get_universe(index: str) -> list[str]:
    return get_universe_df(index)["ticker"].tolist()


def list_sectors(index: str) -> list[str]:
    df = get_universe_df(index)
    if "sector" not in df.columns:
        raise ValueError(f"'{index}' has no sector metadata (try: {', '.join(k for k, v in _INDEX_DF_FNS.items() if k != index)})")
    return sorted(df["sector"].dropna().unique().tolist())


def list_industries(index: str) -> list[str]:
    df = get_universe_df(index)
    if "industry" not in df.columns:
        raise ValueError(f"'{index}' has no industry metadata. Only 'nasdaq' provides industry-level data.")
    return sorted(df["industry"].dropna().unique().tolist())
