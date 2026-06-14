import io
import requests
import pandas as pd

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _wiki_tickers(url: str, table_index: int, column: str, suffix: str = "") -> list[str]:
    html = requests.get(url, headers=_HEADERS, timeout=15).text
    df = pd.read_html(io.StringIO(html))[table_index]
    return [f"{t}{suffix}" for t in df[column].dropna().tolist()]


def get_sp500() -> list[str]:
    return _wiki_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        table_index=0,
        column="Symbol",
    )


def get_ftse100() -> list[str]:
    return _wiki_tickers(
        "https://en.wikipedia.org/wiki/FTSE_100_Index",
        table_index=6,
        column="Ticker",
        suffix=".L",
    )


def get_ftse250() -> list[str]:
    return _wiki_tickers(
        "https://en.wikipedia.org/wiki/FTSE_250_Index",
        table_index=3,
        column="Ticker",
        suffix=".L",
    )


def get_nasdaq() -> list[str]:
    resp = requests.get(
        "https://api.nasdaq.com/api/screener/stocks",
        params={"tableonly": "true", "limit": 10000, "download": "true"},
        headers=_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()["data"]["rows"]
    return [r["symbol"] for r in rows if r.get("symbol")]


def get_ftse_aim() -> list[str]:
    resp = requests.get(
        "https://stockanalysis.com/list/london-stock-exchange-aim/",
        headers=_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    df = pd.read_html(io.StringIO(resp.text))[0]
    return [f"{s}.L" for s in df["Symbol"].dropna().tolist()]


_INDICES = {
    "sp500": get_sp500,
    "ftse100": get_ftse100,
    "ftse250": get_ftse250,
    "nasdaq": get_nasdaq,
    "ftse_aim": get_ftse_aim,
}


def get_universe(index: str) -> list[str]:
    key = index.lower().replace("-", "_")
    if key not in _INDICES:
        raise ValueError(f"Unknown index '{index}'. Choose from: {', '.join(_INDICES)}")
    return _INDICES[key]()
