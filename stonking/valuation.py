import yfinance as yf

# Median P/S benchmarks by industry, with sector-level fallbacks.
# Sources: Damodaran annual sector data (Jan 2025).
_PS_BY_INDUSTRY = {
    "Aerospace & Defense": 2.0,
    "Consumer Electronics": 3.5,
    "Software - Infrastructure": 9.0,
    "Software - Application": 7.0,
    "Semiconductors": 7.0,
    "Restaurants": 1.5,
    "Grocery Stores": 0.4,
    "Drug Manufacturers - General": 4.5,
    "Biotechnology": 6.0,
    "Oil & Gas E&P": 2.5,
    "Oil & Gas Integrated": 1.0,
    "Banks - Diversified": 3.0,
    "Insurance - Diversified": 1.2,
    "Telecom Services": 1.5,
    "Utilities - Regulated Electric": 1.8,
    "Real Estate - Diversified": 5.0,
}

_PS_BY_SECTOR = {
    "Technology": 7.0,
    "Healthcare": 3.5,
    "Consumer Cyclical": 1.5,
    "Consumer Defensive": 1.0,
    "Industrials": 2.0,
    "Energy": 1.5,
    "Financial Services": 2.5,
    "Communication Services": 2.5,
    "Basic Materials": 1.5,
    "Real Estate": 5.0,
    "Utilities": 1.8,
}


def _ps_verdict(ps_ratio: float, industry: str, sector: str) -> dict:
    benchmark = _PS_BY_INDUSTRY.get(industry) or _PS_BY_SECTOR.get(sector)
    if benchmark is None:
        return {
            "ps_ratio": round(ps_ratio, 2),
            "ps_benchmark": None,
            "ps_benchmark_label": None,
            "verdict": f"P/S {ps_ratio:.2f} — no benchmark available for {industry or sector}",
        }

    label = industry if industry in _PS_BY_INDUSTRY else sector
    pct = ((ps_ratio - benchmark) / benchmark) * 100
    if pct < -20:
        rating = "Undervalued vs peers"
    elif pct > 20:
        rating = "Overvalued vs peers"
    else:
        rating = "In line with peers"

    return {
        "ps_ratio": round(ps_ratio, 2),
        "ps_benchmark": benchmark,
        "ps_benchmark_label": label,
        "rating": rating,
        "pct_diff": round(pct, 1),
        "verdict": f"{rating} ({pct:+.1f}% vs {label} median P/S of {benchmark}x)",
    }


def _historical_fcf_growth(stock) -> float | None:
    try:
        fcf_row = stock.cashflow.loc["Free Cash Flow"].dropna().sort_index()
        if len(fcf_row) < 2:
            return None
        oldest, newest = fcf_row.iloc[0], fcf_row.iloc[-1]
        if oldest <= 0:
            return None
        years = len(fcf_row) - 1
        return (newest / oldest) ** (1 / years) - 1
    except (KeyError, ZeroDivisionError):
        return None


def dcf(fcf_per_share: float, growth: float, discount_rate: float,
        terminal_growth: float = 0.025) -> float:
    pv = 0.0
    cf = fcf_per_share
    taper = growth / 2

    for i in range(1, 6):
        cf *= (1 + growth)
        pv += cf / (1 + discount_rate) ** i

    for i in range(6, 11):
        cf *= (1 + taper)
        pv += cf / (1 + discount_rate) ** i

    terminal_value = cf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv += terminal_value / (1 + discount_rate) ** 10

    return pv


def assess(ticker: str, discount_rate: float = 0.10) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    fcf = info.get("freeCashflow")
    shares = info.get("sharesOutstanding")

    # GBp (pence) prices need converting to pounds to match FCF/share units
    currency = info.get("currency", "")
    if currency == "GBp" and price is not None:
        price = price / 100

    growth = info.get("earningsGrowth")
    growth_source = "analyst estimate"
    if growth is None:
        growth = _historical_fcf_growth(stock)
        growth_source = "historical FCF CAGR"

    result = {
        "ticker": ticker.upper(),
        "price": price,
        "currency": "GBP" if currency == "GBp" else currency,
        "fcf": fcf,
        "shares": shares,
        "growth": growth,
        "growth_source": growth_source,
        "discount_rate": discount_rate,
        "intrinsic_value": None,
        "verdict": None,
    }

    if fcf is None or shares is None or shares == 0:
        result["verdict"] = "Insufficient data (missing FCF or share count)"
        return result

    fcf_per_share = fcf / shares
    if fcf_per_share <= 0:
        ps = info.get("priceToSalesTrailing12Months")
        industry = info.get("industry", "")
        sector = info.get("sector", "")
        if ps is not None:
            ps_result = _ps_verdict(ps, industry, sector)
            result.update(ps_result)
            result["method"] = "P/S (negative FCF)"
        else:
            result["verdict"] = "Negative FCF and no P/S data available"
        return result

    if growth is None:
        result["verdict"] = "Insufficient data (no growth rate available)"
        return result

    # Cap growth to prevent absurd projections from high-growth outliers
    growth = min(growth, 0.25)

    iv = dcf(fcf_per_share, growth, discount_rate)
    result["intrinsic_value"] = round(iv, 2)
    result["method"] = "DCF"

    if price is not None:
        pct = ((price - iv) / iv) * 100
        if pct < -20:
            rating = "Undervalued"
        elif pct > 20:
            rating = "Overvalued"
        else:
            rating = "Fairly valued"
        result["rating"] = rating
        result["pct_diff"] = round(pct, 1)
        result["verdict"] = f"{rating} ({pct:+.1f}% vs intrinsic value)"

    return result
