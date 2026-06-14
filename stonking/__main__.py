import argparse
import sys
from .valuation import assess


def main():
    parser = argparse.ArgumentParser(description="DCF stock valuation using yfinance")
    parser.add_argument("tickers", nargs="+", help="Stock ticker symbols")
    parser.add_argument(
        "--discount-rate",
        type=float,
        default=0.10,
        metavar="RATE",
        help="Discount rate as a decimal (default: 0.10 = 10%%)",
    )
    args = parser.parse_args()

    for ticker in args.tickers:
        r = assess(ticker, discount_rate=args.discount_rate)
        ccy = r.get("currency", "")
        method = r.get("method", "N/A")
        print(f"\n{r['ticker']}  [{method}]")
        print(f"  Price:            {r['price']} {ccy}")
        if method == "DCF":
            growth_pct = f"{r['growth'] * 100:.1f}%" if r["growth"] is not None else "N/A"
            fcf_ps = round(r['fcf'] / r['shares'], 2) if r['fcf'] and r['shares'] else 'N/A'
            print(f"  FCF/share:        {fcf_ps} {ccy}")
            print(f"  Growth rate:      {growth_pct} ({r['growth_source']})")
            print(f"  Discount rate:    {r['discount_rate'] * 100:.1f}%")
            print(f"  Intrinsic value:  {r['intrinsic_value']} {ccy}")
        elif method == "P/S (negative FCF)":
            print(f"  P/S ratio:        {r.get('ps_ratio')}")
            print(f"  Sector benchmark: {r.get('ps_benchmark')}x ({r.get('ps_benchmark_label')})")
        print(f"  Verdict:          {r['verdict']}")


if __name__ == "__main__":
    main()
