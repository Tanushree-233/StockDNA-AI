import os
import pandas as pd
import yfinance as yf

TICKERS = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]

OUTPUT = "data/processed/internal/internal_quarterly.csv"

os.makedirs("data/processed/internal", exist_ok=True)

rows = []

for ticker in TICKERS:

    print(f"\nCollecting internal data for {ticker}...")

    stock = yf.Ticker(ticker)

    income = stock.quarterly_income_stmt
    balance = stock.quarterly_balance_sheet
    cashflow = stock.quarterly_cashflow

    dates = set()

    if not income.empty:
        dates.update(income.columns)

    if not balance.empty:
        dates.update(balance.columns)

    if not cashflow.empty:
        dates.update(cashflow.columns)

    for date in sorted(dates):

        today = pd.Timestamp.today().normalize()

        if pd.Timestamp(date) > today:
            continue

        def get_value(statement, name):
            try:
                if name in statement.index and date in statement.columns:
                    value = statement.loc[name, date]
                    return float(value) if pd.notna(value) else None
            except Exception:
                pass

            return None

        revenue = get_value(income, "Total Revenue")
        operating_income = get_value(income, "Operating Income")
        net_income = get_value(income, "Net Income")
        eps = get_value(income, "Diluted EPS")

        total_assets = get_value(balance, "Total Assets")
        total_debt = get_value(balance, "Total Debt")
        equity = get_value(balance, "Stockholders Equity")
        cash = get_value(
            balance,
            "Cash Cash Equivalents And Short Term Investments"
        )

        operating_cashflow = get_value(
            cashflow,
            "Operating Cash Flow"
        )

        free_cashflow = get_value(
            cashflow,
            "Free Cash Flow"
        )

        capex = get_value(
            cashflow,
            "Capital Expenditure"
        )

        rows.append({
            "Ticker": ticker.replace(".NS", ""),
            "FinancialDate": pd.Timestamp(date),

            "Revenue": revenue,
            "OperatingIncome": operating_income,
            "NetIncome": net_income,
            "DilutedEPS": eps,

            "TotalAssets": total_assets,
            "TotalDebt": total_debt,
            "StockholdersEquity": equity,
            "Cash": cash,

            "OperatingCashFlow": operating_cashflow,
            "FreeCashFlow": free_cashflow,
            "CapitalExpenditure": capex
        })

df = pd.DataFrame(rows)

df = df.sort_values(
    ["Ticker", "FinancialDate"]
).reset_index(drop=True)

# Derived internal factors

df["OperatingMargin"] = (
    df["OperatingIncome"] / df["Revenue"]
)

df["NetMargin"] = (
    df["NetIncome"] / df["Revenue"]
)

df["DebtToAssets"] = (
    df["TotalDebt"] / df["TotalAssets"]
)

df["DebtToEquity"] = (
    df["TotalDebt"] / df["StockholdersEquity"]
)

df["RevenueGrowth"] = (
    df.groupby("Ticker")["Revenue"].pct_change()
)

df["NetIncomeGrowth"] = (
    df.groupby("Ticker")["NetIncome"].pct_change()
)

df["EPSGrowth"] = (
    df.groupby("Ticker")["DilutedEPS"].pct_change()
)

df["FCFGrowth"] = (
    df.groupby("Ticker")["FreeCashFlow"].pct_change()
)

df.replace(
    [float("inf"), float("-inf")],
    pd.NA,
    inplace=True
)

df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 60)
print("INTERNAL DATASET CREATED")
print("=" * 60)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nRows by company:")
print(df["Ticker"].value_counts())

print("\nDate range:")
print(df["FinancialDate"].min(), "→", df["FinancialDate"].max())

print("\nSaved:")
print(OUTPUT)