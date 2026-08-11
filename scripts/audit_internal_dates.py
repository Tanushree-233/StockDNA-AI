import pandas as pd

INTERNAL_DATA = "data/processed/internal/internal_quarterly.csv"
EARNINGS_DATA = "data/processed/internal/earnings_events.csv"

print("=" * 70)
print("INTERNAL FINANCIAL ANNOUNCEMENT DATE AUDIT")
print("=" * 70)

# ============================================================
# 1. LOAD
# ============================================================

internal = pd.read_csv(INTERNAL_DATA)
earnings = pd.read_csv(EARNINGS_DATA)

# ============================================================
# 2. CONVERT DATES
# ============================================================

internal["FinancialDate"] = pd.to_datetime(
    internal["FinancialDate"],
    errors="coerce"
)

earnings["Date"] = pd.to_datetime(
    earnings["Date"],
    errors="coerce"
)

# ============================================================
# 3. NORMALIZE TICKERS
# ============================================================

internal["Ticker"] = (
    internal["Ticker"]
    .astype(str)
    .str.replace(".NS", "", regex=False)
    .str.strip()
)

earnings["Ticker"] = (
    earnings["Ticker"]
    .astype(str)
    .str.replace(".NS", "", regex=False)
    .str.strip()
)

# ============================================================
# 4. BASIC CHECKS
# ============================================================

print("\nInternal rows:", len(internal))
print("Earnings rows:", len(earnings))

print("\nInternal columns:")
print(list(internal.columns))

print("\nEarnings columns:")
print(list(earnings.columns))

print("\nMissing FinancialDate:")
print(internal["FinancialDate"].isna().sum())

print("Missing earnings Date:")
print(earnings["Date"].isna().sum())

# ============================================================
# 5. SORT
# ============================================================

internal = (
    internal
    .dropna(subset=["FinancialDate"])
    .sort_values(["Ticker", "FinancialDate"])
    .reset_index(drop=True)
)

earnings = (
    earnings
    .dropna(subset=["Date"])
    .sort_values(["Ticker", "Date"])
    .reset_index(drop=True)
)

# ============================================================
# 6. FIND NEXT ANNOUNCEMENT
# ============================================================

announcement_data = earnings[
    ["Ticker", "Date"]
].copy()

announcement_data = announcement_data.rename(
    columns={
        "Date": "AnnouncementDate"
    }
)

# Use global date sorting for merge_asof compatibility
internal_for_merge = internal.sort_values(
    ["FinancialDate", "Ticker"]
).reset_index(drop=True)

announcement_for_merge = announcement_data.sort_values(
    ["AnnouncementDate", "Ticker"]
).reset_index(drop=True)

matched = pd.merge_asof(
    internal_for_merge,
    announcement_for_merge,
    left_on="FinancialDate",
    right_on="AnnouncementDate",
    by="Ticker",
    direction="forward"
)

# ============================================================
# 7. CALCULATE GAP
# ============================================================

matched["AnnouncementGapDays"] = (
    matched["AnnouncementDate"]
    - matched["FinancialDate"]
).dt.days

# ============================================================
# 8. SHOW MATCHES
# ============================================================

print("\n" + "=" * 70)
print("FINANCIAL DATE → ANNOUNCEMENT DATE MATCHES")
print("=" * 70)

display_columns = [
    "Ticker",
    "FinancialDate",
    "AnnouncementDate",
    "AnnouncementGapDays"
]

print(
    matched[display_columns]
    .to_string(index=False)
)

# ============================================================
# 9. NEGATIVE GAPS
# ============================================================

negative = matched[
    matched["AnnouncementGapDays"] < 0
]

print("\n" + "=" * 70)
print("NEGATIVE DATE CHECK")
print("=" * 70)

print(
    "Announcements before financial date:",
    len(negative)
)

if len(negative) > 0:
    print("\nWARNING:")
    print(
        negative[display_columns]
        .to_string(index=False)
    )
else:
    print("PASS: No announcement occurs before FinancialDate.")

# ============================================================
# 10. SAME-DAY ANNOUNCEMENTS
# ============================================================

same_day = matched[
    matched["AnnouncementGapDays"] == 0
]

print("\nSame-day announcements:", len(same_day))

# ============================================================
# 11. VERY LARGE GAPS
# ============================================================

large_gap = matched[
    matched["AnnouncementGapDays"] > 120
]

print("\n" + "=" * 70)
print("LARGE ANNOUNCEMENT GAP CHECK")
print("=" * 70)

print(
    "Matches with gap > 120 days:",
    len(large_gap)
)

if len(large_gap) > 0:
    print("\nPotentially suspicious matches:")
    print(
        large_gap[display_columns]
        .to_string(index=False)
    )

# ============================================================
# 12. GAP SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANNOUNCEMENT GAP SUMMARY")
print("=" * 70)

print(
    matched["AnnouncementGapDays"]
    .describe()
)

# ============================================================
# 13. PER-TICKER SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PER-TICKER SUMMARY")
print("=" * 70)

summary = (
    matched
    .groupby("Ticker")["AnnouncementGapDays"]
    .agg(
        quarters="count",
        min_gap="min",
        median_gap="median",
        max_gap="max"
    )
)

print(summary.to_string())

# ============================================================
# 14. FINAL
# ============================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)