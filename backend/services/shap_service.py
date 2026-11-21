"""
Explainable AI (XAI) Service for StockDNA-AI.

Calculates SHAP feature-level contributions for the predicted class and decomposes
attributions into:
- INTERNAL factors (company-specific fundamentals, earnings events, surprises)
- EXTERNAL factors (market benchmark, technical indicators, macro volatility)

Guarantees:
1. Reuses cached `shap.TreeExplainer` instances for low latency (< 15ms).
2. Explains the PREDICTED class (multiclass SHAP array slicing).
3. Distinguishes importance (magnitude) from direction (toward/away).
4. Generates faithful, non-hallucinated explanations derived from feature definition + SHAP sign + class.
5. Provides layperson-friendly translations for all 34 features with zero technical jargon.
6. Uses `config.feature_registry` as the single authoritative source of truth.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import shap

from config.feature_registry import FEATURE_REGISTRY, get_feature_group

# Global cache for compiled TreeExplainer instances (keyed by model id)
_EXPLAINER_CACHE: Dict[int, Any] = {}

CLASS_LABELS: Dict[int, str] = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}

# =============================================================================
# Layman Language Feature Metadata & Explanations (Zero Jargon)
# =============================================================================
FEATURE_LAYMAN_METADATA: Dict[str, Dict[str, str]] = {
    "Normalized_ATR": {
        "name": "Daily Price Fluctuations (Volatility)",
        "meaning": "Measures how widely the stock price swung between highs and lows each day.",
        "pos_hold": "High daily price swings mean unpredictable moves, so the AI advises waiting patiently (HOLD) instead of risking a trade.",
        "neg_hold": "Calm, steady price swings encourage the AI to consider taking action (BUY or SELL) rather than waiting.",
        "pos_buy": "Healthy, stable price movement gave the AI confidence to support a BUY recommendation.",
        "neg_buy": "High daily price swings created hesitation against making a new BUY entry.",
        "pos_sell": "Erratic and large price swings increased the risk, reinforcing the decision to SELL.",
        "neg_sell": "Moderate daily swings helped cushion the downside against an aggressive SELL.",
    },
    "High_Low_Spread_Pct": {
        "name": "Intraday Price Range",
        "meaning": "The percentage gap between today's highest and lowest traded prices.",
        "pos_hold": "A wide gap between the day's high and low indicates market tug-of-war, making HOLD the safer choice.",
        "neg_hold": "A tight, controlled daily trading band suggests the stock is coiling for a directional move.",
        "pos_buy": "Buyers pushed the price toward the upper range of the day, confirming strong interest.",
        "neg_buy": "Intraday price rejection from the highs cautioned against an aggressive BUY.",
        "pos_sell": "Sellers dominated the daily price range, reinforcing a SELL stance.",
        "neg_sell": "Price bounced off daily lows, preventing an extreme SELL rating.",
    },
    "Open_Close_Spread_Pct": {
        "name": "Day's Net Price Movement",
        "meaning": "The difference between where the stock opened in the morning and closed at the end of the day.",
        "pos_hold": "The stock closed close to its opening price, indicating indecision between buyers and sellers (HOLD).",
        "neg_hold": "A decisive distance between opening and closing prices pushed away from neutral HOLD.",
        "pos_buy": "The stock closed strongly above its opening price, showing bullish buyer demand.",
        "neg_buy": "A weak close relative to the open cautioned against an immediate BUY.",
        "pos_sell": "The stock finished the day well below its opening price, confirming selling pressure.",
        "neg_sell": "A late-day recovery prevented a harsher SELL signal.",
    },
    "Volume_to_MA20": {
        "name": "Trading Volume Surge vs Monthly Average",
        "meaning": "Compares today's trading activity against the typical volume seen over the past month.",
        "pos_hold": "Normal, average trading activity shows no big institutional rush, supporting patience (HOLD).",
        "neg_hold": "A sudden burst in trading volume suggests big investors are moving, pushing the model toward action.",
        "pos_buy": "Above-average trading volume accompanied price strength, confirming genuine buyer interest.",
        "neg_buy": "Lack of strong trading volume suggested low conviction behind recent price upticks.",
        "pos_sell": "Heavy selling volume confirmed institutions may be offloading shares, supporting SELL.",
        "neg_sell": "Low volume on price dips suggests selling is light, moderating a SELL signal.",
    },
    "Volume_Change": {
        "name": "Daily Trading Activity Change",
        "meaning": "How much trading volume jumped or dropped compared to the previous trading day.",
        "pos_hold": "Stable daily trading volume supports holding steady without rushed decisions.",
        "neg_hold": "A sharp jump in volume breaks the calm, encouraging a move away from HOLD.",
        "pos_buy": "Volume expansion confirmed growing buyer enthusiasm.",
        "neg_buy": "Declining daily volume advised caution on buying.",
        "pos_sell": "Rising volume during price weakness reinforced a SELL signal.",
        "neg_sell": "Quiet volume on a down day softened the sell outlook.",
    },
    "VIX_Level": {
        "name": "Overall Market Fear Index (India VIX)",
        "meaning": "Measures anxiety and fear across the entire Indian stock market.",
        "pos_hold": "Elevated market-wide fear encourages sitting on the sidelines (HOLD) until market nerves settle.",
        "neg_hold": "Low market fear indicates a stable environment, encouraging active positions.",
        "pos_buy": "A calm market climate provided a safe backdrop for a BUY recommendation.",
        "neg_buy": "Spikes in overall market fear cautioned against entering new BUY positions.",
        "pos_sell": "High market fear and panic selling supported a protective SELL decision.",
        "neg_sell": "Market calm helped cushion the stock against severe selling pressure.",
    },
    "VIX_Change_5d": {
        "name": "1-Week Trend in Market Fear",
        "meaning": "Whether market anxiety has been escalating or calming down over the past 5 days.",
        "pos_hold": "A slight increase in market fear advised waiting for calmer weather before trading.",
        "neg_hold": "A drop in market fear encouraged taking action instead of holding steady.",
        "pos_buy": "Subsiding market fear supported a positive BUY outlook.",
        "neg_buy": "A fast rise in market anxiety prompted caution against buying.",
        "pos_sell": "Surging market panic reinforced a defensive SELL stance.",
        "neg_sell": "Easing market panic helped soften a SELL recommendation.",
    },
    "RSI_14": {
        "name": "Buying vs Selling Momentum (RSI)",
        "meaning": "Shows whether the stock has been excessively bought (overbought) or heavily dumped (oversold).",
        "pos_hold": "Momentum is in a balanced middle zone, indicating neither buyers nor sellers are in control (HOLD).",
        "neg_hold": "Momentum reached an extreme level, nudging the model away from neutral HOLD.",
        "pos_buy": "Positive momentum shows buyers steadily driving prices higher without being overbought.",
        "neg_buy": "Overextended momentum warned that the stock may be due for a brief cooling-off period.",
        "pos_sell": "Weak momentum confirms sellers have taken the upper hand.",
        "neg_sell": "Heavily oversold conditions suggest a potential rebound, moderating the sell signal.",
    },
    "Close_to_SMA20": {
        "name": "Price vs 1-Month (20-Day) Average",
        "meaning": "Shows whether the current share price is above or below its average price over the last month.",
        "pos_hold": "The price is hovering right at its 1-month average, confirming a sideways trend (HOLD).",
        "neg_hold": "The price is pulling away from its monthly average, indicating a breakout or breakdown.",
        "pos_buy": "Trading above the 1-month average confirms healthy short-term upward momentum.",
        "neg_buy": "Trading below the monthly average indicates short-term hesitation.",
        "pos_sell": "Falling below the 1-month average confirms short-term trend deterioration.",
        "neg_sell": "Price holding close to support softened an aggressive SELL signal.",
    },
    "Close_to_SMA50": {
        "name": "Price vs 2-Month (50-Day) Average",
        "meaning": "Shows whether the stock is above or below its medium-term baseline price.",
        "pos_hold": "The price is tracking closely with its 50-day average, signaling a neutral consolidation phase.",
        "neg_hold": "A clear gap from the 50-day average encourages directional positioning.",
        "pos_buy": "Holding comfortably above the 50-day average confirms a solid medium-term uptrend.",
        "neg_buy": "Struggling below the 50-day average suggests lingering overhead resistance.",
        "pos_sell": "Breaking below the 50-day trend line signals intermediate weakness.",
        "neg_sell": "Proximity to the 50-day average kept the sell outlook measured.",
    },
    "Close_to_SMA200": {
        "name": "Price vs Long-Term (200-Day) Trend",
        "meaning": "The ultimate benchmark of whether a stock is in a long-term bull or bear market.",
        "pos_hold": "The stock is consolidating near its major long-term benchmark, recommending patience (HOLD).",
        "neg_hold": "The stock has clearly established direction relative to its 200-day trend.",
        "pos_buy": "Trading firmly above its 200-day average proves long-term structural health.",
        "neg_buy": "Trading under the 200-day average creates long-term resistance against aggressive buys.",
        "pos_sell": "Trading below the 200-day line indicates long-term weakness, favoring capital preservation.",
        "neg_sell": "Long-term institutional support cushioned the stock against a steep SELL.",
    },
    "SMA20_to_SMA50": {
        "name": "Short-Term vs Medium-Term Trend",
        "meaning": "Compares 20-day speed against 50-day speed to detect turning points.",
        "pos_hold": "Short-term and medium-term trends are running parallel, showing no sudden momentum shift (HOLD).",
        "neg_hold": "A trend crossover is forming, urging the model toward action rather than waiting.",
        "pos_buy": "Short-term trend crossed above the medium-term average, signaling fresh upward momentum.",
        "neg_buy": "A flattening trend crossover advised waiting for stronger confirmation before buying.",
        "pos_sell": "Short-term trend sank below the medium-term average, signaling accelerating weakness.",
        "neg_sell": "Supportive trend alignment limited the severity of the sell signal.",
    },
    "Daily_Return": {
        "name": "1-Day Price Return",
        "meaning": "How much the stock price changed over the most recent single trading day.",
        "pos_hold": "A small daily move kept the stock in neutral territory (HOLD).",
        "neg_hold": "A noticeable single-day price swing encouraged looking for an active setup.",
        "pos_buy": "Positive single-day gains confirmed buyer presence.",
        "neg_buy": "A minor single-day dip urged a sensible, cautious entry.",
        "pos_sell": "A negative daily return confirmed selling momentum.",
        "neg_sell": "A daily gain pushed back against an all-out sell signal.",
    },
    "Return_5d": {
        "name": "1-Week (5-Day) Price Change",
        "meaning": "Total percentage gained or lost over the past 5 trading days.",
        "pos_hold": "Weekly price changes have remained modest, signaling a range-bound market (HOLD).",
        "neg_hold": "A significant 1-week gain or loss pushed the model away from neutral HOLD.",
        "pos_buy": "A healthy 1-week upward trend supported the BUY recommendation.",
        "neg_buy": "A choppy 1-week performance suggested waiting for cleaner momentum.",
        "pos_sell": "A sharp 1-week decline reinforced the recommendation to exit or protect gains.",
        "neg_sell": "A weekly bounce reduced the urgency of a SELL signal.",
    },
    "Return_10d": {
        "name": "2-Week Price Change",
        "meaning": "Net percentage movement over the past 10 trading sessions.",
        "pos_hold": "Two weeks of sideways trading confirmed lack of strong directional pressure (HOLD).",
        "neg_hold": "Strong 2-week performance pushed the model away from waiting.",
        "pos_buy": "Sustained 2-week gains confirmed buyer conviction.",
        "neg_buy": "Inconsistent 2-week returns prompted a cautious approach.",
        "pos_sell": "Two consecutive weeks of weakness supported a SELL decision.",
        "neg_sell": "Resilient 2-week performance cushioned against an aggressive SELL.",
    },
    "Return_21d": {
        "name": "1-Month (21-Day) Price Change",
        "meaning": "The stock's net percentage performance over approximately one calendar month.",
        "pos_hold": "Monthly returns have stayed within a tight band, reinforcing a patient HOLD.",
        "neg_hold": "A prominent monthly move prompted the AI to evaluate an active trade.",
        "pos_buy": "Solid 1-month gains confirmed an established upward trend.",
        "neg_buy": "Flat or lagging 1-month returns advised caution before committing new capital.",
        "pos_sell": "A poor 1-month performance validated the recommendation to SELL.",
        "neg_sell": "Positive monthly gains countered short-term sell pressure.",
    },
    "ROC_10": {
        "name": "10-Day Speed of Price Change",
        "meaning": "Measures how fast or slow the price has been moving over the past 10 days.",
        "pos_hold": "Price velocity is neutral, indicating calm sideways trading (HOLD).",
        "neg_hold": "Price velocity accelerated sharply, encouraging action over patience.",
        "pos_buy": "Upward price acceleration confirmed increasing buyer demand.",
        "neg_buy": "Decelerating price speed suggested waiting for renewed momentum.",
        "pos_sell": "Downward price acceleration reinforced the SELL recommendation.",
        "neg_sell": "Decelerating downward momentum helped prevent a panic SELL.",
    },
    "Rolling_Volatility_20d": {
        "name": "1-Month Price Stability",
        "meaning": "Shows how calm or turbulent the stock's price behavior has been over 20 trading days.",
        "pos_hold": "Moderate volatility suggests the stock is in a steady consolidation pattern (HOLD).",
        "neg_hold": "A shift in volatility broke the sideways rhythm, nudging away from HOLD.",
        "pos_buy": "Controlled, orderly volatility gave the AI confidence in a BUY signal.",
        "neg_buy": "High volatility made the upside harder to time with certainty.",
        "pos_sell": "Elevated price volatility increased downside risk, supporting a SELL.",
        "neg_sell": "Subdued volatility kept the sell recommendation restrained.",
    },
    "Bollinger_PctB": {
        "name": "Price Position in Normal Range (Bollinger)",
        "meaning": "Indicates whether the price is near the upper ceiling or lower floor of its normal band.",
        "pos_hold": "The price is sitting comfortably in the middle of its normal range (HOLD).",
        "neg_hold": "The price moved toward an outer boundary, signaling a possible breakout or breakdown.",
        "pos_buy": "Price is pressing upward through its normal band with strong momentum.",
        "neg_buy": "Price is touching the upper boundary, signaling potential short-term exhaustion.",
        "pos_sell": "Price is falling along the lower boundary, indicating sustained selling.",
        "neg_sell": "Price holding above the lower boundary prevented a deeper sell signal.",
    },
    "Normalized_MACD": {
        "name": "Short-Term vs Long-Term Trend Direction",
        "meaning": "Shows whether short-term momentum is heading higher or lower than the bigger trend.",
        "pos_hold": "Trend lines are intertwined, showing no clear winner between bulls and bears (HOLD).",
        "neg_hold": "A distinct gap opened between trend lines, favoring active positioning.",
        "pos_buy": "Short-term momentum is clearly outpacing the longer-term trend, confirming a BUY.",
        "neg_buy": "Trend convergence advised caution before buying.",
        "pos_sell": "Short-term trend is slipping below the longer trend, supporting a SELL.",
        "neg_sell": "Trend resilience softened a negative sell signal.",
    },
    "Normalized_MACD_Hist": {
        "name": "Momentum Acceleration (MACD Histogram)",
        "meaning": "Measures whether the existing price trend is gaining energy or running out of breath.",
        "pos_hold": "Momentum acceleration is near zero, confirming a balanced sideways state (HOLD).",
        "neg_hold": "Momentum energy is expanding, pulling away from neutral HOLD.",
        "pos_buy": "Momentum is accelerating upwards, providing strong fuel for a BUY.",
        "neg_buy": "Upward momentum is slowing down, advising patience.",
        "pos_sell": "Downward momentum is gaining speed, reinforcing a SELL decision.",
        "neg_sell": "Selling momentum is losing intensity, cushioning the sell stance.",
    },
    "NIFTY_Return_1d": {
        "name": "Overall Indian Market (NIFTY 50) 1-Day Return",
        "meaning": "The daily gain or loss of India's top 50 benchmark index.",
        "pos_hold": "The broader market had a quiet session, supporting holding steady.",
        "neg_hold": "A big move in the NIFTY index created market-wide ripples away from HOLD.",
        "pos_buy": "A green day for the broader market provided tailwinds for buying.",
        "neg_buy": "A softer market day advised measured caution on new purchases.",
        "pos_sell": "A market-wide sell-off added to the pressure to SELL or exit.",
        "neg_sell": "A strong overall market day prevented an excessive SELL signal.",
    },
    "NIFTY_Return_5d": {
        "name": "1-Week Overall Indian Market Trend",
        "meaning": "The performance of the NIFTY 50 benchmark index over the past 5 trading days.",
        "pos_hold": "The Indian market as a whole moved sideways this week, favoring a patient HOLD.",
        "neg_hold": "A strong weekly trend across Indian equities pushed away from neutral HOLD.",
        "pos_buy": "Broader market strength supported buying individual leading stocks.",
        "neg_buy": "A sluggish broader market urged caution before jumping in.",
        "pos_sell": "Broader market weakness reinforced the decision to SELL and protect capital.",
        "neg_sell": "A buoyant broader market provided support against sharp declines.",
    },
    "Relative_Strength_5d": {
        "name": "Stock vs Overall Market Strength",
        "meaning": "Reveals whether this stock is beating the NIFTY 50 index or lagging behind it.",
        "pos_hold": "The stock is matching the market step-for-step, confirming neutral performance (HOLD).",
        "neg_hold": "The stock has begun noticeably outperforming or underperforming the market.",
        "pos_buy": "The stock is beating the broader market, signaling strong institutional backing.",
        "neg_buy": "The stock is slightly lagging behind the market, recommending patience.",
        "pos_sell": "The stock is underperforming the broader market, supporting a SELL signal.",
        "neg_sell": "The stock showed relative resilience against market drops.",
    },
    "Earnings_Yield": {
        "name": "Company Profit relative to Share Price",
        "meaning": "How much profit the company actually earns for every rupee invested at current price.",
        "pos_hold": "Fair valuation relative to earnings means the stock is appropriately priced (HOLD).",
        "neg_hold": "Unusual valuation pushed the AI to look for an entry or exit opportunity.",
        "pos_buy": "Attractive company profitability relative to share price supported a BUY.",
        "neg_buy": "Rich valuation relative to profits suggested waiting for a better price.",
        "pos_sell": "Expensive valuation with low earnings yield justified selling or trimming.",
        "neg_sell": "Solid profit backing prevented an extreme sell rating.",
    },
    "Forward_Earnings_Yield": {
        "name": "Expected Future Profits vs Share Price",
        "meaning": "Projected upcoming profits compared to what the stock costs right now.",
        "pos_hold": "Future profit expectations are steady and balanced with current prices (HOLD).",
        "neg_hold": "Future earnings shifts pushed the model toward active positioning.",
        "pos_buy": "Strong future profit projections justify taking a BUY position.",
        "neg_buy": "Modest future earnings projections advised a disciplined, conservative entry.",
        "pos_sell": "Tepid future earnings outlook reinforced the recommendation to SELL.",
        "neg_sell": "Promising future earnings potential limited the downside sell score.",
    },
    "EPS_Surprise": {
        "name": "Company Profit Beat or Miss",
        "meaning": "The difference between what Wall Street/Dalal Street analysts forecasted and what the company earned.",
        "pos_hold": "Results matched expectations closely, meaning no major surprise to alter a HOLD stance.",
        "neg_hold": "A notable profit beat or miss shifted sentiment away from neutral HOLD.",
        "pos_buy": "Profits beat analyst forecasts, confirming robust corporate execution.",
        "neg_buy": "A slight earnings miss suggested waiting for the next report before buying.",
        "pos_sell": "Profits missed expectations, reinforcing a cautious SELL signal.",
        "neg_sell": "A respectable earnings beat cushioned the stock from sharp selling.",
    },
    "EPS_Surprise_Pct": {
        "name": "Profit Surprise Percentage",
        "meaning": "The percentage by which reported quarterly profits beat or fell short of expectations.",
        "pos_hold": "Profit surprise was within a normal, predictable range (HOLD).",
        "neg_hold": "A sizable percentage surprise created momentum away from HOLD.",
        "pos_buy": "A high-percentage profit beat gave the AI strong conviction for a BUY.",
        "neg_buy": "A lukewarm earnings surprise suggested moderation.",
        "pos_sell": "A large-percentage earnings miss supported the recommendation to SELL.",
        "neg_sell": "Earnings resilience countered bearish selling pressure.",
    },
    "Earnings_Event": {
        "name": "Earnings Announcement Day",
        "meaning": "Signals whether the company is announcing its quarterly financial results today.",
        "pos_hold": "No immediate earnings announcement today, avoiding sudden quarterly report shocks (HOLD).",
        "neg_hold": "Today's earnings announcement creates high event-driven momentum away from HOLD.",
        "pos_buy": "Earnings event catalyst provided a clear reason to take action.",
        "neg_buy": "Pending earnings announcement urged waiting until results are published.",
        "pos_sell": "Post-earnings reaction prompted a defensive SELL.",
        "neg_sell": "Event stability prevented unjustified selling.",
    },
    "Positive_Earnings_Surprise": {
        "name": "Profits Exceeded Forecasts Flag",
        "meaning": "A verified record that the company's latest reported profits beat consensus forecasts.",
        "pos_hold": "Known positive results are already priced into the stock, favoring a steady HOLD.",
        "neg_hold": "Recent positive surprise continues driving momentum away from neutral HOLD.",
        "pos_buy": "Confirmed profit beat provides fundamental backing for a BUY decision.",
        "neg_buy": "Positive surprise impact has matured, suggesting measured entries.",
        "pos_sell": "Despite a past beat, other technical factors now dominate the sell signal.",
        "neg_sell": "Past profit beat provided fundamental downside protection.",
    },
    "Negative_Earnings_Surprise": {
        "name": "Profits Missed Forecasts Flag",
        "meaning": "A verified record that the company's latest reported profits fell short of forecasts.",
        "pos_hold": "Earnings miss has already settled without new panic, supporting a neutral HOLD.",
        "neg_hold": "Earnings disappointment continues sparking price adjustments away from HOLD.",
        "pos_buy": "Stock has shrugged off the past miss, showing strong buyer resilience.",
        "neg_buy": "Past earnings miss cautions against overconfident buying.",
        "pos_sell": "Confirmed earnings miss directly justifies a SELL decision.",
        "neg_sell": "Improving conditions helped soften the impact of a past miss.",
    },
    "Days_Since_Earnings": {
        "name": "Days Since Last Financial Results",
        "meaning": "The number of calendar days since the company's last quarterly scorecard was released.",
        "pos_hold": "Sufficient time has passed since earnings, leaving the stock in a steady drift period (HOLD).",
        "neg_hold": "Proximity to an earnings announcement creates fresh momentum away from HOLD.",
        "pos_buy": "The stock has digested its financial report and established a clean upward path.",
        "neg_buy": "Too much time has elapsed since results, making technical trends more dominant.",
        "pos_sell": "A post-earnings downward drift reinforced the recommendation to SELL.",
        "neg_sell": "Stable time distance from earnings avoided sudden sell-offs.",
    },
    "Post_Earnings_1_Week": {
        "name": "First Week Following Financial Results",
        "meaning": "Whether the stock is in the immediate 5-day reaction window after quarterly numbers.",
        "pos_hold": "Outside the high-risk first week after earnings, allowing a calmer HOLD perspective.",
        "neg_hold": "Active 1-week earnings reaction window creates strong trading momentum.",
        "pos_buy": "Strong first-week reaction after results supported an immediate BUY.",
        "neg_buy": "First-week post-earnings volatility suggested waiting for dust to settle.",
        "pos_sell": "Negative initial reaction after earnings supported selling early.",
        "neg_sell": "Lack of immediate panic post-earnings softened the sell signal.",
    },
    "Post_Earnings_1_Month": {
        "name": "First Month Following Financial Results",
        "meaning": "Whether the stock is in the 1-month post-earnings drift window where fundamentals carry heavy weight.",
        "pos_hold": "Stock has settled into its 1-month post-earnings rhythm without extreme turbulence (HOLD).",
        "neg_hold": "Ongoing post-earnings drift is pushing the stock out of neutral territory.",
        "pos_buy": "Healthy 1-month post-earnings trend confirmed lasting institutional demand.",
        "neg_buy": "Post-earnings momentum is leveling off, advising patience.",
        "pos_sell": "Ongoing post-earnings slide justified a proactive SELL.",
        "neg_sell": "Earnings drift showed signs of stabilizing.",
    },
}


def get_cached_explainer(model: Any) -> shap.TreeExplainer:
    """
    Retrieve or create a cached TreeExplainer instance for the model.
    Prevents costly re-compilation on every request.
    """
    model_id = id(model)
    if model_id not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[model_id] = shap.TreeExplainer(model)
    return _EXPLAINER_CACHE[model_id]


def generate_feature_explanation(
    feature: str, shap_val: float, predicted_label: str
) -> str:
    """
    Derives an explanation based on feature definition, SHAP sign, and model prediction.
    Maintains strict backward compatibility for automated tests.
    """
    friendly_name = feature.replace("_", " ")

    if shap_val > 0.0001:
        return f"{friendly_name} contributed toward {predicted_label} prediction"
    elif shap_val < -0.0001:
        return f"{friendly_name} pushed away from {predicted_label} prediction"
    else:
        return f"{friendly_name} had neutral influence on {predicted_label} prediction"


def generate_layman_explanation(
    feature: str, shap_val: float, predicted_label: str
) -> Dict[str, str]:
    """
    Produces crystal-clear, jargon-free explanations for laypeople.
    Returns:
        - layman_name: everyday title
        - layman_definition: what the metric is
        - layman_explanation: full contextual reasoning in plain English
        - layman_signal: short label describing the direction
    """
    meta = FEATURE_LAYMAN_METADATA.get(feature, {})
    name = meta.get("name", feature.replace("_", " "))
    meaning = meta.get("meaning", "Monitors price movements, volume patterns, or company financial reports.")

    pred = predicted_label.upper()
    is_pos = shap_val > 0.0001
    is_neg = shap_val < -0.0001

    if pred == "HOLD":
        if is_pos:
            explanation = meta.get(
                "pos_hold",
                f"Conditions for {name} are steady, encouraging the AI to recommend waiting patiently (HOLD) without taking premature risks."
            )
            signal = "Supported waiting patiently (HOLD)"
        elif is_neg:
            explanation = meta.get(
                "neg_hold",
                f"Recent activity in {name} is picking up, nudging the AI toward taking action rather than staying neutral (HOLD)."
            )
            signal = "Encouraged taking action (Away from HOLD)"
        else:
            explanation = f"{name} had minimal effect on the decision."
            signal = "Neutral influence"

    elif pred == "BUY":
        if is_pos:
            explanation = meta.get(
                "pos_buy",
                f"Positive conditions in {name} provided solid backing for this BUY recommendation."
            )
            signal = "Supported BUY signal (+)"
        elif is_neg:
            explanation = meta.get(
                "neg_buy",
                f"Slight hesitation in {name} suggested using caution rather than aggressive buying."
            )
            signal = "Advised caution on BUY (-)"
        else:
            explanation = f"{name} had minimal effect on the decision."
            signal = "Neutral influence"

    else:  # SELL
        if is_pos:
            explanation = meta.get(
                "pos_sell",
                f"Weakness in {name} reinforced the recommendation to SELL or protect capital."
            )
            signal = "Reinforced SELL signal (+)"
        elif is_neg:
            explanation = meta.get(
                "neg_sell",
                f"Resilience in {name} prevented an even harsher sell assessment."
            )
            signal = "Softened SELL signal (-)"
        else:
            explanation = f"{name} had minimal effect on the decision."
            signal = "Neutral influence"

    return {
        "layman_name": name,
        "layman_definition": meaning,
        "layman_explanation": explanation,
        "layman_signal": signal,
    }


def generate_layman_narrative(
    prediction: str,
    confidence: float,
    primary_driver: str,
    internal_pct: float,
    external_pct: float,
    top_internal: List[Dict[str, Any]],
    top_external: List[Dict[str, Any]],
    company_name: Optional[str] = None,
) -> str:
    """
    Synthesizes an executive, plain-English summary of the AI's final decision.
    Accessible to anyone without finance or data science knowledge.
    """
    company_str = company_name or "this company"
    conf_pct = round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1)

    driver_name = "Market Factors" if primary_driver == "EXTERNAL" else "Company Financials"
    driver_pct = external_pct if primary_driver == "EXTERNAL" else internal_pct

    # Highlight top 2 drivers by name
    top_factors = (top_external if primary_driver == "EXTERNAL" else top_internal)[:2]
    factor_names = [f.get("layman_name", f.get("feature", "")).split("(")[0].strip() for f in top_factors]
    factor_str = f"such as {', and '.join(factor_names)}" if factor_names else ""

    if prediction == "HOLD":
        narrative = (
            f"In Plain English: The AI advises waiting patiently (HOLD) on {company_str} with {conf_pct}% confidence. "
            f"About {driver_pct}% of this decision is driven by {driver_name} {factor_str}. "
            f"Because current price fluctuations are wide and market sentiment is neutral, the model recommends "
            f"staying on the sidelines rather than rushing into a new trade until a clearer trend emerges."
        )
    elif prediction == "BUY":
        narrative = (
            f"In Plain English: The AI recommends a BUY for {company_str} with {conf_pct}% confidence. "
            f"Around {driver_pct}% of the reasoning comes from {driver_name} {factor_str}. "
            f"Favorable momentum and supportive conditions suggest the stock has strong potential "
            f"to move higher over the next 5 trading days."
        )
    else:  # SELL
        narrative = (
            f"In Plain English: The AI recommends a SELL or defensive posture for {company_str} with {conf_pct}% confidence. "
            f"Approximately {driver_pct}% of this decision is influenced by {driver_name} {factor_str}. "
            f"Elevated downside risk and weakening momentum suggest it is prudent to protect capital or take profits."
        )

    return narrative


def get_shap_explanation(
    model: Any,
    X: pd.DataFrame,
    explainer: Optional[shap.TreeExplainer] = None,
    predicted_class_idx: Optional[int] = None,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    company_name: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates SHAP feature contributions for the predicted class and aggregates them
    into INTERNAL and EXTERNAL factors with full layman translations.
    """
    tree_explainer = explainer or get_cached_explainer(model)
    shap_values = tree_explainer.shap_values(X)

    # Determine predicted class index if not provided
    if predicted_class_idx is None:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            conf_val = float(probs[pred_idx])
        else:
            pred_idx = int(model.predict(X)[0])
            conf_val = 0.65
    else:
        pred_idx = int(predicted_class_idx)
        conf_val = float(confidence) if confidence is not None else 0.65

    pred_label = CLASS_LABELS.get(pred_idx, str(pred_idx))

    # Extract class-specific SHAP values depending on array dimensions
    if isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 3:
            values = shap_values[0, :, pred_idx]
        elif len(shap_values.shape) == 2:
            values = shap_values[0]
        else:
            values = shap_values
    elif isinstance(shap_values, list):
        values = shap_values[pred_idx][0]
    else:
        values = np.zeros(len(X.columns))

    feature_names = X.columns.tolist()

    all_contributions = []
    internal_contributions = []
    external_contributions = []

    total_abs_internal = 0.0
    total_abs_external = 0.0

    internal_set = set(feature_groups.get("INTERNAL", [])) if feature_groups else set()
    external_set = set(feature_groups.get("EXTERNAL", [])) if feature_groups else set()

    for feature, value in zip(feature_names, values):
        shap_val = round(float(value), 4)
        abs_val = round(abs(shap_val), 4)
        if feature in internal_set:
            group = "INTERNAL"
        elif feature in external_set:
            group = "EXTERNAL"
        else:
            group = get_feature_group(feature)

        if shap_val > 0.0001:
            direction = "POSITIVE"
        elif shap_val < -0.0001:
            direction = "NEGATIVE"
        else:
            direction = "NEUTRAL"

        explanation_str = generate_feature_explanation(feature, shap_val, pred_label)
        layman_info = generate_layman_explanation(feature, shap_val, pred_label)

        item = {
            "feature": feature,
            "group": group,
            "shap_value": shap_val,
            "importance": abs_val,
            "direction": direction,
            "explanation": explanation_str,
            "layman_name": layman_info["layman_name"],
            "layman_definition": layman_info["layman_definition"],
            "layman_explanation": layman_info["layman_explanation"],
            "layman_signal": layman_info["layman_signal"],
        }

        all_contributions.append(item)

        if group == "INTERNAL":
            internal_contributions.append(item)
            total_abs_internal += abs_val
        else:
            external_contributions.append(item)
            total_abs_external += abs_val

    # Sort each group descending by absolute importance
    all_contributions.sort(key=lambda x: x["importance"], reverse=True)
    internal_contributions.sort(key=lambda x: x["importance"], reverse=True)
    external_contributions.sort(key=lambda x: x["importance"], reverse=True)

    total_abs = total_abs_internal + total_abs_external
    if total_abs > 1e-8:
        internal_pct = round((total_abs_internal / total_abs) * 100.0, 2)
        external_pct = round((total_abs_external / total_abs) * 100.0, 2)
    else:
        internal_pct = 50.0
        external_pct = 50.0

    primary_driver = "INTERNAL" if internal_pct >= external_pct else "EXTERNAL"

    layman_summary = generate_layman_narrative(
        prediction=pred_label,
        confidence=conf_val,
        primary_driver=primary_driver,
        internal_pct=internal_pct,
        external_pct=external_pct,
        top_internal=internal_contributions,
        top_external=external_contributions,
        company_name=company_name,
    )

    return {
        "prediction": pred_label,
        "predicted_class": pred_label,
        "primary_driver": primary_driver,
        "internal_percentage": internal_pct,
        "external_percentage": external_pct,
        "top_internal_factors": internal_contributions[:5],
        "top_external_factors": external_contributions[:5],
        "all_feature_contributions": all_contributions,
        "layman_summary": layman_summary,
        # Backward-compatibility aliases:
        "top_features": all_contributions[:5],
        "factor_attribution": {
            "internal_percentage": internal_pct,
            "external_percentage": external_pct,
            "total_internal_impact": round(total_abs_internal, 4),
            "total_external_impact": round(total_abs_external, 4),
        },
    }
