"""
Company Intelligence AI Chatbot Router for StockDNA-AI.

Provides a domain-specialized, conversational assistant that answers any questions
about the 22 tracked Indian equities, their business models, recent price trends,
earnings results, and active Explainable AI predictions in plain layman English.
"""

from typing import Any, Dict, List, Optional
import re
from fastapi import APIRouter, status
from pydantic import BaseModel

from config.settings import get_company_universe
from backend.predictor import predict
from backend.routers.trends import _get_price_df

router = APIRouter(prefix="/chatbot", tags=["AI Chatbot"])

# Ticker and Alias mapping for all 22 tracked equities
COMPANY_ALIASES: Dict[str, str] = {
    "TCS": "TCS", "TATA CONSULTANCY": "TCS", "TATA CONSULTANCY SERVICES": "TCS",
    "INFY": "INFY", "INFOSYS": "INFY",
    "RELIANCE": "RELIANCE", "RIL": "RELIANCE", "MUKESH AMBANI": "RELIANCE", "JIO": "RELIANCE",
    "WIPRO": "WIPRO", "AZIM PREMJI": "WIPRO",
    "HCLTECH": "HCLTECH", "HCL": "HCLTECH", "HCL TECHNOLOGIES": "HCLTECH",
    "HDFCBANK": "HDFCBANK", "HDFC": "HDFCBANK", "HDFC BANK": "HDFCBANK",
    "ICICIBANK": "ICICIBANK", "ICICI": "ICICIBANK", "ICICI BANK": "ICICIBANK",
    "HINDUNILVR": "HINDUNILVR", "HUL": "HINDUNILVR", "HINDUSTAN UNILEVER": "HINDUNILVR", "UNILEVER": "HINDUNILVR",
    "ITC": "ITC", "ITC LIMITED": "ITC",
    "NESTLEIND": "NESTLEIND", "NESTLE": "NESTLEIND", "MAGGI": "NESTLEIND",
    "MARUTI": "MARUTI", "MARUTI SUZUKI": "MARUTI", "SUZUKI": "MARUTI",
    "M&M": "M&M", "MAHINDRA": "M&M", "MAHINDRA & MAHINDRA": "M&M", "MM": "M&M",
    "SUNPHARMA": "SUNPHARMA", "SUN PHARMA": "SUNPHARMA", "SUN PHARMACEUTICALS": "SUNPHARMA",
    "DRREDDY": "DRREDDY", "DR REDDY": "DRREDDY", "DR REDDYS": "DRREDDY", "DOCTOR REDDY": "DRREDDY",
    "TATASTEEL": "TATASTEEL", "TATA STEEL": "TATASTEEL",
    "ONGC": "ONGC", "OIL AND NATURAL GAS": "ONGC",
    "NTPC": "NTPC", "NATIONAL THERMAL POWER": "NTPC",
    "POWERGRID": "POWERGRID", "POWER GRID": "POWERGRID", "POWER GRID CORPORATION": "POWERGRID",
    "LT": "LT", "LARSEN": "LT", "LARSEN & TOUBRO": "LT", "L&T": "LT",
    "ULTRACEMCO": "ULTRACEMCO", "ULTRATECH": "ULTRACEMCO", "ULTRATECH CEMENT": "ULTRACEMCO",
    "BHARTIARTL": "BHARTIARTL", "AIRTEL": "BHARTIARTL", "BHARTI AIRTEL": "BHARTIARTL", "BHARTI": "BHARTIARTL",
    "ASIANPAINT": "ASIANPAINT", "ASIAN PAINTS": "ASIANPAINT", "ASIAN PAINT": "ASIANPAINT",
}

# Rich Company Profiles Knowledge Base
COMPANY_PROFILES: Dict[str, Dict[str, Any]] = {
    "TCS": {
        "name": "Tata Consultancy Services (TCS)",
        "sector": "Information Technology",
        "business": "Global IT services, consulting, cloud infrastructure, AI platforms, and enterprise software.",
        "strengths": "India's highest market-cap IT company, premier blue-chip balance sheet, over 600,000 employees globally, and long-term Fortune 500 enterprise relationships.",
        "risks": "Global enterprise tech spending slowdowns, US/Europe macroeconomic caution, and wage inflation.",
    },
    "INFY": {
        "name": "Infosys",
        "sector": "Information Technology",
        "business": "Next-generation digital consulting, cloud migration (Cobalt), enterprise AI (Topaz), and digital transformation.",
        "strengths": "Pioneering Indian IT giant, strong dividend track record, high operating margins, and leading digital enterprise revenue share.",
        "risks": "Client discretionary tech budget delays, attrition in specialized AI roles, and foreign exchange fluctuations.",
    },
    "RELIANCE": {
        "name": "Reliance Industries (RIL)",
        "sector": "Energy & Diversified Conglomerate",
        "business": "Hydrocarbon refining & petrochemicals, Jio digital connectivity/5G, Reliance Retail supermarket chains, and renewable green energy gigafactories.",
        "strengths": "Largest private enterprise in India, dominant 5G telecom market share (over 470M subscribers), and massive retail footprint.",
        "risks": "Global crude oil refining margins (GRMs), heavy capital expenditure in green energy, and telecom competition.",
    },
    "WIPRO": {
        "name": "Wipro",
        "sector": "Information Technology",
        "business": "Technology services, cloud modernization, cybersecurity, engineering R&D, and business process services.",
        "strengths": "Deep global footprint in banking, healthcare, and utility sectors with extensive consulting presence (Capco).",
        "risks": "Turnaround execution challenges under leadership changes and higher sensitivity to consulting demand pullbacks.",
    },
    "HCLTECH": {
        "name": "HCL Technologies",
        "sector": "Information Technology",
        "business": "Digital engineering, IT infrastructure management, software products, and engineering R&D services.",
        "strengths": "Superior software IP portfolio, industry-leading growth in engineering services, and consistent deal booking momentum.",
        "risks": "Client renewals concentration and slower product license renewals in subdued economic cycles.",
    },
    "HDFCBANK": {
        "name": "HDFC Bank",
        "sector": "Banking & Financial Services",
        "business": "Retail banking, corporate banking, credit cards, mortgages, auto loans, and treasury operations.",
        "strengths": "India's largest private bank following its mega-merger with parent HDFC Ltd, impeccable asset quality, and nationwide branch network.",
        "risks": "Deposit mobilization pressure, net interest margin (NIM) compression, and post-merger integration pace.",
    },
    "ICICIBANK": {
        "name": "ICICI Bank",
        "sector": "Banking & Financial Services",
        "business": "Retail lending, SME financing, digital corporate banking (iMobile, Trade Online), and wealth management.",
        "strengths": "Industry-leading return on assets (RoA), pristine asset quality, rapid digital adoption, and superior loan growth.",
        "risks": "Unsecured personal loan delinquencies in wider economy and competitive deposit pricing.",
    },
    "HINDUNILVR": {
        "name": "Hindustan Unilever (HUL)",
        "sector": "Fast Moving Consumer Goods (FMCG)",
        "business": "Household cleaning (Surf Excel, Rin), beauty & personal care (Dove, Sunsilk, Lux), and packaged foods (Horlicks, Knorr, Brooke Bond).",
        "strengths": "Unmatched FMCG distribution reach touching 9 out of 10 Indian households, formidable brand loyalty, and consistent cash flows.",
        "risks": "Rural consumption slowdowns, intense local unorganized competition, and raw material palm oil price inflation.",
    },
    "ITC": {
        "name": "ITC Limited",
        "sector": "FMCG & Conglomerates",
        "business": "Cigarettes, packaged foods (Aashirvaad, Sunfeast, Bingo), hotels & hospitality, paperboards, and agribusiness.",
        "strengths": "Near-monopoly in legal cigarettes generating enormous free cash flow, debt-free balance sheet, and strong dividend yields.",
        "risks": "Government tax/excise hikes on tobacco products and capital intensity in paperboards and hotels.",
    },
    "NESTLEIND": {
        "name": "Nestle India",
        "sector": "Fast Moving Consumer Goods (FMCG)",
        "business": "Packaged food and beverages: Maggi noodles, Nescafe coffee, KitKat chocolates, Everyday dairy, and Cerelac infant nutrition.",
        "strengths": "Dominant market shares in instant noodles, milk powders, and infant nutrition; premium brand pricing power.",
        "risks": "High agricultural commodity input costs (cocoa, coffee, milk) and regulatory scrutiny on packaged food sugar contents.",
    },
    "MARUTI": {
        "name": "Maruti Suzuki",
        "sector": "Automotive",
        "business": "Passenger vehicles: compact hatchbacks (Swift, Baleno), SUVs (Brezza, Grand Vitara), and smart hybrid powertrains.",
        "strengths": "Over 40% passenger vehicle market share in India, largest dealership & service network across Tier 2/3 cities, and high fuel-efficiency reputation.",
        "risks": "Slower electric vehicle (EV) transition and intense competition from domestic SUV rivals.",
    },
    "M&M": {
        "name": "Mahindra & Mahindra",
        "sector": "Automotive & Farm Equipment",
        "business": "Rugged utility vehicles and SUVs (Scorpio-N, XUV700, Thar), commercial electric 3-wheelers, and world's largest tractor manufacturer by volume.",
        "strengths": "Dominant leadership in Indian SUV segment and rural farm equipment; aggressive electric SUV product pipeline.",
        "risks": "Rural monsoon dependence for farm equipment and supply chain chip constraints on high-tech SUV variants.",
    },
    "SUNPHARMA": {
        "name": "Sun Pharmaceutical Industries",
        "sector": "Pharmaceuticals & Healthcare",
        "business": "Specialty generic medications, dermatological therapeutics, ophthalmics, and active pharmaceutical ingredients (APIs).",
        "strengths": "India's largest pharmaceutical company, high-margin US specialty product portfolio (Ilumya, Cequa), and global emerging market footprint.",
        "risks": "US FDA regulatory inspection audits (Form 483s), price erosion in generic drugs, and patent litigation.",
    },
    "DRREDDY": {
        "name": "Dr. Reddy's Laboratories",
        "sector": "Pharmaceuticals & Healthcare",
        "business": "Generic pharmaceuticals, biosimilars, oncology therapies, and contract development and manufacturing (CDMO).",
        "strengths": "First-to-file generic drug capabilities in the US, robust presence in Russia & CIS markets, and growing biosimilars franchise.",
        "risks": "Lumpy revenues from key generic launches (like generic Revlimid) and regulatory compliance hurdles.",
    },
    "TATASTEEL": {
        "name": "Tata Steel",
        "sector": "Metals & Mining",
        "business": "End-to-end steel production: hot-rolled coils, automotive grade steel, structural steel, and downstream construction steel products.",
        "strengths": "Low-cost captive iron ore mines in India, prime supplier to Indian automotive & infrastructure sectors, and visionary Tata governance.",
        "risks": "Global commodity steel price cycles, Chinese steel export dumping, and green energy transition capex at European plants (UK/Netherlands).",
    },
    "ONGC": {
        "name": "Oil and Natural Gas Corporation (ONGC)",
        "sector": "Energy (Oil & Gas)",
        "business": "Exploration, offshore drilling, and production of crude oil and natural gas across India's domestic basins.",
        "strengths": "Produces over 70% of India's domestic crude oil and natural gas; major beneficiary of sovereign energy security mandates.",
        "risks": "Government windfall profit taxes, global crude price volatility (Brent), and depleting mature oil fields requiring deepwater capex.",
    },
    "NTPC": {
        "name": "NTPC Limited",
        "sector": "Power Generation",
        "business": "Coal-fired thermal power generation, hydro-electric plants, and aggressive solar & wind renewable energy installations.",
        "strengths": "Generates 25% of India's entire electricity, guaranteed regulated returns on invested equity (CERC tariff model), and massive green hydrogen pivot.",
        "risks": "Coal supply logistics, environmental transition pressures, and discom billing payment delays.",
    },
    "POWERGRID": {
        "name": "Power Grid Corporation of India",
        "sector": "Power Transmission & Utilities",
        "business": "Planning, execution, and operation of India's national interstate electricity transmission grid and telecom optical fiber networks.",
        "strengths": "Monopoly status carrying over 85% of India's interstate power transmission, virtually zero commercial risk due to regulated return tariffs, and stellar dividend yields.",
        "risks": "Right-of-way land acquisition challenges and slow tariff-based competitive bidding (TBCB) project execution.",
    },
    "LT": {
        "name": "Larsen & Toubro (L&T)",
        "sector": "Infrastructure & Engineering",
        "business": "Mega civil engineering, defense equipment, heavy industrial infrastructure, metro railways, and hydrocarbon EPC projects.",
        "strengths": "India's undisputed infrastructure champion with an all-time record order book of over ₹4.5 lakh crore spanning India and the Middle East.",
        "risks": "Working capital bloat on fixed-price government contracts, raw material steel/cement inflation, and geopolitical issues in the Gulf.",
    },
    "ULTRACEMCO": {
        "name": "UltraTech Cement",
        "sector": "Cement & Building Materials",
        "business": "Grey cement, white cement, ready-mix concrete (RMC), and advanced building solutions.",
        "strengths": "India's largest cement producer with over 150 MTPA capacity, nationwide logistics network, and superior pricing power.",
        "risks": "High energy (petcoke, coal, diesel) transportation costs and regional pricing wars during monsoon construction lulls.",
    },
    "BHARTIARTL": {
        "name": "Bharti Airtel",
        "sector": "Telecommunications",
        "business": "5G mobile voice & data, Airtel Xstream broadband fiber, DTH digital television, and Airtel Business enterprise cloud connectivity.",
        "strengths": "Industry-leading Average Revenue Per User (ARPU > ₹200), high-value premium subscriber base, and high-growth African operations.",
        "risks": "Heavy telecom spectrum purchase costs, intense tariff competition from Jio, and statutory AGR dues.",
    },
    "ASIANPAINT": {
        "name": "Asian Paints",
        "sector": "Consumer & Home Decor",
        "business": "Decorative architectural coatings, automotive paints, waterproof sealants, modular kitchens, and bath fittings.",
        "strengths": "Dominant market leader with over 50% market share in decorative paints, legendary supply chain delivering directly to 70,000+ dealers twice daily.",
        "risks": "Entry of well-capitalized new conglomerates (e.g. Grasim Birla Opus paints) and crude oil solvent raw material costs.",
    },
}


class ChatRequest(BaseModel):
    message: str
    context_ticker: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    detected_ticker: Optional[str] = None
    detected_intent: str
    suggested_questions: List[str]


def _extract_tickers(text: str) -> List[str]:
    """Finds all mentioned stock tickers or aliases in the input text."""
    found = []
    text_upper = text.upper()

    # Exact word boundary matching for aliases
    for alias, ticker in COMPANY_ALIASES.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, text_upper):
            if ticker not in found:
                found.append(ticker)

    return found


def _get_stock_snapshot(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetches real-time price snapshot from internal dataset."""
    try:
        df = _get_price_df(ticker)
        if df.empty:
            return None
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        month_ago = df.iloc[-22] if len(df) >= 22 else df.iloc[0]

        latest_close = float(latest["Close"])
        prev_close = float(prev["Close"])
        month_close = float(month_ago["Close"])

        change_1d = round(latest_close - prev_close, 2)
        change_pct_1d = round(((latest_close - prev_close) / prev_close) * 100, 2) if prev_close else 0.0
        change_pct_1m = round(((latest_close - month_close) / month_close) * 100, 2) if month_close else 0.0

        tail_252 = df.tail(252)
        high_52w = round(float(tail_252["High"].max()), 2)
        low_52w = round(float(tail_252["Low"].min()), 2)

        sma20 = round(float(latest["SMA20"]), 2)
        sma50 = round(float(latest["SMA50"]), 2)

        return {
            "price": round(latest_close, 2),
            "change_1d": change_1d,
            "change_pct_1d": change_pct_1d,
            "change_pct_1m": change_pct_1m,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "sma20": sma20,
            "sma50": sma50,
            "date": latest["Date"].strftime("%d %b %Y"),
        }
    except Exception:
        return None


@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Interact with the 22-company financial intelligence chatbot",
)
def handle_chat_message(req: ChatRequest) -> ChatResponse:
    """
    Processes user query about any of the 22 companies, predictions, price trends, or financial concepts.
    """
    msg = req.message.strip()
    msg_lower = msg.lower()

    tickers = _extract_tickers(msg)
    if not tickers and req.context_ticker:
        context_clean = req.context_ticker.replace(".NS", "").strip().upper()
        if context_clean in COMPANY_PROFILES:
            tickers = [context_clean]

    detected_ticker = tickers[0] if tickers else None

    # Intent Detection
    is_compare = len(tickers) >= 2 or ("compare" in msg_lower) or ("vs" in msg_lower and len(tickers) >= 1)
    is_prediction = any(w in msg_lower for w in ["predict", "prediction", "buy", "sell", "hold", "rating", "recommend", "target", "forecast", "xai", "shap"])
    is_price_trend = any(w in msg_lower for w in ["price", "trend", "chart", "performance", "52 week", "high", "low", "moving average", "sma", "volume"])
    is_layman_explain = any(w in msg_lower for w in ["layman", "simple words", "simple terms", "explain", "why", "easy", "child", "eli5", "what is"])
    is_universe_scan = any(w in msg_lower for w in ["list", "all stocks", "which stocks", "sectors", "companies", "tracked", "how many"])

    # -------------------------------------------------------------------------
    # CASE 1: Comparison between two or more companies
    # -------------------------------------------------------------------------
    if is_compare and len(tickers) >= 2:
        t1, t2 = tickers[0], tickers[1]
        p1 = COMPANY_PROFILES.get(t1, {})
        p2 = COMPANY_PROFILES.get(t2, {})
        s1 = _get_stock_snapshot(t1)
        s2 = _get_stock_snapshot(t2)

        # Run live model predictions for both
        pred1_str, pred2_str = "HOLD", "HOLD"
        conf1, conf2 = 55.0, 55.0
        try:
            r1 = predict(t1)
            pred1_str = r1["prediction"]
            conf1 = round(r1["confidence"] * 100, 1)
        except Exception:
            pass

        try:
            r2 = predict(t2)
            pred2_str = r2["prediction"]
            conf2 = round(r2["confidence"] * 100, 1)
        except Exception:
            pass

        reply = f"""### ⚖️ Side-by-Side Comparison: **{p1.get('name', t1)}** vs **{p2.get('name', t2)}**

Here is how both companies stack up across current pricing, recent market momentum, and our AI prediction engine:

| Metric | **{t1}** | **{t2}** |
| :--- | :--- | :--- |
| **Sector** | {p1.get('sector', 'N/A')} | {p2.get('sector', 'N/A')} |
| **Current Share Price** | ₹{s1['price'] if s1 else 'N/A'} | ₹{s2['price'] if s2 else 'N/A'} |
| **1-Day Price Change** | {s1['change_pct_1d'] if s1 else 0}% | {s2['change_pct_1d'] if s2 else 0}% |
| **1-Month Return** | {s1['change_pct_1m'] if s1 else 0}% | {s2['change_pct_1m'] if s2 else 0}% |
| **52-Week High / Low** | ₹{s1['high_52w'] if s1 else 0} / ₹{s1['low_52w'] if s1 else 0} | ₹{s2['high_52w'] if s2 else 0} / ₹{s2['low_52w'] if s2 else 0} |
| **AI 5-Day Prediction** | **`{pred1_str}`** ({conf1}% confidence) | **`{pred2_str}`** ({conf2}% confidence) |

---
**Key Business Takeaways:**
* **{t1}:** {p1.get('strengths', '')}
* **{t2}:** {p2.get('strengths', '')}

**In Simple Words:** Both are industry leaders. If you are looking for short-term directional movement, our AI currently assigns **{pred1_str}** to {t1} and **{pred2_str}** to {t2}.
"""
        return ChatResponse(
            reply=reply,
            detected_ticker=t1,
            detected_intent="comparison",
            suggested_questions=[
                f"Why does the AI predict {pred1_str} for {t1} in plain English?",
                f"Show price trend chart for {t1}",
                f"Show price trend chart for {t2}",
                f"What are the main risks for {t1}?",
            ]
        )

    # -------------------------------------------------------------------------
    # CASE 2: Single Company - AI Prediction & XAI
    # -------------------------------------------------------------------------
    if detected_ticker and (is_prediction or (is_layman_explain and "why" in msg_lower)):
        profile = COMPANY_PROFILES.get(detected_ticker, {})
        snapshot = _get_stock_snapshot(detected_ticker)

        try:
            pred_res = predict(detected_ticker)
            prediction = pred_res["prediction"]
            confidence = round(pred_res["confidence"] * 100, 1)
            probs = pred_res.get("probabilities", {})
            prob_buy = round(probs.get("BUY", 0.0) * 100, 1)
            prob_hold = round(probs.get("HOLD", 0.0) * 100, 1)
            prob_sell = round(probs.get("SELL", 0.0) * 100, 1)

            xai = pred_res.get("xai", {})
            primary_driver = xai.get("primary_driver", "EXTERNAL")
            internal_pct = xai.get("internal_percentage", 20.0)
            external_pct = xai.get("external_percentage", 80.0)
            layman_summary = pred_res.get("layman_summary") or xai.get("layman_summary", "")

            # Format top factors
            top_ext = xai.get("top_external_factors", [])[:2]
            top_int = xai.get("top_internal_factors", [])[:2]

            factors_md = ""
            for f in top_ext:
                factors_md += f"* 🌐 **{f.get('layman_name', f.get('feature'))}:** {f.get('layman_explanation', f.get('explanation'))}\n"
            for f in top_int:
                factors_md += f"* 🏢 **{f.get('layman_name', f.get('feature'))}:** {f.get('layman_explanation', f.get('explanation'))}\n"

            reply = f"""### 🤖 AI Prediction for **{profile.get('name', detected_ticker)}** (`{detected_ticker}`)

**Final Verdict:** **`{prediction}`** with **{confidence}% Confidence**

* 🟢 **BUY Probability:** {prob_buy}%
* 🟡 **HOLD Probability:** {prob_hold}%
* 🔴 **SELL Probability:** {prob_sell}%

---

#### 💡 In Plain English: Why Did the Model Choose {prediction}?
{layman_summary}

**Decision Breakdown:**
* **{external_pct}%** of the reasoning came from **Market Factors** (overall market fear, recent price swings, trading volume).
* **{internal_pct}%** of the reasoning came from **Company Fundamentals** (quarterly earnings surprises, profitability vs share price).

#### 🔍 Top Influencing Clues:
{factors_md}

*Current Market Price:* ₹{snapshot['price'] if snapshot else 'N/A'} (as of {snapshot['date'] if snapshot else 'Latest'}).
"""
            return ChatResponse(
                reply=reply,
                detected_ticker=detected_ticker,
                detected_intent="prediction",
                suggested_questions=[
                    f"Show me the price trend chart for {detected_ticker}",
                    f"What does {profile.get('name', detected_ticker)} actually do?",
                    f"Explain what India VIX means in simple words",
                    f"Compare {detected_ticker} with another stock in its sector",
                ]
            )
        except Exception as e:
            # Fallback if prediction error occurs
            pass

    # -------------------------------------------------------------------------
    # CASE 3: Single Company - Price Trends & Performance
    # -------------------------------------------------------------------------
    if detected_ticker and is_price_trend:
        profile = COMPANY_PROFILES.get(detected_ticker, {})
        s = _get_stock_snapshot(detected_ticker)

        if s:
            price_trend_text = (
                "bullish and trending above moving averages"
                if s["price"] >= s["sma50"]
                else "consolidating or under short-term pressure below its 50-day average"
            )

            reply = f"""### 📈 Price Trends & Technical Snapshot: **{profile.get('name', detected_ticker)}**

* **Current Close Price:** **₹{s['price']}** ({s['date']})
* **1-Day Change:** {s['change_1d']:+} ( **{s['change_pct_1d']:+}%** )
* **Past Month Return:** **{s['change_pct_1m']:+}%**
* **52-Week High:** ₹{s['high_52w']} *(Price is currently {round(((s['price'] - s['high_52w']) / s['high_52w']) * 100, 1)}% from all-time peak)*
* **52-Week Low:** ₹{s['low_52w']} *(Price is currently {round(((s['price'] - s['low_52w']) / s['low_52w']) * 100, 1)}% above lowest point)*
* **20-Day Moving Average (1-Month Average):** ₹{s['sma20']}
* **50-Day Moving Average (2-Month Baseline):** ₹{s['sma50']}

**Trend Verdict in Plain Words:**
The stock is currently **{price_trend_text}**. 

> **Interactive Chart Tip:** You can inspect interactive interactive price graphs, candlestick trends, and volume bars on our **Price Trends** page!
"""
            return ChatResponse(
                reply=reply,
                detected_ticker=detected_ticker,
                detected_intent="price_trend",
                suggested_questions=[
                    f"What is the AI prediction for {detected_ticker}?",
                    f"Tell me about {detected_ticker}'s core business and strengths",
                    f"What is the 52-week high for {detected_ticker}?",
                    f"Compare {detected_ticker} with its main competitor",
                ]
            )

    # -------------------------------------------------------------------------
    # CASE 4: Single Company - Business Overview & Fundamentals
    # -------------------------------------------------------------------------
    if detected_ticker:
        profile = COMPANY_PROFILES.get(detected_ticker, {})
        snapshot = _get_stock_snapshot(detected_ticker)

        reply = f"""### 🏢 Company Profile: **{profile.get('name', detected_ticker)}** (`{detected_ticker}`)

**Sector:** {profile.get('sector', 'N/A')}  
**Current Share Price:** ₹{snapshot['price'] if snapshot else 'N/A'}  

#### 💼 What Does the Company Do?
{profile.get('business', 'A prominent publicly traded enterprise in India.')}

#### 🌟 Key Competitive Strengths:
{profile.get('strengths', 'Strong brand recognition and nationwide footprint.')}

#### ⚠️ Key Risk Factors to Watch:
{profile.get('risks', 'Market volatility and industry competitive dynamics.')}

---
*Would you like to see our AI's 5-day directional prediction or interactive price chart for {detected_ticker}?*
"""
        return ChatResponse(
            reply=reply,
            detected_ticker=detected_ticker,
            detected_intent="company_profile",
            suggested_questions=[
                f"What is the AI prediction for {detected_ticker}?",
                f"Show recent price trends and moving averages for {detected_ticker}",
                f"Why would the AI say HOLD or BUY for {detected_ticker}?",
                f"Compare {detected_ticker} with another company",
            ]
        )

    # -------------------------------------------------------------------------
    # CASE 5: Universe Scanning & Stock Lists
    # -------------------------------------------------------------------------
    if is_universe_scan or "track" in msg_lower or "which" in msg_lower:
        universe = get_company_universe()
        sectors: Dict[str, List[str]] = {}
        for _, row in universe.iterrows():
            sec = row["sector"]
            name_str = f"{row['clean_ticker']} ({row['company']})"
            sectors.setdefault(sec, []).append(name_str)

        lines = []
        for sec, items in sorted(sectors.items()):
            lines.append(f"* **{sec}:** {', '.join(items)}")

        reply = f"""### 🌐 StockDNA-AI 22-Stock Universe

We track **22 premier Indian equities** across **11 major economic sectors**:

{chr(10).join(lines)}

---
**What you can do:**
* Ask me: *"Tell me about Reliance"* or *"What does TCS do?"*
* Ask for predictions: *"What is the prediction for HDFC Bank?"*
* Compare stocks: *"Compare TCS vs Infosys"* or *"Compare Maruti vs M&M"*
* Explore technicals: *"Show price trends for Tata Steel"*
"""
        return ChatResponse(
            reply=reply,
            detected_ticker=None,
            detected_intent="universe_list",
            suggested_questions=[
                "What is the AI prediction for Reliance?",
                "Compare HDFC Bank and ICICI Bank",
                "Tell me about Sun Pharma",
                "Explain how Explainable AI works in simple words",
            ]
        )

    # -------------------------------------------------------------------------
    # CASE 6: Financial Term / Concept Explanations in Plain English
    # -------------------------------------------------------------------------
    if "shap" in msg_lower or "xai" in msg_lower or "explainable" in msg_lower:
        reply = """### 💡 Explainable AI (SHAP) in Plain English

**What is it?**  
Imagine you are applying for a home loan, and a bank manager decides whether to approve it. Instead of just saying "Approved" or "Rejected" without explaining why, the manager gives you a scorecard showing:
* +20 points for your high steady income
* +15 points for a clean credit score
* -10 points for existing credit card debt

**That is exactly what SHAP does for StockDNA-AI!**  
Instead of being a mysterious black box, our model inspects 34 indicators (like daily price swings, company quarterly earnings, trading volume, and market fear) and assigns exact plus or minus scores to each clue.

* **Internal Factors:** Clues coming from the company itself (e.g. quarterly profit beats, profit yields).
* **External Factors:** Clues coming from the outside world (e.g. overall market sentiment, price momentum, trading surges).
"""
        return ChatResponse(
            reply=reply,
            detected_ticker=None,
            detected_intent="concept_explanation",
            suggested_questions=[
                "What is India VIX in simple words?",
                "What does Normalized ATR mean?",
                "What is the prediction for TCS?",
                "Which stocks are currently tracked?",
            ]
        )

    if "vix" in msg_lower:
        reply = """### 🌡️ India VIX (The Market Fear Thermometer)

**In Plain Words:**  
**India VIX** is often called the **"Fear Gauge"** of Dalal Street. 

* **Think of it like a medical thermometer for market panic:**
  * When VIX is **low (below 13–15)**: Investors are calm, content, and expect smooth sailing.
  * When VIX is **high (above 18–22)**: Investors are anxious and fearful, anticipating violent price swings up or down.

**How our AI uses it:**  
When market-wide fear is spiking, our model often leans toward **HOLD** or **SELL** to protect you from unpredictable market turbulence!
"""
        return ChatResponse(
            reply=reply,
            detected_ticker=None,
            detected_intent="concept_explanation",
            suggested_questions=[
                "What is the AI prediction for Reliance?",
                "What does Normalized ATR mean?",
                "Compare TCS and Infosys",
            ]
        )

    if "atr" in msg_lower or "volatility" in msg_lower:
        reply = """### 🌊 Daily Price Volatility (ATR) in Plain English

**What is ATR (Average True Range)?**  
Think of ATR as measuring the **height of the daily waves** in a swimming pool:

* If a stock has **low volatility**: It's like calm bathwater. If the stock is ₹1,000, it might only move up or down ₹5 in a normal day.
* If a stock has **high volatility**: It's like a stormy sea. The stock might whip between ₹950 and ₹1,050 in a single afternoon.

**Why does it matter to our AI?**  
When price swings become unusually erratic, our AI recognizes that the risk of getting caught in a whipsaw is elevated, so it recommends waiting patiently (**HOLD**) until the waters calm down!
"""
        return ChatResponse(
            reply=reply,
            detected_ticker=None,
            detected_intent="concept_explanation",
            suggested_questions=[
                "What does the AI say about Tata Steel?",
                "What is the prediction for Maruti?",
                "Explain what SHAP means",
            ]
        )

    # -------------------------------------------------------------------------
    # CASE 7: General Default Fallback
    # -------------------------------------------------------------------------
    reply = f"""Hello! I am your **StockDNA-AI Financial Assistant**. I have deep intelligence on all **22 premier Indian equities** tracked by our platform.

You can ask me anything about:
1. **Company Profiles & Businesses:** *"What does L&T do?"* or *"Tell me about Reliance"*
2. **AI Directional Predictions:** *"What is the prediction for TCS?"* or *"Why is Tata Steel rated HOLD?"*
3. **Price Trends & Technicals:** *"Show price trends for Infosys"* or *"What is Sun Pharma's 52-week high?"*
4. **Comparisons:** *"Compare TCS vs Infosys"* or *"HDFC Bank vs ICICI Bank"*
5. **Plain English Explanations:** *"What is VIX?"* or *"How does SHAP explainability work?"*

What would you like to explore today?
"""
    return ChatResponse(
        reply=reply,
        detected_ticker=None,
        detected_intent="general_help",
        suggested_questions=[
            "What is the AI prediction for TCS?",
            "Compare TCS vs Infosys",
            "Tell me about Reliance Industries",
            "Show price trends for Tata Steel",
        ]
    )

