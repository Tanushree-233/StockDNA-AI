import React, { useState, useEffect, useMemo } from 'react';
import { api } from '../services/api';

export default function Trends({ initialTicker = 'TCS', onNavigate }) {
  const [ticker, setTicker] = useState(initialTicker);
  const [range, setRange] = useState('6M');
  const [trendsData, setTrendsData] = useState(null);
  const [summaryList, setSummaryList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSector, setSelectedSector] = useState('ALL');
  const [showSMA20, setShowSMA20] = useState(true);
  const [showSMA50, setShowSMA50] = useState(true);
  const [showVolume, setShowVolume] = useState(true);
  const [hoverIndex, setHoverIndex] = useState(null);

  // Fetch all 22 stocks summary on mount
  useEffect(() => {
    api
      .getTrendsSummary()
      .then((data) => {
        if (Array.isArray(data)) {
          setSummaryList(data);
        }
      })
      .catch((err) => console.error('Failed to load trends summary:', err));
  }, []);

  // Fetch detailed time-series trends whenever ticker or range changes
  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    api
      .getTickerTrends(ticker, range)
      .then((res) => {
        if (isMounted) {
          setTrendsData(res);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || `Failed to load chart data for ${ticker}`);
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [ticker, range]);

  // Extract unique sectors
  const sectors = useMemo(() => {
    const set = new Set(summaryList.map((s) => s.sector));
    return ['ALL', ...Array.from(set).sort()];
  }, [summaryList]);

  // Filtered summary list for search
  const filteredStocks = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    return summaryList.filter((s) => {
      const matchSector = selectedSector === 'ALL' || s.sector === selectedSector;
      const matchQuery =
        !q ||
        s.ticker.toLowerCase().includes(q) ||
        s.company.toLowerCase().includes(q) ||
        s.sector.toLowerCase().includes(q);
      return matchSector && matchQuery;
    });
  }, [summaryList, searchQuery, selectedSector]);

  // Chart coordinate computations
  const series = trendsData?.series || [];
  const pointsCount = series.length;

  const chartBounds = useMemo(() => {
    if (!series.length) return { minPrice: 0, maxPrice: 100, maxVol: 100 };

    let minP = Infinity;
    let maxP = -Infinity;
    let maxV = 0;

    series.forEach((d) => {
      if (d.low < minP) minP = d.low;
      if (d.high > maxP) maxP = d.high;
      if (d.sma20 && d.sma20 < minP) minP = d.sma20;
      if (d.sma20 && d.sma20 > maxP) maxP = d.sma20;
      if (d.sma50 && d.sma50 < minP) minP = d.sma50;
      if (d.sma50 && d.sma50 > maxP) maxP = d.sma50;
      if (d.volume > maxV) maxV = d.volume;
    });

    const padding = (maxP - minP) * 0.05 || 10;
    return {
      minPrice: Math.floor(minP - padding),
      maxPrice: Math.ceil(maxP + padding),
      maxVol: maxV || 1,
    };
  }, [series]);

  // SVG dimensions
  const svgWidth = 850;
  const priceChartHeight = 320;
  const volChartHeight = showVolume ? 90 : 0;
  const totalSvgHeight = priceChartHeight + (showVolume ? volChartHeight + 20 : 0);

  const getX = (idx) => {
    if (pointsCount <= 1) return 40;
    return 40 + (idx / (pointsCount - 1)) * (svgWidth - 90);
  };

  const getY = (val) => {
    const { minPrice, maxPrice } = chartBounds;
    if (maxPrice <= minPrice) return priceChartHeight / 2;
    const ratio = (val - minPrice) / (maxPrice - minPrice);
    return priceChartHeight - 20 - ratio * (priceChartHeight - 40);
  };

  const getVolY = (vol) => {
    const { maxVol } = chartBounds;
    const ratio = Math.min(1, vol / maxVol);
    const startY = priceChartHeight + 20;
    return startY + volChartHeight - ratio * volChartHeight;
  };

  // Generate SVG path strings
  const pricePath = useMemo(() => {
    if (!series.length) return '';
    return series.reduce((acc, pt, i) => {
      const x = getX(i);
      const y = getY(pt.close);
      return i === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`;
    }, '');
  }, [series, chartBounds]);

  const areaPath = useMemo(() => {
    if (!series.length) return '';
    const firstX = getX(0);
    const lastX = getX(series.length - 1);
    const bottomY = priceChartHeight - 10;
    return `${pricePath} L ${lastX} ${bottomY} L ${firstX} ${bottomY} Z`;
  }, [pricePath, series]);

  const sma20Path = useMemo(() => {
    if (!showSMA20 || !series.length) return '';
    let p = '';
    series.forEach((pt, i) => {
      if (pt.sma20) {
        const x = getX(i);
        const y = getY(pt.sma20);
        p = p ? `${p} L ${x} ${y}` : `M ${x} ${y}`;
      }
    });
    return p;
  }, [series, showSMA20, chartBounds]);

  const sma50Path = useMemo(() => {
    if (!showSMA50 || !series.length) return '';
    let p = '';
    series.forEach((pt, i) => {
      if (pt.sma50) {
        const x = getX(i);
        const y = getY(pt.sma50);
        p = p ? `${p} L ${x} ${y}` : `M ${x} ${y}`;
      }
    });
    return p;
  }, [series, showSMA50, chartBounds]);

  const activePoint = hoverIndex !== null && series[hoverIndex] ? series[hoverIndex] : series[series.length - 1];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Page Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.35rem' }}>
            <span className="badge badge-brand">Technical Analysis</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Historical OHLCV & Moving Averages</span>
          </div>
          <h2 style={{ fontSize: '2rem' }}>Company Price Trends & Interactive Charts</h2>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
            Search and track historical price action, moving averages (SMA 20/50), volume spikes, and trend momentum across all 22 tracked Indian equities.
          </p>
        </div>

        {/* Quick Action Buttons */}
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => onNavigate && onNavigate('chatbot', { ticker })}
            style={{ fontSize: '0.85rem', padding: '0.5rem 1rem' }}
          >
            💬 Ask Chatbot about {ticker}
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => onNavigate && onNavigate('predict', { ticker })}
            style={{ fontSize: '0.85rem', padding: '0.5rem 1rem' }}
          >
            🤖 Run AI Prediction on {ticker} &rarr;
          </button>
        </div>
      </div>

      {/* Top Search & Filter Bar */}
      <div
        className="glass-panel"
        style={{
          padding: '1.25rem 1.5rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
        }}
      >
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
          {/* Search Input */}
          <div style={{ flex: '1 1 280px', position: 'relative' }}>
            <input
              type="text"
              className="form-control"
              placeholder="Search by company name (e.g. Infosys, Reliance, Tata) or ticker (e.g. INFY)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                width: '100%',
                padding: '0.65rem 1rem',
                fontSize: '0.9rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid var(--border-subtle)',
                color: 'white',
              }}
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                style={{
                  position: 'absolute',
                  right: '12px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                }}
              >
                ✕
              </button>
            )}
          </div>

          {/* Direct Dropdown Selection */}
          <div style={{ width: '260px' }}>
            <select
              className="form-select"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              style={{ width: '100%', padding: '0.65rem', fontSize: '0.88rem' }}
            >
              {summaryList.map((s) => (
                <option key={s.ticker} value={s.ticker}>
                  {s.ticker} — {s.company} ({s.sector})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Sector Filter Chips */}
        <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginRight: '0.3rem' }}>
            Filter Sector:
          </span>
          {sectors.map((sec) => (
            <button
              key={sec}
              type="button"
              onClick={() => setSelectedSector(sec)}
              style={{
                background: selectedSector === sec ? 'var(--color-brand)' : 'rgba(255, 255, 255, 0.04)',
                color: selectedSector === sec ? 'white' : 'var(--text-secondary)',
                border: selectedSector === sec ? '1px solid var(--color-brand)' : '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: '16px',
                padding: '0.25rem 0.75rem',
                fontSize: '0.75rem',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {sec}
            </button>
          ))}
        </div>
      </div>

      {/* Main Chart Card */}
      <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {/* Chart Header & Active Info */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <h3 style={{ fontSize: '1.6rem', fontWeight: '800', fontFamily: 'var(--font-display)', margin: 0 }}>
                {trendsData?.ticker || ticker}
              </h3>
              <span style={{ fontSize: '1rem', color: 'var(--text-secondary)' }}>
                {trendsData?.company}
              </span>
              <span className="badge badge-brand">{trendsData?.sector}</span>
            </div>

            {/* Current Price & Change */}
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '1rem', marginTop: '0.4rem' }}>
              <span style={{ fontSize: '2rem', fontWeight: '800', fontFamily: 'var(--font-mono)' }}>
                ₹{(activePoint?.close || trendsData?.current_price || 0).toLocaleString('en-IN', { minimumFractionDigits: 2 })}
              </span>
              <div
                style={{
                  fontSize: '0.95rem',
                  fontWeight: '700',
                  color: (trendsData?.change_pct_1d || 0) >= 0 ? 'var(--color-buy)' : 'var(--color-sell)',
                }}
              >
                {(trendsData?.change_1d || 0) >= 0 ? `+₹${trendsData?.change_1d}` : `-₹${Math.abs(trendsData?.change_1d || 0)}`} ({trendsData?.change_pct_1d >= 0 ? `+${trendsData?.change_pct_1d}%` : `${trendsData?.change_pct_1d}%`})
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginLeft: '0.35rem', fontWeight: '400' }}>Today</span>
              </div>
            </div>
          </div>

          {/* Timeframe Range Selector */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.5rem' }}>
            <div style={{ display: 'flex', gap: '0.35rem', background: 'rgba(255, 255, 255, 0.05)', padding: '0.25rem', borderRadius: 'var(--radius-md)' }}>
              {['1M', '3M', '6M', '1Y', 'ALL'].map((r) => (
                <button
                  key={r}
                  type="button"
                  onClick={() => setRange(r)}
                  style={{
                    background: range === r ? 'var(--color-brand)' : 'transparent',
                    color: range === r ? 'white' : 'var(--text-secondary)',
                    border: 'none',
                    borderRadius: 'var(--radius-sm)',
                    padding: '0.35rem 0.8rem',
                    fontSize: '0.8rem',
                    fontWeight: '700',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {r}
                </button>
              ))}
            </div>

            {/* Indicator Toggles */}
            <div style={{ display: 'flex', gap: '1rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={showSMA20}
                  onChange={(e) => setShowSMA20(e.target.checked)}
                />
                <span style={{ color: 'var(--color-internal)' }}>● 20-Day SMA</span>
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={showSMA50}
                  onChange={(e) => setShowSMA50(e.target.checked)}
                />
                <span style={{ color: 'var(--color-external)' }}>● 50-Day SMA</span>
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={showVolume}
                  onChange={(e) => setShowVolume(e.target.checked)}
                />
                <span>Volume Bars</span>
              </label>
            </div>
          </div>
        </div>

        {/* Hover Point Tooltip Bar */}
        {activePoint && (
          <div
            style={{
              padding: '0.65rem 1rem',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              gap: '1.5rem',
              flexWrap: 'wrap',
              fontSize: '0.82rem',
              color: 'var(--text-secondary)',
            }}
          >
            <span>📅 Date: <strong style={{ color: 'white' }}>{activePoint.date}</strong></span>
            <span>Open: <strong style={{ color: 'white' }}>₹{activePoint.open}</strong></span>
            <span>High: <strong style={{ color: 'var(--color-buy)' }}>₹{activePoint.high}</strong></span>
            <span>Low: <strong style={{ color: 'var(--color-sell)' }}>₹{activePoint.low}</strong></span>
            <span>Close: <strong style={{ color: 'white' }}>₹{activePoint.close}</strong></span>
            {activePoint.sma20 && <span>SMA20: <strong style={{ color: 'var(--color-internal)' }}>₹{activePoint.sma20}</strong></span>}
            {activePoint.sma50 && <span>SMA50: <strong style={{ color: 'var(--color-external)' }}>₹{activePoint.sma50}</strong></span>}
            <span>Volume: <strong style={{ color: 'white' }}>{activePoint.volume?.toLocaleString()}</strong></span>
          </div>
        )}

        {/* Loading and Error States */}
        {loading && (
          <div style={{ height: '320px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '1rem' }}>
            <span className="spinner" style={{ width: '2rem', height: '2rem' }} />
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Loading high-resolution price series...</span>
          </div>
        )}

        {error && (
          <div style={{ padding: '1.5rem', background: 'var(--color-sell-bg)', color: 'var(--color-sell)', borderRadius: 'var(--radius-md)' }}>
            {error}
          </div>
        )}

        {/* Interactive SVG Chart */}
        {!loading && !error && series.length > 0 && (
          <div style={{ width: '100%', overflowX: 'auto' }}>
            <svg
              viewBox={`0 0 ${svgWidth} ${totalSvgHeight}`}
              style={{ width: '100%', minWidth: '700px', height: 'auto', display: 'block' }}
              onMouseLeave={() => setHoverIndex(null)}
            >
              <defs>
                {/* Gradient for area fill */}
                <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--color-brand)" stopOpacity="0.35" />
                  <stop offset="85%" stopColor="var(--color-brand)" stopOpacity="0.0" />
                </linearGradient>

                <linearGradient id="volBuyGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="rgba(34, 197, 94, 0.6)" />
                  <stop offset="100%" stopColor="rgba(34, 197, 94, 0.15)" />
                </linearGradient>

                <linearGradient id="volSellGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="rgba(239, 68, 68, 0.6)" />
                  <stop offset="100%" stopColor="rgba(239, 68, 68, 0.15)" />
                </linearGradient>
              </defs>

              {/* Price Horizontal Grid Lines & Y-Axis Labels */}
              {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
                const p = chartBounds.minPrice + pct * (chartBounds.maxPrice - chartBounds.minPrice);
                const y = getY(p);
                return (
                  <g key={pct}>
                    <line
                      x1={40}
                      y1={y}
                      x2={svgWidth - 50}
                      y2={y}
                      stroke="rgba(255, 255, 255, 0.06)"
                      strokeDasharray="4 4"
                    />
                    <text
                      x={svgWidth - 45}
                      y={y + 4}
                      fill="var(--text-muted)"
                      fontSize="10"
                      fontFamily="var(--font-mono)"
                    >
                      ₹{Math.round(p)}
                    </text>
                  </g>
                );
              })}

              {/* Area Gradient Fill */}
              <path d={areaPath} fill="url(#priceGradient)" />

              {/* 50-Day Moving Average Line (Purple) */}
              {showSMA50 && sma50Path && (
                <path
                  d={sma50Path}
                  fill="none"
                  stroke="var(--color-external)"
                  strokeWidth="2"
                  strokeDasharray="3 3"
                />
              )}

              {/* 20-Day Moving Average Line (Cyan) */}
              {showSMA20 && sma20Path && (
                <path
                  d={sma20Path}
                  fill="none"
                  stroke="var(--color-internal)"
                  strokeWidth="2"
                />
              )}

              {/* Main Price Line (Brand Blue/Purple) */}
              <path
                d={pricePath}
                fill="none"
                stroke="var(--color-brand)"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Volume Bars */}
              {showVolume && (
                <g>
                  {/* Volume baseline */}
                  <line
                    x1={40}
                    y1={priceChartHeight + 20 + volChartHeight}
                    x2={svgWidth - 50}
                    y2={priceChartHeight + 20 + volChartHeight}
                    stroke="rgba(255, 255, 255, 0.1)"
                  />
                  <text
                    x={40}
                    y={priceChartHeight + 16}
                    fill="var(--text-muted)"
                    fontSize="9"
                    fontWeight="700"
                  >
                    DAILY VOLUME
                  </text>

                  {series.map((d, i) => {
                    const x = getX(i);
                    const y = getVolY(d.volume);
                    const barHeight = priceChartHeight + 20 + volChartHeight - y;
                    const isUp = d.close >= d.open;
                    const barWidth = Math.max(2, (svgWidth - 90) / pointsCount - 1.5);

                    return (
                      <rect
                        key={d.date}
                        x={x - barWidth / 2}
                        y={y}
                        width={barWidth}
                        height={Math.max(1, barHeight)}
                        fill={isUp ? 'url(#volBuyGradient)' : 'url(#volSellGradient)'}
                        rx="1"
                      />
                    );
                  })}
                </g>
              )}

              {/* Interactive Hover Vertical Crosshair */}
              {hoverIndex !== null && (
                <g>
                  <line
                    x1={getX(hoverIndex)}
                    y1={10}
                    x2={getX(hoverIndex)}
                    y2={totalSvgHeight - 5}
                    stroke="rgba(255, 255, 255, 0.4)"
                    strokeDasharray="2 2"
                  />
                  <circle
                    cx={getX(hoverIndex)}
                    cy={getY(series[hoverIndex].close)}
                    r="5"
                    fill="var(--color-brand)"
                    stroke="white"
                    strokeWidth="2"
                  />
                </g>
              )}

              {/* Transparent Click & Hover Sensor Slices */}
              {series.map((_, i) => {
                const x = getX(i);
                const sliceW = (svgWidth - 90) / pointsCount;
                return (
                  <rect
                    key={i}
                    x={x - sliceW / 2}
                    y={0}
                    width={sliceW}
                    height={totalSvgHeight}
                    fill="transparent"
                    style={{ cursor: 'crosshair' }}
                    onMouseEnter={() => setHoverIndex(i)}
                  />
                );
              })}
            </svg>
          </div>
        )}

        {/* Technical Summary KPI Cards */}
        {trendsData && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: '1rem',
              borderTop: '1px solid var(--border-subtle)',
              paddingTop: '1.5rem',
            }}
          >
            <div style={{ padding: '0.85rem 1rem', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-md)' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Period Return ({range})</span>
              <div
                style={{
                  fontSize: '1.15rem',
                  fontWeight: '800',
                  color: trendsData.period_change_pct >= 0 ? 'var(--color-buy)' : 'var(--color-sell)',
                  marginTop: '0.2rem',
                }}
              >
                {trendsData.period_change_pct >= 0 ? `+${trendsData.period_change_pct}%` : `${trendsData.period_change_pct}%`}
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                {trendsData.period_change >= 0 ? `+₹${trendsData.period_change}` : `-₹${Math.abs(trendsData.period_change)}`}
              </span>
            </div>

            <div style={{ padding: '0.85rem 1rem', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-md)' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>52-Week High / Low</span>
              <div style={{ fontSize: '1.15rem', fontWeight: '800', marginTop: '0.2rem', color: 'white' }}>
                ₹{trendsData.high_52w}
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Low: ₹{trendsData.low_52w}
              </span>
            </div>

            <div style={{ padding: '0.85rem 1rem', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-md)' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Moving Averages</span>
              <div style={{ fontSize: '0.85rem', color: 'var(--color-internal)', fontWeight: '700', marginTop: '0.2rem' }}>
                20-Day: ₹{trendsData.sma20}
              </div>
              <div style={{ fontSize: '0.85rem', color: 'var(--color-external)', fontWeight: '700' }}>
                50-Day: ₹{trendsData.sma50}
              </div>
            </div>

            <div style={{ padding: '0.85rem 1rem', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-md)' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Trend Posture</span>
              <div style={{ fontSize: '1.05rem', fontWeight: '800', marginTop: '0.2rem', color: 'var(--color-brand)' }}>
                {trendsData.trend_status}
              </div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', display: 'block', marginTop: '0.2rem', lineHeight: '1.3' }}>
                {trendsData.trend_description}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* 22-Stock Watchlist & Search Grid */}
      <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <h3 style={{ fontSize: '1.25rem', margin: 0 }}>All 22 Companies Watchlist & Momentum</h3>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Click any company below to instantly view its charts and indicators
            </span>
          </div>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Showing {filteredStocks.length} of 22 companies
          </span>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
            gap: '1rem',
          }}
        >
          {filteredStocks.map((stock) => {
            const isSelected = stock.ticker === ticker;
            const isPositive = stock.change_pct_1d >= 0;

            return (
              <div
                key={stock.ticker}
                onClick={() => setTicker(stock.ticker)}
                style={{
                  padding: '1rem 1.15rem',
                  borderRadius: 'var(--radius-md)',
                  background: isSelected
                    ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%)'
                    : 'rgba(255, 255, 255, 0.02)',
                  border: isSelected ? '1px solid var(--color-brand)' : '1px solid var(--border-subtle)',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.4rem',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <span style={{ fontSize: '1.05rem', fontWeight: '800', fontFamily: 'var(--font-display)', color: 'white' }}>
                      {stock.ticker}
                    </span>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '140px' }}>
                      {stock.company}
                    </div>
                  </div>
                  <span className="badge" style={{ fontSize: '0.65rem', background: 'rgba(255, 255, 255, 0.06)' }}>
                    {stock.sector}
                  </span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginTop: '0.4rem' }}>
                  <span style={{ fontSize: '1.1rem', fontWeight: '700', fontFamily: 'var(--font-mono)', color: 'white' }}>
                    ₹{stock.latest_price?.toLocaleString('en-IN')}
                  </span>
                  <span
                    style={{
                      fontSize: '0.82rem',
                      fontWeight: '700',
                      color: isPositive ? 'var(--color-buy)' : 'var(--color-sell)',
                    }}
                  >
                    {isPositive ? `+${stock.change_pct_1d}%` : `${stock.change_pct_1d}%`}
                  </span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', borderTop: '1px solid rgba(255, 255, 255, 0.04)', paddingTop: '0.4rem' }}>
                  <span>1M: <strong style={{ color: stock.change_pct_1m >= 0 ? 'var(--color-buy)' : 'var(--color-sell)' }}>{stock.change_pct_1m >= 0 ? `+${stock.change_pct_1m}%` : `${stock.change_pct_1m}%`}</strong></span>
                  <span>{stock.trend}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

