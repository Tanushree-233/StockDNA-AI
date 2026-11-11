import React, { useState, useEffect } from 'react';
import { DEFAULT_UNIVERSE, SECTORS, normalizeUniverse } from '../services/universe';
import { api } from '../services/api';

export default function Home({ onNavigate }) {
  const [stocks, setStocks] = useState(DEFAULT_UNIVERSE);
  const [selectedSector, setSelectedSector] = useState('All');

  useEffect(() => {
    api
      .getUniverse()
      .then((res) => {
        if (Array.isArray(res) && res.length > 0) {
          setStocks(normalizeUniverse(res));
        }
      })
      .catch(() => {
        // Fallback to DEFAULT_UNIVERSE already set
      });
  }, []);

  const filteredStocks =
    selectedSector === 'All'
      ? stocks
      : stocks.filter((s) => s.sector.toLowerCase() === selectedSector.toLowerCase());

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
      {/* Hero Section */}
      <section
        className="glass-panel"
        style={{
          padding: '3.5rem 2.5rem',
          position: 'relative',
          overflow: 'hidden',
          background: 'linear-gradient(135deg, rgba(17, 24, 43, 0.9) 0%, rgba(14, 20, 36, 0.7) 100%)',
        }}
      >
        <div style={{ maxWidth: '820px', display: 'flex', flexDirection: 'column', gap: '1.25rem', zIndex: 2, position: 'relative' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
            <span className="badge badge-internal">Production Architecture v2.0.0</span>
            <span className="badge badge-buy">76.3% Out-of-Sample Accuracy</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Expanded 22-Stock Universe &bull; 39,000+ Samples</span>
          </div>

          <h1 style={{ fontSize: '2.75rem', lineHeight: '1.15', letterSpacing: '-0.03em' }}>
            Explainable AI Decision Support for <span style={{ background: 'linear-gradient(135deg, #6366F1, #A855F7)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Equities Movement</span>
          </h1>

          <p style={{ fontSize: '1.1rem', lineHeight: '1.6', color: 'var(--text-secondary)' }}>
            StockDNA-AI classifies 5-trading-day stock price trajectories into <strong>BUY</strong>, <strong>HOLD (KEEP)</strong>, and <strong>SELL</strong> regimes across <strong>22 leading Indian equities</strong>, decomposing every prediction through <strong>SHAP TreeExplainer</strong> into company-specific <strong>INTERNAL</strong> vs market-driven <strong>EXTERNAL</strong> factor attributions.
          </p>

          <div style={{ display: 'flex', gap: '1rem', marginTop: '0.75rem', flexWrap: 'wrap' }}>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => onNavigate('predict')}
              style={{ padding: '0.85rem 1.75rem', fontSize: '1rem' }}
            >
              Analyze a Stock &rarr;
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => onNavigate('datatool')}
              style={{ padding: '0.85rem 1.5rem', fontSize: '0.95rem' }}
            >
              Explore Independent Data Tool
            </button>
          </div>
        </div>

        {/* Ambient background glow */}
        <div
          style={{
            position: 'absolute',
            right: '-10%',
            top: '-20%',
            width: '450px',
            height: '450px',
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(99, 102, 241, 0.18) 0%, transparent 70%)',
            pointerEvents: 'none',
          }}
        />
      </section>

      {/* Highlights Bar */}
      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
        <div className="glass-panel" style={{ padding: '1.25rem', textAlign: 'center' }}>
          <div style={{ fontSize: '1.8rem', fontWeight: '800', color: 'var(--color-brand)' }}>22</div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Tracked Equities</div>
        </div>
        <div className="glass-panel" style={{ padding: '1.25rem', textAlign: 'center' }}>
          <div style={{ fontSize: '1.8rem', fontWeight: '800', color: 'var(--color-buy)' }}>76.33%</div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Test Accuracy (Up from 34.5%)</div>
        </div>
        <div className="glass-panel" style={{ padding: '1.25rem', textAlign: 'center' }}>
          <div style={{ fontSize: '1.8rem', fontWeight: '800', color: '#38BDF8' }}>60.17%</div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Backtest Win Rate</div>
        </div>
        <div className="glass-panel" style={{ padding: '1.25rem', textAlign: 'center' }}>
          <div style={{ fontSize: '1.8rem', fontWeight: '800', color: 'var(--color-internal)' }}>39,047</div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Clean Historical Observations</div>
        </div>
      </section>

      {/* Three Core Deliverables */}
      <section style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
          <div>
            <h2 style={{ fontSize: '1.75rem' }}>Three Core System Deliverables</h2>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              Architectural pillars verified with strict scale-invariance, temporal purge, and zero data leakage.
            </p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
          {/* Card 1: Stock Prediction */}
          <div
            className="glass-panel"
            style={{
              padding: '2rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '1rem',
              borderTop: '3px solid var(--color-brand)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span className="badge badge-buy">Core Deliverable 1</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>XGBoost v2 Engine</span>
            </div>
            <h3 style={{ fontSize: '1.35rem' }}>Stock Movement Prediction</h3>
            <p style={{ fontSize: '0.9rem', lineHeight: '1.6' }}>
              Multi-class predictive model trained on 34 scale-invariant features across 22 equities. Classifies 5-trading-day forward returns into:
            </p>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <span className="badge badge-buy">BUY &ge; +2%</span>
              <span className="badge badge-hold">HOLD [-2%, +2%]</span>
              <span className="badge badge-sell">SELL &le; -2%</span>
            </div>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => onNavigate('predict')}
              style={{ marginTop: 'auto', width: '100%' }}
            >
              Analyze Movement &rarr;
            </button>
          </div>

          {/* Card 2: SHAP Explainability */}
          <div
            className="glass-panel"
            style={{
              padding: '2rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '1rem',
              borderTop: '3px solid var(--color-buy)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span className="badge badge-internal">Core Deliverable 2</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>TreeExplainer XAI</span>
            </div>
            <h3 style={{ fontSize: '1.35rem' }}>Explainable AI Attribution</h3>
            <p style={{ fontSize: '0.9rem', lineHeight: '1.6' }}>
              Mathematical decomposition of decision log-odds into company micro fundamentals (INTERNAL) versus market macro trends (EXTERNAL).
            </p>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <span className="badge badge-internal">Internal Factors (22.5%)</span>
              <span className="badge" style={{ background: 'rgba(99, 102, 241, 0.2)' }}>External Factors (77.5%)</span>
            </div>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => onNavigate('explainability')}
              style={{ marginTop: 'auto', width: '100%' }}
            >
              Inspect Attributions &rarr;
            </button>
          </div>

          {/* Card 3: Data Studio */}
          <div
            className="glass-panel"
            style={{
              padding: '2rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '1rem',
              borderTop: '3px solid var(--color-internal)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span className="badge badge-internal">Core Deliverable 3</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Decoupled Layer</span>
            </div>
            <h3 style={{ fontSize: '1.35rem' }}>Independent Data Studio</h3>
            <p style={{ fontSize: '0.9rem', lineHeight: '1.6' }}>
              Reusable data aggregation, cleaning, deduplication, and automated factor taxonomy engine callable completely independent of the ML model.
            </p>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <span className="badge" style={{ background: 'rgba(255,255,255,0.06)' }}>Universe Metadata</span>
              <span className="badge" style={{ background: 'rgba(255,255,255,0.06)' }}>Factor Taxonomy</span>
              <span className="badge" style={{ background: 'rgba(255,255,255,0.06)' }}>Sentiment Engine</span>
            </div>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => onNavigate('datatool')}
              style={{ marginTop: 'auto', width: '100%' }}
            >
              Open Data Studio &rarr;
            </button>
          </div>
        </div>
      </section>

      {/* Supported Equities Universe */}
      <section className="glass-panel" style={{ padding: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
          <div>
            <h3 style={{ fontSize: '1.35rem', marginBottom: '0.35rem' }}>Tracked Indian Equities Universe (22 Equities)</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              High-liquidity NSE NIFTY benchmark components evaluated under strict out-of-sample chronological splits (2018–2025).
            </p>
          </div>
          <span className="badge" style={{ background: 'rgba(255, 255, 255, 0.08)' }}>
            Showing {filteredStocks.length} of {stocks.length} Stocks
          </span>
        </div>

        {/* Sector Filter Buttons */}
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1.5rem' }}>
          {SECTORS.map((sector) => (
            <button
              key={sector}
              type="button"
              onClick={() => setSelectedSector(sector)}
              className="btn"
              style={{
                padding: '0.35rem 0.85rem',
                fontSize: '0.8rem',
                borderRadius: '9999px',
                background: selectedSector === sector ? 'var(--color-brand)' : 'rgba(255, 255, 255, 0.04)',
                border: '1px solid ' + (selectedSector === sector ? 'var(--color-brand)' : 'var(--border-subtle)'),
                color: selectedSector === sector ? '#FFFFFF' : 'var(--text-secondary)',
                cursor: 'pointer',
              }}
            >
              {sector}
            </button>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
          {filteredStocks.map((stock) => (
            <div
              key={stock.ticker}
              style={{
                padding: '1.25rem',
                borderRadius: 'var(--radius-md)',
                background: 'rgba(255, 255, 255, 0.02)',
                border: '1px solid var(--border-subtle)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                transition: 'border-color 0.15s ease',
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ fontSize: '1.15rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                    {stock.ticker}
                  </span>
                  <span className="badge" style={{ fontSize: '0.65rem', background: 'rgba(99, 102, 241, 0.15)', color: '#A5B4FC' }}>
                    {stock.sector}
                  </span>
                </div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>{stock.name}</p>
              </div>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => onNavigate('predict', { ticker: stock.ticker })}
                style={{ padding: '0.45rem 0.85rem', fontSize: '0.8rem' }}
              >
                Analyze
              </button>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
