import React from 'react';

export default function Home({ onNavigate }) {
  const supportedStocks = [
    { ticker: 'TCS', name: 'Tata Consultancy Services Ltd', sector: 'Information Technology' },
    { ticker: 'INFY', name: 'Infosys Ltd', sector: 'Information Technology' },
    { ticker: 'RELIANCE', name: 'Reliance Industries Ltd', sector: 'Energy & Conglomerate' },
  ];

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
        <div style={{ maxWidth: '780px', display: 'flex', flexDirection: 'column', gap: '1.25rem', zIndex: 2, position: 'relative' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <span className="badge badge-internal">Production Architecture</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Locked Model Version 1.0.0</span>
          </div>

          <h1 style={{ fontSize: '2.75rem', lineHeight: '1.15', letterSpacing: '-0.03em' }}>
            Explainable AI Decision Support for <span style={{ background: 'linear-gradient(135deg, #6366F1, #A855F7)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Equities Movement</span>
          </h1>

          <p style={{ fontSize: '1.1rem', lineHeight: '1.6', color: 'var(--text-secondary)' }}>
            StockDNA-AI classifies 5-trading-day stock price trajectories into <strong>BUY</strong>, <strong>HOLD (KEEP)</strong>, and <strong>SELL</strong> regimes, decomposing every prediction through <strong>SHAP TreeExplainer</strong> into company-specific <strong>INTERNAL</strong> vs market-driven <strong>EXTERNAL</strong> factor attributions.
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

      {/* Three Core Deliverables */}
      <section style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
          <div>
            <h2 style={{ fontSize: '1.75rem' }}>Three Core System Deliverables</h2>
            <p style={{ fontSize: '0.9rem' }}>Architectural pillars verified with strict scale-invariance and zero leakage.</p>
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
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>XGBoost Engine</span>
            </div>
            <h3 style={{ fontSize: '1.35rem' }}>Stock Movement Prediction</h3>
            <p style={{ fontSize: '0.9rem', lineHeight: '1.6' }}>
              Multi-class predictive model trained on 34 scale-invariant features. Classifies 5-trading-day forward returns into:
            </p>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <span className="badge badge-buy">BUY (&ge; +2.0%)</span>
              <span className="badge badge-hold">HOLD (&plusmn;2.0%)</span>
              <span className="badge badge-sell">SELL (&le; -2.0%)</span>
            </div>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => onNavigate('predict')}
              style={{ marginTop: 'auto', width: '100%' }}
            >
              Launch Prediction &rarr;
            </button>
          </div>

          {/* Card 2: XAI Decomposition */}
          <div
            className="glass-panel"
            style={{
              padding: '2rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '1rem',
              borderTop: '3px solid var(--color-external)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span className="badge badge-external">Core Deliverable 2</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>TreeExplainer</span>
            </div>
            <h3 style={{ fontSize: '1.35rem' }}>Explainable AI (SHAP)</h3>
            <p style={{ fontSize: '0.9rem', lineHeight: '1.6' }}>
              Decomposes every individual prediction into company-specific micro factors vs broader market and technical factors:
            </p>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <span className="badge badge-internal">INTERNAL (Earnings/EPS)</span>
              <span className="badge badge-external">EXTERNAL (VIX/Technical)</span>
            </div>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => onNavigate('xai')}
              style={{ marginTop: 'auto', width: '100%' }}
            >
              View SHAP Visualizer &rarr;
            </button>
          </div>

          {/* Card 3: Independent Data Tool */}
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
        <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Audited Indian Equities Test Universe</h3>
        <p style={{ fontSize: '0.85rem', marginBottom: '1.5rem', color: 'var(--text-secondary)' }}>
          High-liquidity NSE NIFTY benchmark components evaluated under strict out-of-sample chronological splits (2018–2025).
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
          {supportedStocks.map((stock) => (
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
              }}
            >
              <div>
                <span style={{ fontSize: '1.15rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                  {stock.ticker}
                </span>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{stock.name}</p>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{stock.sector}</span>
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
