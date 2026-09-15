import React, { useState, useEffect } from 'react';
import { api, authStorage } from '../services/api';
import AttributionDonut from '../components/AttributionDonut';
import ShapWaterfall from '../components/ShapWaterfall';
import DecisionBadge from '../components/DecisionBadge';

export default function Explainability({ initialPrediction, onNavigate }) {
  const [predictionData, setPredictionData] = useState(initialPrediction || null);
  const [ticker, setTicker] = useState('TCS');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (initialPrediction) {
      setPredictionData(initialPrediction);
    }
  }, [initialPrediction]);

  const handleFetchExplanation = async (e) => {
    if (e) e.preventDefault();
    if (!authStorage.isLoggedIn()) {
      setError('Please sign in or register to execute live model predictions and SHAP explanations.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await api.predict(ticker);
      setPredictionData(res);
    } catch (err) {
      setError(err.message || 'Failed to retrieve SHAP explanation.');
    } finally {
      setLoading(false);
    }
  };

  const xai = predictionData?.xai;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.35rem' }}>
            <span className="badge badge-external">Explainable AI Core</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>TreeExplainer Algorithm</span>
          </div>
          <h2 style={{ fontSize: '2rem' }}>SHAP Explainability & Factor Attribution</h2>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
            Transparently answers <em>"Why did the model make this decision?"</em> by isolating company-specific INTERNAL vs market EXTERNAL factor attributions.
          </p>
        </div>

        {/* Quick ticker switcher if already viewing an explanation */}
        <form onSubmit={handleFetchExplanation} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <select
            className="form-select"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            style={{ padding: '0.45rem 0.75rem', fontSize: '0.85rem' }}
          >
            <option value="TCS">TCS</option>
            <option value="INFY">INFY</option>
            <option value="RELIANCE">RELIANCE</option>
          </select>
          <button
            type="submit"
            className="btn btn-secondary"
            disabled={loading}
            style={{ padding: '0.45rem 0.85rem', fontSize: '0.85rem' }}
          >
            {loading ? <span className="spinner" /> : 'Explain Ticker'}
          </button>
        </form>
      </div>

      {error && (
        <div
          style={{
            padding: '1rem',
            borderRadius: 'var(--radius-md)',
            background: 'var(--color-sell-bg)',
            border: '1px solid var(--color-sell-border)',
            color: 'var(--color-sell)',
            fontSize: '0.85rem',
          }}
        >
          {error}
        </div>
      )}

      {loading && (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
          <span className="spinner" style={{ width: '2rem', height: '2rem', marginBottom: '1rem' }} />
          <p>Calculating multiclass SHAP attribution vectors across 34 features...</p>
        </div>
      )}

      {!predictionData && !loading && (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>No Active Prediction Loaded</h3>
          <p style={{ fontSize: '0.9rem', marginBottom: '1.5rem' }}>
            Select a stock and click "Explain Ticker" or run a prediction from the Prediction page.
          </p>
          <button type="button" className="btn btn-primary" onClick={handleFetchExplanation}>
            Run Explanation for {ticker} &rarr;
          </button>
        </div>
      )}

      {predictionData && !loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {/* Active Context Bar */}
          <div
            className="glass-panel"
            style={{
              padding: '1.25rem 1.5rem',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '1rem',
            }}
          >
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Analyzed Asset</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <span style={{ fontSize: '1.35rem', fontWeight: '800', fontFamily: 'var(--font-display)' }}>
                  {predictionData.ticker}
                </span>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{predictionData.company}</span>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Predicted Class</span>
                <div>
                  <DecisionBadge prediction={predictionData.prediction} />
                </div>
              </div>

              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Confidence</span>
                <div style={{ fontSize: '1.15rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                  {(predictionData.confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {/* Section A: Attribution Donut & Academic Scope */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.5rem' }}>
            {/* Donut Chart */}
            <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <AttributionDonut
                internalPercentage={xai?.internal_percentage}
                externalPercentage={xai?.external_percentage}
                primaryDriver={xai?.primary_driver}
              />
            </div>

            {/* Academic Explanation & Definitions */}
            <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <h3 style={{ fontSize: '1.25rem' }}>Methodological Attribution Scope</h3>

              <p style={{ fontSize: '0.88rem', lineHeight: '1.6' }}>
                StockDNA-AI utilizes <strong>SHAP (SHapley Additive exPlanations)</strong> via tree-kernel margins to quantify the exact contribution of each variable toward the target class log-odds.
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div style={{ padding: '0.75rem 1rem', borderRadius: 'var(--radius-md)', background: 'var(--color-internal-bg)', border: '1px solid var(--color-internal-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.2rem' }}>
                    <span className="badge badge-internal">INTERNAL Factors</span>
                    <strong style={{ fontSize: '0.85rem', color: 'var(--color-internal)' }}>Company-Specific Micro Indicators</strong>
                  </div>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Reported EPS, consensus surprises, earnings announcement events, and point-in-time post-earnings drift windows.
                  </p>
                </div>

                <div style={{ padding: '0.75rem 1rem', borderRadius: 'var(--radius-md)', background: 'var(--color-external-bg)', border: '1px solid var(--color-external-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.2rem' }}>
                    <span className="badge badge-external">EXTERNAL Factors</span>
                    <strong style={{ fontSize: '0.85rem', color: 'var(--color-external)' }}>Macro, Sector & Technical Indicators</strong>
                  </div>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    India VIX volatility benchmark, NIFTY 50 relative strength, normalized ATR, MACD spreads, and moving average distances.
                  </p>
                </div>
              </div>

              <div style={{ padding: '0.75rem', borderRadius: 'var(--radius-sm)', background: 'rgba(255, 255, 255, 0.03)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                <strong>Causality Disclaimer:</strong> Percentages reflect <em>model decision attribution</em> (how inputs altered the model's output distribution), not real-world economic causation.
              </div>
            </div>
          </div>

          {/* Section B: Top Drivers Breakdown */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.5rem' }}>
            {/* Top Internal Factors */}
            <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span className="badge badge-internal">Company Drivers</span>
                <h4 style={{ fontSize: '1.1rem' }}>Top Internal Factors</h4>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {(xai?.top_internal_factors || []).map((factor, idx) => (
                  <div
                    key={factor.feature || idx}
                    style={{
                      padding: '0.85rem',
                      borderRadius: 'var(--radius-md)',
                      background: 'rgba(255, 255, 255, 0.02)',
                      border: '1px solid var(--border-subtle)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.35rem',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.85rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                        {factor.feature}
                      </span>
                      <span
                        style={{
                          fontSize: '0.75rem',
                          fontWeight: '600',
                          fontFamily: 'var(--font-mono)',
                          color: factor.shap_value >= 0 ? 'var(--color-buy)' : 'var(--color-sell)',
                        }}
                      >
                        {factor.shap_value >= 0 ? `+${factor.shap_value.toFixed(4)}` : factor.shap_value.toFixed(4)}
                      </span>
                    </div>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>{factor.explanation}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Top External Factors */}
            <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span className="badge badge-external">Market Drivers</span>
                <h4 style={{ fontSize: '1.1rem' }}>Top External Factors</h4>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {(xai?.top_external_factors || []).map((factor, idx) => (
                  <div
                    key={factor.feature || idx}
                    style={{
                      padding: '0.85rem',
                      borderRadius: 'var(--radius-md)',
                      background: 'rgba(255, 255, 255, 0.02)',
                      border: '1px solid var(--border-subtle)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.35rem',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.85rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                        {factor.feature}
                      </span>
                      <span
                        style={{
                          fontSize: '0.75rem',
                          fontWeight: '600',
                          fontFamily: 'var(--font-mono)',
                          color: factor.shap_value >= 0 ? 'var(--color-buy)' : 'var(--color-sell)',
                        }}
                      >
                        {factor.shap_value >= 0 ? `+${factor.shap_value.toFixed(4)}` : factor.shap_value.toFixed(4)}
                      </span>
                    </div>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>{factor.explanation}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Section C: Full Bi-directional Waterfall Visualization */}
          <div className="glass-panel" style={{ padding: '2rem' }}>
            <ShapWaterfall
              contributions={xai?.all_feature_contributions || [...(xai?.top_external_factors || []), ...(xai?.top_internal_factors || [])]}
              predictedClass={predictionData.prediction}
              maxItems={20}
            />
          </div>
        </div>
      )}
    </div>
  );
}
