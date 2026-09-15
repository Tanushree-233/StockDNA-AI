import React, { useState } from 'react';
import { api, authStorage } from '../services/api';
import DecisionBadge from '../components/DecisionBadge';
import ProbabilityGauge from '../components/ProbabilityGauge';
import AuthModal from '../components/AuthModal';

export default function Predict({ onNavigate, initialTicker = 'TCS', onPredictionSuccess }) {
  const [ticker, setTicker] = useState(initialTicker);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isAuthOpen, setIsAuthOpen] = useState(false);

  const isLoggedIn = authStorage.isLoggedIn();

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    if (!isLoggedIn) {
      setIsAuthOpen(true);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await api.predict(ticker);
      setResult(data);
      if (onPredictionSuccess) {
        onPredictionSuccess(data);
      }
    } catch (err) {
      setError(err.message || 'Failed to generate prediction. Verify backend status.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header */}
      <div>
        <h2 style={{ fontSize: '2rem' }}>Stock Movement Prediction Engine</h2>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
          Generates 5-trading-day directional return probabilities using the locked XGBoost production model.
        </p>
      </div>

      {/* Input Form Card */}
      <div className="glass-panel" style={{ padding: '1.75rem' }}>
        <form onSubmit={handlePredict} style={{ display: 'flex', gap: '1.25rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-group" style={{ minWidth: '260px', flex: 1 }}>
            <label className="form-label">Select Equity Ticker</label>
            <select
              className="form-select"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              disabled={loading}
            >
              <option value="TCS">TCS — Tata Consultancy Services Ltd</option>
              <option value="INFY">INFY — Infosys Ltd</option>
              <option value="RELIANCE">RELIANCE — Reliance Industries Ltd</option>
            </select>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            style={{ height: '42px', minWidth: '160px' }}
          >
            {loading ? <span className="spinner" /> : 'Run Prediction'}
          </button>
        </form>

        {!isLoggedIn && (
          <div
            style={{
              marginTop: '1rem',
              padding: '0.75rem 1rem',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(99, 102, 241, 0.08)',
              border: '1px solid rgba(99, 102, 241, 0.25)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Authentication required to run predictions and record isolated history.
            </span>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => setIsAuthOpen(true)}
              style={{ padding: '0.35rem 0.85rem', fontSize: '0.8rem' }}
            >
              Sign In / Register
            </button>
          </div>
        )}
      </div>

      {/* Error Alert */}
      {error && (
        <div
          style={{
            padding: '1rem 1.25rem',
            borderRadius: 'var(--radius-md)',
            background: 'var(--color-sell-bg)',
            border: '1px solid var(--color-sell-border)',
            color: 'var(--color-sell)',
            fontSize: '0.9rem',
          }}
        >
          <strong>Prediction Error:</strong> {error}
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
          <span className="spinner" style={{ width: '2rem', height: '2rem' }} />
          <div>
            <h4 style={{ fontSize: '1.1rem' }}>Executing Machine Learning Pipeline...</h4>
            <p style={{ fontSize: '0.85rem' }}>Fetching point-in-time features &rarr; Evaluating 34 scale-invariant indicators &rarr; Computing TreeExplainer SHAP</p>
          </div>
        </div>
      )}

      {/* Result: Production Decision Card */}
      {result && !loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
            {/* Decision Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.35rem' }}>
                  <span style={{ fontSize: '1.5rem', fontWeight: '800', fontFamily: 'var(--font-display)' }}>
                    {result.ticker}
                  </span>
                  <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    {result.company}
                  </span>
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Analysis Timestamp: {new Date(result.timestamp).toLocaleString()} &bull; Model Contract v{result.model_version || '1.0.0'}
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>Predicted Movement</span>
                  <DecisionBadge prediction={result.prediction} size="large" />
                </div>
                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>Model Confidence</span>
                  <span style={{ fontSize: '1.35rem', fontWeight: '800', fontFamily: 'var(--font-display)', color: 'var(--text-primary)' }}>
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Probability Gauge */}
            <ProbabilityGauge probabilities={result.probabilities} />

            {/* High-Level Attribution Callout */}
            <div
              style={{
                padding: '1.25rem',
                borderRadius: 'var(--radius-md)',
                background: 'rgba(255, 255, 255, 0.02)',
                border: '1px solid var(--border-subtle)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '1rem',
              }}
            >
              <div>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  SHAP Primary Driver
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.2rem' }}>
                  <span
                    className={`badge ${result.xai?.primary_driver === 'INTERNAL' ? 'badge-internal' : 'badge-external'}`}
                  >
                    {result.xai?.primary_driver} Factors
                  </span>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    Internal: <strong>{result.xai?.internal_percentage?.toFixed(1)}%</strong> &bull; External: <strong>{result.xai?.external_percentage?.toFixed(1)}%</strong>
                  </span>
                </div>
              </div>

              <button
                type="button"
                className="btn btn-primary"
                onClick={() => onNavigate('xai', { prediction: result })}
              >
                Inspect SHAP Decomposition &rarr;
              </button>
            </div>

            {/* Academic Decision-Support Disclaimer */}
            <div
              style={{
                fontSize: '0.75rem',
                color: 'var(--text-muted)',
                borderTop: '1px solid var(--border-subtle)',
                paddingTop: '0.85rem',
                lineHeight: '1.4',
              }}
            >
              <strong>Disclaimer:</strong> Decision-support output — not financial advice. The model predicts an expected 5-trading-day return category based on historical statistical patterns and does not guarantee future price movement or capital preservation.
            </div>
          </div>
        </div>
      )}

      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        onSuccess={() => {
          setIsAuthOpen(false);
          handlePredict();
        }}
      />
    </div>
  );
}
