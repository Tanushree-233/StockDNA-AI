import React, { useState, useEffect } from 'react';
import { api, authStorage } from '../services/api';
import { DEFAULT_UNIVERSE, normalizeUniverse } from '../services/universe';
import AttributionDonut from '../components/AttributionDonut';
import ShapWaterfall from '../components/ShapWaterfall';
import DecisionBadge from '../components/DecisionBadge';

export default function Explainability({ initialPrediction, onNavigate }) {
  const [predictionData, setPredictionData] = useState(initialPrediction || null);
  const [ticker, setTicker] = useState('TCS');
  const [universe, setUniverse] = useState(DEFAULT_UNIVERSE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  useEffect(() => {
    api
      .getUniverse()
      .then((res) => {
        if (Array.isArray(res) && res.length > 0) {
          setUniverse(normalizeUniverse(res));
        }
      })
      .catch(() => {});
  }, []);

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
  const laymanSummary =
    predictionData?.layman_summary ||
    xai?.layman_summary ||
    (predictionData
      ? `The AI recommends ${predictionData.prediction} with ${(predictionData.confidence * 100).toFixed(1)}% confidence based on ${xai?.primary_driver === 'EXTERNAL' ? 'broader market sentiment and price momentum' : 'company quarterly earnings'}.`
      : null);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.35rem' }}>
            <span className="badge badge-external">Explainable AI Hub</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--color-brand)', fontWeight: '600' }}>Layman-Friendly Intelligence</span>
          </div>
          <h2 style={{ fontSize: '2rem' }}>Why Did the AI Make This Prediction?</h2>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', maxWidth: '750px' }}>
            Zero mysterious black boxes. Our Explainable AI translates mathematical signals into plain, everyday English so anyone can understand why a stock is rated <strong>BUY</strong>, <strong>HOLD</strong>, or <strong>SELL</strong>.
          </p>
        </div>

        {/* Quick ticker switcher */}
        <form onSubmit={handleFetchExplanation} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <select
            className="form-select"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            style={{ padding: '0.45rem 0.75rem', fontSize: '0.85rem' }}
          >
            {universe.map((stock) => (
              <option key={stock.ticker} value={stock.ticker}>
                {stock.ticker} — {stock.name}
              </option>
            ))}
          </select>
          <button
            type="submit"
            className="btn btn-secondary"
            disabled={loading}
            style={{ padding: '0.45rem 0.85rem', fontSize: '0.85rem' }}
          >
            {loading ? <span className="spinner" /> : 'Explain Stock'}
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
          <p>Translating 34 statistical clues into plain English explanations...</p>
        </div>
      )}

      {!predictionData && !loading && (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>No Active Stock Analysis Loaded</h3>
          <p style={{ fontSize: '0.9rem', marginBottom: '1.5rem', color: 'var(--text-secondary)' }}>
            Select a stock above and click "Explain Stock" or run a prediction from the Prediction Engine.
          </p>
          <button type="button" className="btn btn-primary" onClick={handleFetchExplanation}>
            Explain {ticker} in Plain English &rarr;
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
              borderLeft: '4px solid var(--color-brand)',
            }}
          >
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Analyzed Company</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <span style={{ fontSize: '1.35rem', fontWeight: '800', fontFamily: 'var(--font-display)' }}>
                  {predictionData.ticker}
                </span>
                <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{predictionData.company}</span>
                <span className="badge" style={{ background: 'rgba(255, 255, 255, 0.05)', fontSize: '0.72rem' }}>
                  {predictionData.sector || 'Equities'}
                </span>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1.75rem', alignItems: 'center' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>AI 5-Day Verdict</span>
                <div>
                  <DecisionBadge prediction={predictionData.prediction} />
                </div>
              </div>

              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Confidence Score</span>
                <div style={{ fontSize: '1.25rem', fontWeight: '800', color: 'var(--text-primary)' }}>
                  {(predictionData.confidence * 100).toFixed(1)}%
                </div>
              </div>

              <div style={{ borderLeft: '1px solid var(--border-subtle)', paddingLeft: '1.25rem' }}>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                  style={{ fontSize: '0.78rem', padding: '0.35rem 0.75rem' }}
                >
                  {showTechnicalDetails ? 'Simple View (Hide Math)' : 'Show Technical Math'}
                </button>
              </div>
            </div>
          </div>

          {/* Layman Executive Summary Banner */}
          {laymanSummary && (
            <div
              className="glass-panel"
              style={{
                padding: '1.5rem 1.75rem',
                background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.08) 100%)',
                border: '1px solid rgba(99, 102, 241, 0.35)',
                borderRadius: 'var(--radius-lg)',
                display: 'flex',
                gap: '1.25rem',
                alignItems: 'flex-start',
              }}
            >
              <div
                style={{
                  width: '42px',
                  height: '42px',
                  borderRadius: '50%',
                  background: 'var(--color-brand)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                  fontSize: '1.25rem',
                }}
              >
                💡
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <h3 style={{ fontSize: '1.1rem', margin: 0 }}>Executive Summary (In Plain English)</h3>
                  <span className="badge badge-internal" style={{ fontSize: '0.7rem' }}>Zero Jargon</span>
                </div>
                <p style={{ fontSize: '0.92rem', lineHeight: '1.65', color: 'var(--text-primary)', margin: 0 }}>
                  {laymanSummary}
                </p>
              </div>
            </div>
          )}

          {/* Section A: Attribution Donut & Layman Explanation */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.5rem' }}>
            {/* Donut Chart */}
            <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <AttributionDonut
                internalPercentage={xai?.internal_percentage}
                externalPercentage={xai?.external_percentage}
                primaryDriver={xai?.primary_driver}
              />
              <div style={{ textAlign: 'center', marginTop: '1rem', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                {xai?.primary_driver === 'EXTERNAL' ? (
                  <span>
                    The decision was primarily driven by <strong>Market Factors</strong> ({xai?.external_percentage}%): broader stock market sentiment, price swings, and trading activity.
                  </span>
                ) : (
                  <span>
                    The decision was primarily driven by <strong>Company Fundamentals</strong> ({xai?.internal_percentage}%): quarterly profits, earnings beats, and financial report timing.
                  </span>
                )}
              </div>
            </div>

            {/* Plain English Guide Card */}
            <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <h3 style={{ fontSize: '1.25rem' }}>How StockDNA Explains Its Decisions</h3>

              <p style={{ fontSize: '0.88rem', lineHeight: '1.6', color: 'var(--text-secondary)' }}>
                Instead of being a mysterious black box, our AI works like a panel of judges scoring <strong>34 distinct clues</strong>. It groups these clues into two understandable categories:
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                <div style={{ padding: '0.85rem 1.1rem', borderRadius: 'var(--radius-md)', background: 'var(--color-internal-bg)', border: '1px solid var(--color-internal-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                    <span className="badge badge-internal">Company Factors</span>
                    <strong style={{ fontSize: '0.88rem', color: 'var(--color-internal)' }}>Company Financial Clues (10 Indicators)</strong>
                  </div>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', margin: 0, lineHeight: '1.5' }}>
                    What the company reported about itself: quarterly profits, whether profits beat or missed analyst expectations, profit yield compared to share price, and time passed since the last financial report.
                  </p>
                </div>

                <div style={{ padding: '0.85rem 1.1rem', borderRadius: 'var(--radius-md)', background: 'var(--color-external-bg)', border: '1px solid var(--color-external-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                    <span className="badge badge-external">Market Factors</span>
                    <strong style={{ fontSize: '0.88rem', color: 'var(--color-external)' }}>Market Climate & Price Trends (24 Indicators)</strong>
                  </div>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', margin: 0, lineHeight: '1.5' }}>
                    What is happening on Dalal Street: day-to-day price swings, overall market fear (India VIX), trading volume surges, and whether the stock is trading above or below its 20-day or 50-day average.
                  </p>
                </div>
              </div>

              <div style={{ padding: '0.75rem', borderRadius: 'var(--radius-sm)', background: 'rgba(255, 255, 255, 0.03)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                <strong>Transparent Attribution:</strong> The percentages indicate how much each category nudged the AI's final score up or down.
              </div>
            </div>
          </div>

          {/* Section B: Top Drivers Breakdown in Layman Terms */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
            {/* Top Internal Factors */}
            <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.15rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span className="badge badge-internal">Company Clues</span>
                  <h4 style={{ fontSize: '1.15rem', margin: 0 }}>Top Company Factors</h4>
                </div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Earnings & Financials</span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                {(xai?.top_internal_factors || []).map((factor, idx) => {
                  const displayName = factor.layman_name || factor.feature.replace(/_/g, ' ');
                  const explanationText = factor.layman_explanation || factor.explanation;
                  const isPositive = factor.shap_value >= 0;

                  return (
                    <div
                      key={factor.feature || idx}
                      style={{
                        padding: '1rem',
                        borderRadius: 'var(--radius-md)',
                        background: 'rgba(255, 255, 255, 0.025)',
                        border: '1px solid var(--border-subtle)',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '0.5rem',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem' }}>
                        <div>
                          <div style={{ fontSize: '0.92rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                            {displayName}
                          </div>
                          <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                            Indicator code: {factor.feature}
                          </span>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <span
                            className="badge"
                            style={{
                              fontSize: '0.7rem',
                              background: isPositive ? 'rgba(34, 197, 94, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                              color: isPositive ? 'var(--color-buy)' : 'var(--color-sell)',
                              border: `1px solid ${isPositive ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                            }}
                          >
                            {factor.layman_signal || (isPositive ? `Supports ${predictionData.prediction}` : 'Pushes alternative')}
                          </span>
                          {showTechnicalDetails && (
                            <div style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                              SHAP: {factor.shap_value >= 0 ? `+${factor.shap_value.toFixed(4)}` : factor.shap_value.toFixed(4)}
                            </div>
                          )}
                        </div>
                      </div>

                      <p style={{ fontSize: '0.84rem', color: 'var(--text-primary)', margin: 0, lineHeight: '1.5' }}>
                        {explanationText}
                      </p>

                      {factor.layman_definition && (
                        <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', background: 'rgba(255, 255, 255, 0.02)', padding: '0.4rem 0.6rem', borderRadius: 'var(--radius-sm)' }}>
                          <strong>What this measures:</strong> {factor.layman_definition}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Top External Factors */}
            <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.15rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span className="badge badge-external">Market Clues</span>
                  <h4 style={{ fontSize: '1.15rem', margin: 0 }}>Top Market Factors</h4>
                </div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Price Swings & Market Climate</span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                {(xai?.top_external_factors || []).map((factor, idx) => {
                  const displayName = factor.layman_name || factor.feature.replace(/_/g, ' ');
                  const explanationText = factor.layman_explanation || factor.explanation;
                  const isPositive = factor.shap_value >= 0;

                  return (
                    <div
                      key={factor.feature || idx}
                      style={{
                        padding: '1rem',
                        borderRadius: 'var(--radius-md)',
                        background: 'rgba(255, 255, 255, 0.025)',
                        border: '1px solid var(--border-subtle)',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '0.5rem',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem' }}>
                        <div>
                          <div style={{ fontSize: '0.92rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                            {displayName}
                          </div>
                          <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                            Indicator code: {factor.feature}
                          </span>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <span
                            className="badge"
                            style={{
                              fontSize: '0.7rem',
                              background: isPositive ? 'rgba(34, 197, 94, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                              color: isPositive ? 'var(--color-buy)' : 'var(--color-sell)',
                              border: `1px solid ${isPositive ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                            }}
                          >
                            {factor.layman_signal || (isPositive ? `Supports ${predictionData.prediction}` : 'Pushes alternative')}
                          </span>
                          {showTechnicalDetails && (
                            <div style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                              SHAP: {factor.shap_value >= 0 ? `+${factor.shap_value.toFixed(4)}` : factor.shap_value.toFixed(4)}
                            </div>
                          )}
                        </div>
                      </div>

                      <p style={{ fontSize: '0.84rem', color: 'var(--text-primary)', margin: 0, lineHeight: '1.5' }}>
                        {explanationText}
                      </p>

                      {factor.layman_definition && (
                        <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', background: 'rgba(255, 255, 255, 0.02)', padding: '0.4rem 0.6rem', borderRadius: 'var(--radius-sm)' }}>
                          <strong>What this measures:</strong> {factor.layman_definition}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Section C: Complete Clue-by-Clue Breakdown */}
          <div className="glass-panel" style={{ padding: '2rem' }}>
            <div style={{ marginBottom: '1.25rem' }}>
              <h3 style={{ fontSize: '1.25rem', marginBottom: '0.35rem' }}>Complete Clue-by-Clue Breakdown (Waterfall)</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Bars to the right (+ green) pushed the AI toward deciding <strong>{predictionData.prediction}</strong>. Bars to the left (- red) urged hesitation or favored another choice.
              </p>
            </div>
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
