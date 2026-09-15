import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

export default function DataTool() {
  const [activeTab, setActiveTab] = useState('universe'); // 'universe' | 'factors' | 'sentiment' | 'cleaner'

  // Universe state
  const [universe, setUniverse] = useState(null);
  const [universeLoading, setUniverseLoading] = useState(false);

  // Factor Taxonomy state
  const [factors, setFactors] = useState(null);
  const [factorsLoading, setFactorsLoading] = useState(false);
  const [customFeatures, setCustomFeatures] = useState('Reported_EPS, Daily_Return, VIX_Level, EPS_Surprise, RSI_14, Volume_to_MA20, Days_Since_Earnings');

  // Sentiment state
  const [headline, setHeadline] = useState('TCS reports 12% profit growth, beating analyst expectations amid strong deal wins.');
  const [sentimentResult, setSentimentResult] = useState(null);
  const [sentimentLoading, setSentimentLoading] = useState(false);

  // Cleaner state
  const [cleanerResult, setCleanerResult] = useState(null);
  const [cleanerLoading, setCleanerLoading] = useState(false);

  useEffect(() => {
    fetchUniverse();
    fetchFactors();
  }, []);

  const fetchUniverse = async () => {
    setUniverseLoading(true);
    try {
      const data = await api.getUniverse();
      setUniverse(data);
    } catch (err) {
      console.error('Universe fetch error:', err);
    } finally {
      setUniverseLoading(false);
    }
  };

  const fetchFactors = async (featureList = []) => {
    setFactorsLoading(true);
    const listToClassify = featureList.length > 0 
      ? featureList 
      : customFeatures.split(',').map((s) => s.trim()).filter(Boolean);

    try {
      const data = await api.classifyFactors(listToClassify);
      setFactors(data);
    } catch (err) {
      console.error('Factor classify error:', err);
    } finally {
      setFactorsLoading(false);
    }
  };

  const handleCustomFactorClassify = (e) => {
    e.preventDefault();
    const list = customFeatures
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    fetchFactors(list);
  };

  const handleSentimentAnalyze = async (e) => {
    e.preventDefault();
    setSentimentLoading(true);
    try {
      const res = await api.classifySentiment(headline);
      setSentimentResult(res);
    } catch (err) {
      console.error('Sentiment error:', err);
    } finally {
      setSentimentLoading(false);
    }
  };

  const handleRunSampleCleaning = async () => {
    setCleanerLoading(true);
    const sampleRawRecords = [
      { Date: '2024-01-15', Ticker: 'TCS', Close: 3850.5, Volume: 1250000 },
      { Date: '2024-01-15', Ticker: 'TCS', Close: 3850.5, Volume: 1250000 }, // Duplicate
      { Date: '2024-01-16', Ticker: 'TCS', Close: null, Volume: 980000 }, // Missing price
      { Date: '2024-01-17', Ticker: 'TCS', Close: 3910.0, Volume: -500 }, // Invalid volume
      { Date: '2024-01-18', Ticker: 'INFY', Close: 1540.2, Volume: 2400000 },
      { Date: '2024-01-19', Ticker: 'RELIANCE', Close: 2750.0, Volume: 3100000 },
    ];

    try {
      const res = await api.processData(sampleRawRecords);
      setCleanerResult(res);
    } catch (err) {
      console.error('Cleaner error:', err);
    } finally {
      setCleanerLoading(false);
    }
  };

  const universeList = Array.isArray(universe) ? universe : (universe?.universe || []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.35rem' }}>
          <span className="badge badge-internal">Decoupled Architecture</span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Standalone Layer &bull; No ML Model Coupling</span>
        </div>
        <h2 style={{ fontSize: '2rem' }}>Independent Data Studio</h2>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', maxWidth: '820px' }}>
          The StockDNA Data Tool is a reusable data aggregation and classification layer that operates independently of the predictive model. It can be consumed by other pipelines through the <code>BaseDataProvider</code> interface.
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.5rem', overflowX: 'auto' }}>
        {[
          { id: 'universe', label: '1. Universe Metadata' },
          { id: 'factors', label: '2. Factor Taxonomy Classifier' },
          { id: 'sentiment', label: '3. Financial Sentiment' },
          { id: 'cleaner', label: '4. Data Cleaning & Provenance' },
        ].map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`btn ${activeTab === tab.id ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setActiveTab(tab.id)}
            style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab 1: Universe Metadata */}
      {activeTab === 'universe' && (
        <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h3 style={{ fontSize: '1.25rem' }}>Supported Equity Universe</h3>
              <p style={{ fontSize: '0.85rem' }}>Served directly by <code>GET /datatool/universe</code></p>
            </div>
            <button type="button" className="btn btn-secondary" onClick={fetchUniverse} disabled={universeLoading}>
              {universeLoading ? <span className="spinner" /> : 'Refresh Universe'}
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
            {universeList.map((item) => (
              <div
                key={item.ticker}
                style={{
                  padding: '1.25rem',
                  borderRadius: 'var(--radius-md)',
                  background: 'rgba(255, 255, 255, 0.02)',
                  border: '1px solid var(--border-subtle)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.5rem',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '1.25rem', fontWeight: '800', fontFamily: 'var(--font-display)' }}>
                    {item.ticker}
                  </span>
                  <span className="badge" style={{ background: 'rgba(99, 102, 241, 0.15)', color: 'var(--color-brand)' }}>
                    {item.exchange || 'NSE'}
                  </span>
                </div>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{item.name}</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{item.sector}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 2: Factor Taxonomy Classifier */}
      {activeTab === 'factors' && (
        <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div>
            <h3 style={{ fontSize: '1.25rem' }}>Automated Factor Classification Engine</h3>
            <p style={{ fontSize: '0.85rem' }}>
              Classifies feature variables into <strong>INTERNAL</strong> (company-specific fundamentals) vs <strong>EXTERNAL</strong> (market/technical factors) using <code>config/feature_registry.py</code> as source of truth.
            </p>
          </div>

          {/* Interactive Input */}
          <form onSubmit={handleCustomFactorClassify} style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <div className="form-group" style={{ flex: 1, minWidth: '300px' }}>
              <label className="form-label">Comma-Separated Features to Classify</label>
              <input
                type="text"
                className="form-input"
                value={customFeatures}
                onChange={(e) => setCustomFeatures(e.target.value)}
              />
            </div>
            <button type="submit" className="btn btn-primary" disabled={factorsLoading} style={{ height: '42px' }}>
              {factorsLoading ? <span className="spinner" /> : 'Classify Factors'}
            </button>
          </form>

          {/* Classification Results */}
          {factors && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem', marginTop: '0.5rem' }}>
              {/* Internal Factors */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span className="badge badge-internal">INTERNAL Factors ({factors.classified?.INTERNAL?.length || 0})</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {(factors.classified?.INTERNAL || []).map((feat) => (
                    <div
                      key={feat}
                      style={{
                        padding: '0.65rem 0.85rem',
                        borderRadius: 'var(--radius-sm)',
                        background: 'var(--color-internal-bg)',
                        border: '1px solid var(--color-internal-border)',
                        color: 'var(--color-internal)',
                        fontSize: '0.85rem',
                        fontWeight: '600',
                      }}
                    >
                      {feat}
                    </div>
                  ))}
                </div>
              </div>

              {/* External Factors */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span className="badge badge-external">EXTERNAL Factors ({factors.classified?.EXTERNAL?.length || 0})</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {(factors.classified?.EXTERNAL || []).map((feat) => (
                    <div
                      key={feat}
                      style={{
                        padding: '0.65rem 0.85rem',
                        borderRadius: 'var(--radius-sm)',
                        background: 'var(--color-external-bg)',
                        border: '1px solid var(--color-external-border)',
                        color: 'var(--color-external)',
                        fontSize: '0.85rem',
                        fontWeight: '600',
                      }}
                    >
                      {feat}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Financial Sentiment */}
      {activeTab === 'sentiment' && (
        <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div>
            <h3 style={{ fontSize: '1.25rem' }}>Unstructured Financial Text Sentiment Classifier</h3>
            <p style={{ fontSize: '0.85rem' }}>Served directly by <code>POST /datatool/sentiment</code></p>
          </div>

          <form onSubmit={handleSentimentAnalyze} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div className="form-group">
              <label className="form-label">Headline / Financial Announcement</label>
              <textarea
                className="form-textarea"
                rows={3}
                value={headline}
                onChange={(e) => setHeadline(e.target.value)}
              />
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
              <button type="submit" className="btn btn-primary" disabled={sentimentLoading}>
                {sentimentLoading ? <span className="spinner" /> : 'Classify Sentiment'}
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setHeadline('Reliance quarterly EBITDA fell 4.5% due to soft refining margins and global macro headwinds.')}
              >
                Load Bearish Example
              </button>
            </div>
          </form>

          {sentimentResult && (
            <div
              style={{
                padding: '1.5rem',
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
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Classified Sentiment</span>
                <div style={{ marginTop: '0.25rem' }}>
                  <span
                    className="badge"
                    style={{
                      background:
                        sentimentResult.sentiment === 'POSITIVE'
                          ? 'var(--color-buy-bg)'
                          : sentimentResult.sentiment === 'NEGATIVE'
                          ? 'var(--color-sell-bg)'
                          : 'var(--color-hold-bg)',
                      color:
                        sentimentResult.sentiment === 'POSITIVE'
                          ? 'var(--color-buy)'
                          : sentimentResult.sentiment === 'NEGATIVE'
                          ? 'var(--color-sell)'
                          : 'var(--color-hold)',
                      fontSize: '0.95rem',
                      padding: '0.35rem 0.85rem',
                    }}
                  >
                    {sentimentResult.sentiment}
                  </span>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '2rem' }}>
                <div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Sentiment Score</span>
                  <div style={{ fontSize: '1.1rem', fontWeight: '700', fontFamily: 'var(--font-mono)' }}>
                    {typeof sentimentResult.score === 'number' ? sentimentResult.score.toFixed(3) : '0.000'}
                  </div>
                </div>

                <div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Category</span>
                  <div style={{ fontSize: '1rem', fontWeight: '600', color: 'var(--color-internal)' }}>
                    {sentimentResult.factor_category || 'INTERNAL'}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 4: Data Cleaning & Provenance */}
      {activeTab === 'cleaner' && (
        <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <h3 style={{ fontSize: '1.25rem' }}>Data Normalization & Provenance Audit</h3>
              <p style={{ fontSize: '0.85rem' }}>
                Cleans dirty inputs, drops corrupted null records, deduplicates chronological keys, and generates an immutable ProcessedDataBundle.
              </p>
            </div>
            <button type="button" className="btn btn-primary" onClick={handleRunSampleCleaning} disabled={cleanerLoading}>
              {cleanerLoading ? <span className="spinner" /> : 'Run Sample Cleaning Pipeline'}
            </button>
          </div>

          {cleanerResult && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              {/* Statistics Callout */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                <div style={{ padding: '1rem', borderRadius: 'var(--radius-md)', background: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-subtle)' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Input Records</span>
                  <div style={{ fontSize: '1.35rem', fontWeight: '700' }}>6 Raw Rows</div>
                </div>
                <div style={{ padding: '1rem', borderRadius: 'var(--radius-md)', background: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-subtle)' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Cleaned Structured Records</span>
                  <div style={{ fontSize: '1.35rem', fontWeight: '700', color: 'var(--color-buy)' }}>
                    {cleanerResult.records_count || cleanerResult.records?.length || 0}
                  </div>
                </div>
                <div style={{ padding: '1rem', borderRadius: 'var(--radius-md)', background: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-subtle)' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Bundle ID</span>
                  <div style={{ fontSize: '0.85rem', fontWeight: '600', fontFamily: 'var(--font-mono)', color: 'var(--color-brand)', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {cleanerResult.bundle_id || 'N/A'}
                  </div>
                </div>
              </div>

              {/* Processed Bundle Records */}
              {cleanerResult.records && cleanerResult.records.length > 0 && (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'left' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                        <th style={{ padding: '0.5rem' }}>Ticker</th>
                        <th style={{ padding: '0.5rem' }}>Timestamp</th>
                        <th style={{ padding: '0.5rem' }}>Category</th>
                        <th style={{ padding: '0.5rem' }}>Validation Status</th>
                        <th style={{ padding: '0.5rem' }}>Close Price</th>
                        <th style={{ padding: '0.5rem' }}>Volume</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cleanerResult.records.map((r, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.03)' }}>
                          <td style={{ padding: '0.5rem', fontWeight: '700' }}>{r.ticker}</td>
                          <td style={{ padding: '0.5rem', fontFamily: 'var(--font-mono)' }}>{r.timestamp}</td>
                          <td style={{ padding: '0.5rem' }}>
                            <span className="badge badge-external">{r.provenance?.data_category || 'EXTERNAL'}</span>
                          </td>
                          <td style={{ padding: '0.5rem', color: 'var(--color-buy)' }}>{r.provenance?.validation_status || 'VALID'}</td>
                          <td style={{ padding: '0.5rem', fontFamily: 'var(--font-mono)' }}>{r.features?.Close || '—'}</td>
                          <td style={{ padding: '0.5rem', fontFamily: 'var(--font-mono)' }}>{r.features?.Volume?.toLocaleString() || '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
