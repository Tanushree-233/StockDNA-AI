import React, { useState, useEffect } from 'react';
import { api, authStorage } from '../services/api';
import DecisionBadge from '../components/DecisionBadge';
import AuthModal from '../components/AuthModal';

export default function History({ onNavigate }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filterTicker, setFilterTicker] = useState('ALL');
  const [filterClass, setFilterClass] = useState('ALL');
  const [isAuthOpen, setIsAuthOpen] = useState(false);

  const isLoggedIn = authStorage.isLoggedIn();

  useEffect(() => {
    if (isLoggedIn) {
      loadHistory();
    }
  }, [isLoggedIn]);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getHistory();
      setHistory(data || []);
    } catch (err) {
      setError(err.message || 'Failed to load prediction history.');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteItem = async (id) => {
    if (!window.confirm('Delete this prediction record?')) return;
    try {
      await api.deleteHistoryItem(id);
      setHistory((prev) => prev.filter((item) => item.id !== id));
    } catch (err) {
      alert(`Error deleting item: ${err.message}`);
    }
  };

  const handleClearHistory = async () => {
    if (!window.confirm('Are you sure you want to clear your entire prediction history?')) return;
    try {
      await api.clearHistory();
      setHistory([]);
    } catch (err) {
      alert(`Error clearing history: ${err.message}`);
    }
  };

  const filteredHistory = history.filter((item) => {
    if (filterTicker !== 'ALL' && item.ticker !== filterTicker) return false;
    if (filterClass !== 'ALL' && item.prediction !== filterClass) return false;
    return true;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.35rem' }}>
            <span className="badge badge-internal">User Isolation Verified</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Scoped Relational Persistence</span>
          </div>
          <h2 style={{ fontSize: '2rem' }}>Audit History & Stored Predictions</h2>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
            Chronological audit trail of executed predictions, confidence scores, and SHAP factor attribution breakdown.
          </p>
        </div>

        {isLoggedIn && history.length > 0 && (
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button type="button" className="btn btn-secondary" onClick={loadHistory} disabled={loading}>
              Refresh
            </button>
            <button type="button" className="btn btn-outline-danger" onClick={handleClearHistory} disabled={loading}>
              Clear History
            </button>
          </div>
        )}
      </div>

      {!isLoggedIn && (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Authentication Required</h3>
          <p style={{ fontSize: '0.9rem', marginBottom: '1.5rem', color: 'var(--text-secondary)' }}>
            Prediction history is isolated per user. Please sign in or register to view your personal audit records.
          </p>
          <button type="button" className="btn btn-primary" onClick={() => setIsAuthOpen(true)}>
            Sign In / Register &rarr;
          </button>
        </div>
      )}

      {isLoggedIn && error && (
        <div
          style={{
            padding: '1rem',
            borderRadius: 'var(--radius-md)',
            background: 'var(--color-sell-bg)',
            border: '1px solid var(--color-sell-border)',
            color: 'var(--color-sell)',
          }}
        >
          {error}
        </div>
      )}

      {isLoggedIn && loading && (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
          <span className="spinner" style={{ width: '2rem', height: '2rem' }} />
          <p style={{ marginTop: '1rem' }}>Loading user prediction history...</p>
        </div>
      )}

      {isLoggedIn && !loading && (
        <>
          {/* Filters */}
          <div className="glass-panel" style={{ padding: '1.25rem 1.5rem', display: 'flex', gap: '1.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Ticker:</span>
              <select
                className="form-select"
                value={filterTicker}
                onChange={(e) => setFilterTicker(e.target.value)}
                style={{ padding: '0.35rem 0.65rem', fontSize: '0.8rem' }}
              >
                <option value="ALL">All Tickers</option>
                <option value="TCS">TCS</option>
                <option value="INFY">INFY</option>
                <option value="RELIANCE">RELIANCE</option>
              </select>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Class:</span>
              <select
                className="form-select"
                value={filterClass}
                onChange={(e) => setFilterClass(e.target.value)}
                style={{ padding: '0.35rem 0.65rem', fontSize: '0.8rem' }}
              >
                <option value="ALL">All Classes</option>
                <option value="BUY">BUY</option>
                <option value="HOLD">HOLD</option>
                <option value="SELL">SELL</option>
              </select>
            </div>

            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
              Showing {filteredHistory.length} of {history.length} predictions
            </span>
          </div>

          {/* History List */}
          {filteredHistory.length === 0 ? (
            <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
              <h4 style={{ fontSize: '1.1rem', marginBottom: '0.35rem' }}>No Prediction Records Found</h4>
              <p style={{ fontSize: '0.85rem', marginBottom: '1.25rem' }}>
                You have not executed any predictions under the current filter criteria.
              </p>
              <button type="button" className="btn btn-primary" onClick={() => onNavigate('predict')}>
                Run a Prediction Now &rarr;
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              {filteredHistory.map((record) => (
                <div
                  key={record.id}
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
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
                    <div style={{ width: '48px', height: '48px', borderRadius: 'var(--radius-md)', background: 'rgba(255, 255, 255, 0.04)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1rem', fontWeight: '800' }}>
                      {record.ticker}
                    </div>

                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                        <DecisionBadge prediction={record.prediction} />
                        <span style={{ fontSize: '0.95rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                          {(record.confidence * 100).toFixed(1)}% Confidence
                        </span>
                      </div>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {new Date(record.timestamp).toLocaleString()} &bull; v{record.model_version || '1.0.0'}
                      </span>
                    </div>
                  </div>

                  {/* XAI Attribution Summary */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                    <div style={{ textAlign: 'right' }}>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block' }}>Primary Driver</span>
                      <span
                        className={`badge ${record.primary_driver === 'INTERNAL' ? 'badge-internal' : 'badge-external'}`}
                        style={{ fontSize: '0.75rem' }}
                      >
                        {record.primary_driver || 'EXTERNAL'} ({record.internal_percentage?.toFixed(0)}% Int / {record.external_percentage?.toFixed(0)}% Ext)
                      </span>
                    </div>

                    <button
                      type="button"
                      className="btn btn-outline-danger"
                      onClick={() => handleDeleteItem(record.id)}
                      style={{ padding: '0.35rem 0.65rem', fontSize: '0.75rem' }}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        onSuccess={() => {
          setIsAuthOpen(false);
          loadHistory();
        }}
      />
    </div>
  );
}
