import React, { useState } from 'react';
import { api } from '../services/api';

export default function AuthModal({ isOpen, onClose, onSuccess }) {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (isLogin) {
        await api.login(email, password);
        onSuccess();
        onClose();
      } else {
        await api.register(username, email, password);
        // Automatically login after successful registration
        await api.login(email, password);
        onSuccess();
        onClose();
      }
    } catch (err) {
      setError(err.message || 'Authentication failed. Please check credentials.');
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = async () => {
    setError(null);
    setLoading(true);
    const demoEmail = 'analyst@stockdna.io';
    const demoPass = 'StockDNA2026!';
    try {
      try {
        await api.login(demoEmail, demoPass);
      } catch {
        // Register if not yet seeded
        await api.register('lead_analyst', demoEmail, demoPass);
        await api.login(demoEmail, demoPass);
      }
      onSuccess();
      onClose();
    } catch (err) {
      setError(err.message || 'Quick login failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        {/* Header Tabs */}
        <div style={{ display: 'flex', borderBottom: '1px solid var(--border-subtle)', marginBottom: '1.5rem' }}>
          <button
            type="button"
            onClick={() => { setIsLogin(true); setError(null); }}
            style={{
              flex: 1,
              padding: '0.75rem',
              background: 'transparent',
              border: 'none',
              borderBottom: isLogin ? '2px solid var(--color-brand)' : 'none',
              color: isLogin ? 'var(--text-primary)' : 'var(--text-secondary)',
              fontWeight: isLogin ? '700' : '500',
              cursor: 'pointer',
              fontSize: '0.9rem',
            }}
          >
            Log In
          </button>
          <button
            type="button"
            onClick={() => { setIsLogin(false); setError(null); }}
            style={{
              flex: 1,
              padding: '0.75rem',
              background: 'transparent',
              border: 'none',
              borderBottom: !isLogin ? '2px solid var(--color-brand)' : 'none',
              color: !isLogin ? 'var(--text-primary)' : 'var(--text-secondary)',
              fontWeight: !isLogin ? '700' : '500',
              cursor: 'pointer',
              fontSize: '0.9rem',
            }}
          >
            Register
          </button>
        </div>

        {error && (
          <div
            style={{
              padding: '0.65rem 0.85rem',
              marginBottom: '1rem',
              borderRadius: 'var(--radius-md)',
              background: 'var(--color-sell-bg)',
              border: '1px solid var(--color-sell-border)',
              color: 'var(--color-sell)',
              fontSize: '0.8rem',
            }}
          >
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {!isLogin && (
            <div className="form-group">
              <label className="form-label">Username</label>
              <input
                type="text"
                className="form-input"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="e.g. quant_analyst"
              />
            </div>
          )}

          <div className="form-group">
            <label className="form-label">Email Address</label>
            <input
              type="email"
              className="form-input"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="analyst@firm.com"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Password</label>
            <input
              type="password"
              className="form-input"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
            />
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1rem' }}>
            <button type="button" className="btn btn-secondary" onClick={onClose} disabled={loading}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? (
                <span className="spinner" />
              ) : isLogin ? (
                'Sign In'
              ) : (
                'Create Account'
              )}
            </button>
          </div>

          <div style={{ marginTop: '0.85rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '1rem', textAlign: 'center' }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleDemoLogin}
              disabled={loading}
              style={{ width: '100%', fontSize: '0.82rem', padding: '0.55rem', border: '1px solid var(--border-focus)' }}
            >
              ⚡ Quick 1-Click Analyst Login
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
