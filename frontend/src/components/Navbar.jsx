import React, { useState } from 'react';
import { authStorage } from '../services/api';
import AuthModal from './AuthModal';

export default function Navbar({ activePage, onNavigate }) {
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const isLoggedIn = authStorage.isLoggedIn();
  const username = authStorage.getUser();

  const handleLogout = () => {
    authStorage.clearToken();
    window.location.reload();
  };

  const navItems = [
    { id: 'home', label: 'Dashboard' },
    { id: 'predict', label: 'Stock Prediction' },
    { id: 'xai', label: 'Explainability' },
    { id: 'datatool', label: 'Data Studio' },
    { id: 'history', label: 'History' },
    { id: 'methodology', label: 'Methodology' },
  ];

  return (
    <>
      <header className="navbar">
        <div className="nav-container">
          {/* Brand Logo */}
          <div className="nav-brand" onClick={() => onNavigate('home')}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--color-brand)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 12h5l3-8 4 16 3-8h5" />
            </svg>
            <span>StockDNA<span style={{ color: 'var(--color-brand)' }}>-AI</span></span>
            <span className="brand-badge">XAI Platform</span>
          </div>

          {/* Navigation Links */}
          <nav className="nav-links">
            {navItems.map((item) => (
              <button
                key={item.id}
                className={`nav-link ${activePage === item.id ? 'active' : ''}`}
                onClick={() => onNavigate(item.id)}
              >
                {item.label}
              </button>
            ))}
          </nav>

          {/* Auth Controls */}
          <div className="nav-auth">
            {isLoggedIn ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Analyst: <strong style={{ color: 'var(--text-primary)' }}>{username}</strong>
                </span>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleLogout}
                  style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem' }}
                >
                  Sign Out
                </button>
              </div>
            ) : (
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => setIsAuthOpen(true)}
                style={{ padding: '0.45rem 0.95rem', fontSize: '0.8rem' }}
              >
                Sign In / Register
              </button>
            )}
          </div>
        </div>
      </header>

      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        onSuccess={() => {
          setIsAuthOpen(false);
          window.location.reload();
        }}
      />
    </>
  );
}
