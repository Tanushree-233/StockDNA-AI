import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Predict from './pages/Predict';
import Explainability from './pages/Explainability';
import DataTool from './pages/DataTool';
import History from './pages/History';
import Methodology from './pages/Methodology';

export default function App() {
  const [activePage, setActivePage] = useState('home');
  const [selectedTicker, setSelectedTicker] = useState('TCS');
  const [latestPrediction, setLatestPrediction] = useState(null);

  const handleNavigate = (page, params = {}) => {
    setActivePage(page);
    if (params.ticker) {
      setSelectedTicker(params.ticker);
    }
    if (params.prediction) {
      setLatestPrediction(params.prediction);
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handlePredictionSuccess = (predictionData) => {
    setLatestPrediction(predictionData);
  };

  return (
    <div className="app-container">
      <Navbar activePage={activePage} onNavigate={handleNavigate} />

      <main className="main-content">
        {activePage === 'home' && <Home onNavigate={handleNavigate} />}
        {activePage === 'predict' && (
          <Predict
            initialTicker={selectedTicker}
            onNavigate={handleNavigate}
            onPredictionSuccess={handlePredictionSuccess}
          />
        )}
        {activePage === 'xai' && (
          <Explainability
            initialPrediction={latestPrediction}
            onNavigate={handleNavigate}
          />
        )}
        {activePage === 'datatool' && <DataTool />}
        {activePage === 'history' && <History onNavigate={handleNavigate} />}
        {activePage === 'methodology' && <Methodology />}
      </main>

      <footer className="footer">
        <div className="footer-content">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: '700' }}>
            <span>StockDNA-AI Platform</span>
            <span style={{ color: 'var(--text-muted)' }}>&bull;</span>
            <span style={{ color: 'var(--text-secondary)' }}>Production ML & Explainable AI</span>
          </div>
          <p style={{ maxWidth: '680px', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Developed as an academic decision-support framework. Evaluated on out-of-sample NSE equities with point-in-time embargoes. Not investment advice or an automated trading recommendation.
          </p>
        </div>
      </footer>
    </div>
  );
}
