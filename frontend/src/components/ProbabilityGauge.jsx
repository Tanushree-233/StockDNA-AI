import React from 'react';

export default function ProbabilityGauge({ probabilities = {} }) {
  const sellProb = probabilities.SELL ?? 0;
  const holdProb = probabilities.HOLD ?? 0;
  const buyProb = probabilities.BUY ?? 0;

  const sellPct = (sellProb * 100).toFixed(1);
  const holdPct = (holdProb * 100).toFixed(1);
  const buyPct = (buyProb * 100).toFixed(1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: '600', color: 'var(--text-secondary)' }}>
          Model Probability Distribution
        </span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Softmax probabilities sum to 100%
        </span>
      </div>

      {/* Tri-color Segmented Progress Bar */}
      <div
        style={{
          width: '100%',
          height: '14px',
          borderRadius: 'var(--radius-full)',
          overflow: 'hidden',
          display: 'flex',
          backgroundColor: 'rgba(255, 255, 255, 0.05)',
          border: '1px solid var(--border-subtle)',
        }}
      >
        <div
          style={{
            width: `${sellPct}%`,
            backgroundColor: 'var(--color-sell)',
            transition: 'width 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
          }}
          title={`SELL: ${sellPct}%`}
        />
        <div
          style={{
            width: `${holdPct}%`,
            backgroundColor: 'var(--color-hold)',
            transition: 'width 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
          }}
          title={`HOLD: ${holdPct}%`}
        />
        <div
          style={{
            width: `${buyPct}%`,
            backgroundColor: 'var(--color-buy)',
            transition: 'width 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
          }}
          title={`BUY: ${buyPct}%`}
        />
      </div>

      {/* Individual Class Probability Pills */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
        <div
          style={{
            padding: '0.65rem 0.85rem',
            borderRadius: 'var(--radius-md)',
            background: 'var(--color-sell-bg)',
            border: '1px solid var(--color-sell-border)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.2rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: '600', color: 'var(--color-sell)' }}>
              SELL
            </span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>&le; -2.0%</span>
          </div>
          <span style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--text-primary)', fontFamily: 'var(--font-display)' }}>
            {sellPct}%
          </span>
        </div>

        <div
          style={{
            padding: '0.65rem 0.85rem',
            borderRadius: 'var(--radius-md)',
            background: 'var(--color-hold-bg)',
            border: '1px solid var(--color-hold-border)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.2rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: '600', color: 'var(--color-hold)' }}>
              HOLD (KEEP)
            </span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>&plusmn;2.0%</span>
          </div>
          <span style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--text-primary)', fontFamily: 'var(--font-display)' }}>
            {holdPct}%
          </span>
        </div>

        <div
          style={{
            padding: '0.65rem 0.85rem',
            borderRadius: 'var(--radius-md)',
            background: 'var(--color-buy-bg)',
            border: '1px solid var(--color-buy-border)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.2rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: '600', color: 'var(--color-buy)' }}>
              BUY
            </span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>&ge; +2.0%</span>
          </div>
          <span style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--text-primary)', fontFamily: 'var(--font-display)' }}>
            {buyPct}%
          </span>
        </div>
      </div>
    </div>
  );
}
