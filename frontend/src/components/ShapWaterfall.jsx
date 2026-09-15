import React, { useState } from 'react';

export default function ShapWaterfall({
  contributions = [],
  predictedClass = 'HOLD',
  maxItems = 15,
}) {
  const [filterGroup, setFilterGroup] = useState('ALL'); // 'ALL' | 'INTERNAL' | 'EXTERNAL'

  if (!contributions || contributions.length === 0) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        No feature contributions available for this prediction.
      </div>
    );
  }

  // Filter features
  const filtered = contributions.filter((c) => {
    if (filterGroup === 'ALL') return true;
    return c.group === filterGroup;
  });

  // Sort by absolute importance descending
  const sorted = [...filtered]
    .sort((a, b) => (b.importance || Math.abs(b.shap_value)) - (a.importance || Math.abs(a.shap_value)))
    .slice(0, maxItems);

  // Find max absolute SHAP value for scaling bars
  const maxAbs = Math.max(...sorted.map((c) => Math.abs(c.shap_value || 0)), 0.001);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Controls / Filter Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h4 style={{ fontSize: '1rem', color: 'var(--text-primary)' }}>
            Bi-directional Feature Contributions
          </h4>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            SHAP values for predicted class: <strong style={{ color: 'var(--text-primary)' }}>{predictedClass}</strong>
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.35rem', background: 'rgba(255, 255, 255, 0.05)', padding: '0.2rem', borderRadius: 'var(--radius-md)' }}>
          {['ALL', 'INTERNAL', 'EXTERNAL'].map((grp) => (
            <button
              key={grp}
              onClick={() => setFilterGroup(grp)}
              style={{
                background: filterGroup === grp ? 'var(--color-brand)' : 'transparent',
                color: filterGroup === grp ? 'white' : 'var(--text-secondary)',
                border: 'none',
                borderRadius: 'var(--radius-sm)',
                padding: '0.25rem 0.65rem',
                fontSize: '0.75rem',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {grp}
            </button>
          ))}
        </div>
      </div>

      {/* Zero Center Line Header */}
      <div style={{ display: 'grid', gridTemplateColumns: '180px 1fr', gap: '1rem', alignItems: 'center' }}>
        <span style={{ fontSize: '0.75rem', fontWeight: '600', color: 'var(--text-muted)' }}>Feature / Group</span>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', position: 'relative' }}>
          <span>&larr; Moderates / Away from {predictedClass} (-SHAP)</span>
          <span style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', fontWeight: '700', color: 'var(--text-secondary)' }}>0</span>
          <span>Reinforces {predictedClass} (+SHAP) &rarr;</span>
        </div>
      </div>

      {/* Feature Contribution Bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
        {sorted.map((item, idx) => {
          const val = item.shap_value || 0;
          const isPositive = val >= 0;
          const widthPct = Math.min(50, (Math.abs(val) / maxAbs) * 50);

          return (
            <div
              key={item.feature || idx}
              style={{
                display: 'grid',
                gridTemplateColumns: '180px 1fr',
                gap: '1rem',
                alignItems: 'center',
                padding: '0.5rem 0.65rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'rgba(255, 255, 255, 0.02)',
                border: '1px solid rgba(255, 255, 255, 0.04)',
              }}
            >
              {/* Feature Name & Group Badge */}
              <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                <span
                  style={{
                    fontSize: '0.82rem',
                    fontWeight: '600',
                    color: 'var(--text-primary)',
                    whiteSpace: 'nowrap',
                    textOverflow: 'ellipsis',
                    overflow: 'hidden',
                  }}
                  title={item.feature}
                >
                  {item.feature}
                </span>
                <span
                  style={{
                    fontSize: '0.65rem',
                    fontWeight: '700',
                    color: item.group === 'INTERNAL' ? 'var(--color-internal)' : 'var(--color-external)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.04em',
                  }}
                >
                  {item.group}
                </span>
              </div>

              {/* Bi-directional Bar Area */}
              <div
                style={{
                  height: '24px',
                  display: 'flex',
                  alignItems: 'center',
                  position: 'relative',
                  backgroundColor: 'rgba(255, 255, 255, 0.03)',
                  borderRadius: 'var(--radius-sm)',
                  overflow: 'hidden',
                }}
              >
                {/* Center zero vertical marker */}
                <div
                  style={{
                    position: 'absolute',
                    left: '50%',
                    top: 0,
                    bottom: 0,
                    width: '1px',
                    backgroundColor: 'rgba(255, 255, 255, 0.25)',
                    zIndex: 2,
                  }}
                />

                {/* Bar */}
                {isPositive ? (
                  <div
                    style={{
                      position: 'absolute',
                      left: '50%',
                      width: `${widthPct}%`,
                      height: '14px',
                      backgroundColor: item.group === 'INTERNAL' ? 'var(--color-internal)' : 'var(--color-buy)',
                      borderRadius: '0 3px 3px 0',
                      transition: 'width 0.4s ease',
                    }}
                  />
                ) : (
                  <div
                    style={{
                      position: 'absolute',
                      right: '50%',
                      width: `${widthPct}%`,
                      height: '14px',
                      backgroundColor: 'var(--color-sell)',
                      borderRadius: '3px 0 0 3px',
                      transition: 'width 0.4s ease',
                    }}
                  />
                )}

                {/* Value Label */}
                <span
                  style={{
                    position: 'absolute',
                    right: isPositive ? '8px' : 'auto',
                    left: !isPositive ? '8px' : 'auto',
                    fontSize: '0.72rem',
                    fontWeight: '600',
                    fontFamily: 'var(--font-mono)',
                    color: isPositive ? 'var(--color-buy)' : 'var(--color-sell)',
                    zIndex: 3,
                  }}
                >
                  {val > 0 ? `+${val.toFixed(4)}` : val.toFixed(4)}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
        <strong>Interpretation Note:</strong> SHAP values are calculated via TreeExplainer for the target class log-odds margin. Positive values push the model toward the predicted class; negative values push the model away.
      </div>
    </div>
  );
}
