import React from 'react';

export default function DecisionBadge({ prediction, size = 'normal' }) {
  const norm = (prediction || '').toUpperCase();
  let badgeClass = 'badge-hold';
  let displayLabel = 'HOLD (KEEP)';

  if (norm === 'BUY') {
    badgeClass = 'badge-buy';
    displayLabel = 'BUY';
  } else if (norm === 'SELL') {
    badgeClass = 'badge-sell';
    displayLabel = 'SELL';
  } else if (norm === 'HOLD') {
    badgeClass = 'badge-hold';
    displayLabel = 'HOLD (KEEP)';
  }

  const paddingStyle = size === 'large' 
    ? { padding: '0.45rem 1.1rem', fontSize: '1rem', fontWeight: '700' }
    : { padding: '0.25rem 0.65rem', fontSize: '0.75rem', fontWeight: '600' };

  return (
    <span className={`badge ${badgeClass}`} style={paddingStyle}>
      <span style={{ 
        width: size === 'large' ? '8px' : '6px', 
        height: size === 'large' ? '8px' : '6px', 
        borderRadius: '50%', 
        backgroundColor: 'currentColor' 
      }} />
      {displayLabel}
    </span>
  );
}
