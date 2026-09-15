import React from 'react';

export default function AttributionDonut({
  internalPercentage = 0,
  externalPercentage = 0,
  primaryDriver = 'EXTERNAL',
  size = 220,
}) {
  const intVal = Math.max(0, Math.min(100, Number(internalPercentage) || 0));
  const extVal = Math.max(0, Math.min(100, Number(externalPercentage) || 0));

  const strokeWidth = 24;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  const intOffset = circumference - (intVal / 100) * circumference;
  const extOffset = circumference - (extVal / 100) * circumference;

  // External stroke starts at top (-90deg), Internal starts after External
  const extDeg = -90;
  const intDeg = -90 + (extVal / 100) * 360;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '1.25rem',
      }}
    >
      <div style={{ position: 'relative', width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          {/* Background Ring */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="transparent"
            stroke="rgba(255, 255, 255, 0.05)"
            strokeWidth={strokeWidth}
          />

          {/* External Arc (Purple) */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="transparent"
            stroke="var(--color-external)"
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={extOffset}
            strokeLinecap="round"
            transform={`rotate(${extDeg} ${size / 2} ${size / 2})`}
            style={{ transition: 'stroke-dashoffset 0.8s ease' }}
          />

          {/* Internal Arc (Cyan) */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="transparent"
            stroke="var(--color-internal)"
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={intOffset}
            strokeLinecap="round"
            transform={`rotate(${intDeg} ${size / 2} ${size / 2})`}
            style={{ transition: 'stroke-dashoffset 0.8s ease' }}
          />
        </svg>

        {/* Center Label */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            pointerEvents: 'none',
          }}
        >
          <span style={{ fontSize: '0.75rem', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Primary Driver
          </span>
          <span
            style={{
              fontSize: '1.35rem',
              fontWeight: '800',
              fontFamily: 'var(--font-display)',
              color: primaryDriver === 'INTERNAL' ? 'var(--color-internal)' : 'var(--color-external)',
            }}
          >
            {primaryDriver}
          </span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            {primaryDriver === 'INTERNAL' ? `${intVal.toFixed(1)}%` : `${extVal.toFixed(1)}%`}
          </span>
        </div>
      </div>

      {/* Legend & Exact Values */}
      <div style={{ display: 'flex', gap: '1.5rem', width: '100%', justifyContent: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span
            style={{
              width: '12px',
              height: '12px',
              borderRadius: '3px',
              backgroundColor: 'var(--color-internal)',
            }}
          />
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>INTERNAL (Company)</span>
            <span style={{ fontSize: '1.1rem', fontWeight: '700', color: 'var(--color-internal)' }}>
              {intVal.toFixed(1)}%
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span
            style={{
              width: '12px',
              height: '12px',
              borderRadius: '3px',
              backgroundColor: 'var(--color-external)',
            }}
          />
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>EXTERNAL (Market)</span>
            <span style={{ fontSize: '1.1rem', fontWeight: '700', color: 'var(--color-external)' }}>
              {extVal.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      <div
        style={{
          fontSize: '0.75rem',
          color: 'var(--text-muted)',
          textAlign: 'center',
          maxWidth: '320px',
          lineHeight: '1.4',
        }}
      >
        Percentage of the model's absolute SHAP attribution for this prediction (Internal + External = 100%).
      </div>
    </div>
  );
}
