import React, { useState, useEffect, useRef } from 'react';
import { api } from '../services/api';
import { DEFAULT_UNIVERSE } from '../services/universe';

export default function Chatbot({ initialTicker = 'TCS', onNavigate }) {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'bot',
      text: `Hello! I am your **StockDNA-AI Financial Assistant**. I have deep domain intelligence across all **22 premier Indian equities** tracked on our platform.\n\nYou can ask me anything about:\n* 🏢 **Company Profiles & Business Models**\n* 🤖 **AI 5-Day Movement Predictions & Confidence**\n* 📈 **Price Trends, 52-Week Highs/Lows, and Moving Averages**\n* ⚖️ **Side-by-Side Stock Comparisons** (e.g., *Compare TCS vs Infosys*)\n* 💡 **Plain English Explanations** of any financial indicator or why the AI recommended a particular rating\n\nHow can I help you today?`,
      suggestedQuestions: [
        'What is the AI prediction for Reliance?',
        'Compare TCS vs Infosys',
        'Tell me about Sun Pharma',
        'Explain what India VIX means in simple words',
      ],
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);

  const [input, setInput] = useState('');
  const [contextTicker, setContextTicker] = useState(initialTicker);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSendMessage = async (queryText = null) => {
    const textToSend = (queryText || input).trim();
    if (!textToSend || loading) return;

    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await api.sendChatMessage(textToSend, contextTicker);

      if (res.detected_ticker) {
        setContextTicker(res.detected_ticker);
      }

      const botMsg = {
        id: Date.now() + 1,
        sender: 'bot',
        text: res.reply,
        detectedTicker: res.detected_ticker,
        suggestedQuestions: res.suggested_questions || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      const errorMsg = {
        id: Date.now() + 1,
        sender: 'bot',
        text: `⚠️ **Connection Error:** Could not connect to intelligence backend: ${err.message || 'Please verify server is running.'}`,
        suggestedQuestions: ['What is the prediction for TCS?', 'Show all 22 tracked stocks'],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  // Helper to format basic markdown (bold, lists, tables)
  const renderFormattedText = (rawText) => {
    const lines = rawText.split('\n');
    const elements = [];
    let inTable = false;
    let tableRows = [];

    const flushTable = (key) => {
      if (tableRows.length > 0) {
        const header = tableRows[0];
        const bodyRows = tableRows.slice(2); // Skip separator row
        elements.push(
          <div key={`table-${key}`} style={{ overflowX: 'auto', margin: '0.8rem 0' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
              <thead>
                <tr style={{ background: 'rgba(255, 255, 255, 0.05)', borderBottom: '1px solid var(--border-subtle)' }}>
                  {header.map((th, idx) => (
                    <th key={idx} style={{ padding: '0.5rem 0.75rem', textAlign: 'left', fontWeight: '700', color: 'white' }}>
                      {th.replace(/\*\*/g, '')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {bodyRows.map((row, rIdx) => (
                  <tr key={rIdx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.03)' }}>
                    {row.map((cell, cIdx) => (
                      <td key={cIdx} style={{ padding: '0.45rem 0.75rem', color: 'var(--text-secondary)' }}>
                        {cell.replace(/\*\*/g, '')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        tableRows = [];
      }
      inTable = false;
    };

    lines.forEach((line, idx) => {
      const trimmed = line.trim();

      // Table row detection
      if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
        inTable = true;
        const cells = trimmed
          .split('|')
          .slice(1, -1)
          .map((c) => c.trim());
        tableRows.push(cells);
        return;
      } else if (inTable) {
        flushTable(idx);
      }

      // Headers
      if (trimmed.startsWith('### ')) {
        elements.push(
          <h4 key={idx} style={{ fontSize: '1.1rem', color: 'white', marginTop: '0.75rem', marginBottom: '0.35rem' }}>
            {trimmed.replace('### ', '')}
          </h4>
        );
        return;
      }
      if (trimmed.startsWith('#### ')) {
        elements.push(
          <h5 key={idx} style={{ fontSize: '0.95rem', color: 'var(--color-brand)', marginTop: '0.6rem', marginBottom: '0.25rem' }}>
            {trimmed.replace('#### ', '')}
          </h5>
        );
        return;
      }

      // Bullet points
      if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
        const bulletText = trimmed.substring(2);
        elements.push(
          <div key={idx} style={{ display: 'flex', gap: '0.5rem', margin: '0.25rem 0', fontSize: '0.88rem', lineHeight: '1.5' }}>
            <span style={{ color: 'var(--color-brand)' }}>•</span>
            <span>{renderInlineMarkdown(bulletText)}</span>
          </div>
        );
        return;
      }

      // Blockquotes
      if (trimmed.startsWith('> ')) {
        elements.push(
          <div
            key={idx}
            style={{
              padding: '0.5rem 0.85rem',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(99, 102, 241, 0.08)',
              borderLeft: '3px solid var(--color-brand)',
              fontSize: '0.82rem',
              color: 'var(--text-secondary)',
              margin: '0.5rem 0',
            }}
          >
            {renderInlineMarkdown(trimmed.replace('> ', ''))}
          </div>
        );
        return;
      }

      // Dividers
      if (trimmed === '---') {
        elements.push(
          <hr key={idx} style={{ border: 'none', borderTop: '1px solid var(--border-subtle)', margin: '0.85rem 0' }} />
        );
        return;
      }

      // Normal paragraph
      if (trimmed.length > 0) {
        elements.push(
          <p key={idx} style={{ margin: '0.4rem 0', fontSize: '0.88rem', lineHeight: '1.6', color: 'var(--text-primary)' }}>
            {renderInlineMarkdown(trimmed)}
          </p>
        );
      }
    });

    if (inTable) {
      flushTable('end');
    }

    return elements;
  };

  // Inline formatting for **bold**, `code`, etc.
  const renderInlineMarkdown = (text) => {
    const parts = text.split(/(\*\*.*?\*\*|`.*?`)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i} style={{ color: 'white' }}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        const content = part.slice(1, -1);
        let badgeColor = 'var(--text-primary)';
        let badgeBg = 'rgba(255, 255, 255, 0.08)';

        if (content === 'BUY') {
          badgeColor = 'var(--color-buy)';
          badgeBg = 'rgba(34, 197, 94, 0.15)';
        } else if (content === 'HOLD') {
          badgeColor = 'var(--color-hold)';
          badgeBg = 'rgba(245, 158, 11, 0.15)';
        } else if (content === 'SELL') {
          badgeColor = 'var(--color-sell)';
          badgeBg = 'rgba(239, 68, 68, 0.15)';
        }

        return (
          <code
            key={i}
            style={{
              padding: '0.15rem 0.4rem',
              borderRadius: 'var(--radius-sm)',
              background: badgeBg,
              color: badgeColor,
              fontFamily: 'var(--font-mono)',
              fontWeight: '700',
              fontSize: '0.82rem',
            }}
          >
            {content}
          </code>
        );
      }
      return part;
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', height: 'calc(100vh - 180px)', minHeight: '620px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.2rem' }}>
            <span className="badge badge-brand">AI Copilot</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>22-Company Financial Intelligence</span>
          </div>
          <h2 style={{ fontSize: '1.8rem', margin: 0 }}>Company Intelligence AI Chatbot</h2>
        </div>

        {/* Active Context Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Focused Stock:</span>
          <select
            className="form-select"
            value={contextTicker}
            onChange={(e) => setContextTicker(e.target.value)}
            style={{ padding: '0.35rem 0.75rem', fontSize: '0.82rem' }}
          >
            {DEFAULT_UNIVERSE.map((s) => (
              <option key={s.ticker} value={s.ticker}>
                {s.ticker} — {s.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Chat Conversation Container */}
      <div
        className="glass-panel"
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          padding: '0',
          border: '1px solid var(--border-subtle)',
        }}
      >
        {/* Messages Scroll Area */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.25rem',
          }}
        >
          {messages.map((msg) => {
            const isUser = msg.sender === 'user';

            return (
              <div
                key={msg.id}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: isUser ? 'flex-end' : 'flex-start',
                  gap: '0.35rem',
                }}
              >
                {/* Sender badge & timestamp */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.72rem', color: 'var(--text-muted)', padding: '0 0.25rem' }}>
                  <span>{isUser ? 'You' : 'StockDNA AI Assistant'}</span>
                  <span>&bull;</span>
                  <span>{msg.timestamp}</span>
                </div>

                {/* Message Bubble */}
                <div
                  style={{
                    maxWidth: '82%',
                    padding: '1.15rem 1.35rem',
                    borderRadius: isUser ? '16px 16px 2px 16px' : '16px 16px 16px 2px',
                    background: isUser
                      ? 'var(--color-brand)'
                      : 'rgba(255, 255, 255, 0.04)',
                    border: isUser ? 'none' : '1px solid var(--border-subtle)',
                    color: 'white',
                    boxShadow: isUser ? '0 4px 12px rgba(99, 102, 241, 0.25)' : 'none',
                  }}
                >
                  {isUser ? (
                    <div style={{ fontSize: '0.92rem', lineHeight: '1.5' }}>{msg.text}</div>
                  ) : (
                    <div>{renderFormattedText(msg.text)}</div>
                  )}

                  {/* Actions for detected company */}
                  {msg.detectedTicker && onNavigate && (
                    <div
                      style={{
                        marginTop: '0.85rem',
                        paddingTop: '0.65rem',
                        borderTop: '1px solid rgba(255, 255, 255, 0.08)',
                        display: 'flex',
                        gap: '0.65rem',
                        flexWrap: 'wrap',
                      }}
                    >
                      <button
                        type="button"
                        onClick={() => onNavigate('trends', { ticker: msg.detectedTicker })}
                        style={{
                          background: 'rgba(255, 255, 255, 0.08)',
                          color: 'white',
                          border: 'none',
                          borderRadius: 'var(--radius-sm)',
                          padding: '0.3rem 0.7rem',
                          fontSize: '0.75rem',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.35rem',
                        }}
                      >
                        📈 Open {msg.detectedTicker} Price Trends & Charts
                      </button>
                      <button
                        type="button"
                        onClick={() => onNavigate('predict', { ticker: msg.detectedTicker })}
                        style={{
                          background: 'rgba(99, 102, 241, 0.2)',
                          color: 'var(--color-brand-light)',
                          border: '1px solid rgba(99, 102, 241, 0.4)',
                          borderRadius: 'var(--radius-sm)',
                          padding: '0.3rem 0.7rem',
                          fontSize: '0.75rem',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.35rem',
                        }}
                      >
                        🤖 Run {msg.detectedTicker} AI Prediction
                      </button>
                    </div>
                  )}
                </div>

                {/* Suggested Questions Quick Chips */}
                {!isUser && msg.suggestedQuestions && msg.suggestedQuestions.length > 0 && (
                  <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginTop: '0.4rem', maxWidth: '85%' }}>
                    {msg.suggestedQuestions.map((sq, i) => (
                      <button
                        key={i}
                        type="button"
                        onClick={() => handleSendMessage(sq)}
                        disabled={loading}
                        style={{
                          background: 'rgba(255, 255, 255, 0.03)',
                          border: '1px solid var(--border-subtle)',
                          borderRadius: '16px',
                          color: 'var(--text-secondary)',
                          fontSize: '0.75rem',
                          padding: '0.3rem 0.75rem',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease',
                          textAlign: 'left',
                        }}
                        onMouseEnter={(e) => {
                          e.target.style.background = 'rgba(99, 102, 241, 0.15)';
                          e.target.style.color = 'white';
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.background = 'rgba(255, 255, 255, 0.03)';
                          e.target.style.color = 'var(--text-secondary)';
                        }}
                      >
                        💡 {sq}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}

          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              <span className="spinner" style={{ width: '1.25rem', height: '1.25rem' }} />
              <span>Analyzing company fundamentals, price series, and active AI model...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div
          style={{
            padding: '1rem 1.25rem',
            borderTop: '1px solid var(--border-subtle)',
            background: 'rgba(255, 255, 255, 0.015)',
            display: 'flex',
            gap: '0.75rem',
            alignItems: 'center',
          }}
        >
          <input
            type="text"
            className="form-control"
            placeholder={`Ask anything about ${contextTicker} or any of the 22 companies (e.g. "What is the AI prediction for ${contextTicker}?", "Compare TCS vs Infosys")...`}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSendMessage();
            }}
            disabled={loading}
            style={{
              flex: 1,
              padding: '0.75rem 1rem',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid var(--border-subtle)',
              color: 'white',
              fontSize: '0.9rem',
            }}
          />

          <button
            type="button"
            className="btn btn-primary"
            onClick={() => handleSendMessage()}
            disabled={loading || !input.trim()}
            style={{
              padding: '0.75rem 1.25rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              fontSize: '0.9rem',
            }}
          >
            {loading ? <span className="spinner" /> : <span>Send &rarr;</span>}
          </button>
        </div>
      </div>
    </div>
  );
}

