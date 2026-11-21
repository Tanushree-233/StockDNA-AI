import React, { useState, useRef, useEffect } from 'react';
import { api } from '../services/api';

export default function FloatingChatbot({ currentTicker = 'TCS', onNavigate }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'bot',
      text: `Hi! Need quick insights on **${currentTicker}** or any of our 22 tracked companies? Ask me anything below!`,
      suggestedQuestions: [
        `What is the prediction for ${currentTicker}?`,
        `Show price trends for ${currentTicker}`,
      ],
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen]);

  const handleSend = async (customText = null) => {
    const textToSend = (customText || input).trim();
    if (!textToSend || loading) return;

    setMessages((prev) => [
      ...prev,
      { id: Date.now(), sender: 'user', text: textToSend },
    ]);
    setInput('');
    setLoading(true);

    try {
      const res = await api.sendChatMessage(textToSend, currentTicker);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: 'bot',
          text: res.reply,
          suggestedQuestions: res.suggested_questions?.slice(0, 2) || [],
          detectedTicker: res.detected_ticker,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: 'bot',
          text: 'Unable to connect to intelligence backend. Please try again.',
          suggestedQuestions: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Floating Toggle Button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          zIndex: 9999,
          width: '56px',
          height: '56px',
          borderRadius: '50%',
          background: 'var(--color-brand)',
          color: 'white',
          border: 'none',
          boxShadow: '0 8px 24px rgba(99, 102, 241, 0.45)',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '1.4rem',
          transition: 'transform 0.2s ease',
        }}
        title="Open AI Financial Assistant"
        onMouseEnter={(e) => (e.currentTarget.style.transform = 'scale(1.08)')}
        onMouseLeave={(e) => (e.currentTarget.style.transform = 'scale(1)')}
      >
        {isOpen ? '✕' : '💬'}
      </button>

      {/* Floating Chat Drawer */}
      {isOpen && (
        <div
          style={{
            position: 'fixed',
            bottom: '90px',
            right: '24px',
            zIndex: 9998,
            width: '380px',
            maxWidth: 'calc(100vw - 32px)',
            height: '500px',
            borderRadius: 'var(--radius-lg)',
            background: 'var(--color-bg-panel)',
            border: '1px solid var(--border-subtle)',
            boxShadow: '0 16px 40px rgba(0, 0, 0, 0.6)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          {/* Drawer Header */}
          <div
            style={{
              padding: '0.85rem 1.15rem',
              background: 'rgba(255, 255, 255, 0.03)',
              borderBottom: '1px solid var(--border-subtle)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ fontSize: '1.1rem' }}>🤖</span>
              <div>
                <strong style={{ fontSize: '0.85rem', color: 'white', display: 'block' }}>
                  StockDNA AI Assistant
                </strong>
                <span style={{ fontSize: '0.7rem', color: 'var(--color-brand-light)' }}>
                  Active context: {currentTicker}
                </span>
              </div>
            </div>

            <button
              type="button"
              onClick={() => onNavigate && onNavigate('chatbot', { ticker: currentTicker })}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-muted)',
                fontSize: '0.75rem',
                cursor: 'pointer',
                textDecoration: 'underline',
              }}
            >
              Full Screen &rarr;
            </button>
          </div>

          {/* Messages Scroll Area */}
          <div
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: '1rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.85rem',
            }}
          >
            {messages.map((m) => {
              const isUser = m.sender === 'user';
              return (
                <div
                  key={m.id}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: isUser ? 'flex-end' : 'flex-start',
                    gap: '0.25rem',
                  }}
                >
                  <div
                    style={{
                      maxWidth: '85%',
                      padding: '0.65rem 0.85rem',
                      borderRadius: isUser ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
                      background: isUser ? 'var(--color-brand)' : 'rgba(255, 255, 255, 0.05)',
                      border: isUser ? 'none' : '1px solid var(--border-subtle)',
                      fontSize: '0.82rem',
                      lineHeight: '1.45',
                      color: 'white',
                    }}
                  >
                    {m.text}
                  </div>

                  {!isUser && m.suggestedQuestions && m.suggestedQuestions.length > 0 && (
                    <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', marginTop: '0.25rem' }}>
                      {m.suggestedQuestions.map((sq, i) => (
                        <button
                          key={i}
                          type="button"
                          onClick={() => handleSend(sq)}
                          style={{
                            background: 'rgba(255, 255, 255, 0.04)',
                            border: '1px solid var(--border-subtle)',
                            borderRadius: '12px',
                            color: 'var(--text-secondary)',
                            fontSize: '0.7rem',
                            padding: '0.2rem 0.55rem',
                            cursor: 'pointer',
                          }}
                        >
                          {sq}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}

            {loading && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                <span className="spinner" style={{ width: '1rem', height: '1rem' }} />
                <span>Thinking...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Drawer Input */}
          <div
            style={{
              padding: '0.75rem',
              borderTop: '1px solid var(--border-subtle)',
              background: 'rgba(255, 255, 255, 0.02)',
              display: 'flex',
              gap: '0.5rem',
            }}
          >
            <input
              type="text"
              placeholder={`Ask about ${currentTicker}...`}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSend();
              }}
              disabled={loading}
              style={{
                flex: 1,
                padding: '0.55rem 0.75rem',
                borderRadius: 'var(--radius-sm)',
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid var(--border-subtle)',
                color: 'white',
                fontSize: '0.82rem',
              }}
            />
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => handleSend()}
              disabled={loading || !input.trim()}
              style={{ padding: '0.55rem 0.85rem', fontSize: '0.8rem' }}
            >
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
}

