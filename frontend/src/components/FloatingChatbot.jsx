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
            width: '390px',
            maxWidth: 'calc(100vw - 32px)',
            height: '520px',
            borderRadius: 'var(--radius-lg)',
            backgroundColor: '#0B101E',
            background: '#0B101E',
            border: '1px solid rgba(99, 102, 241, 0.35)',
            boxShadow: '0 24px 60px rgba(0, 0, 0, 0.95), 0 4px 20px rgba(99, 102, 241, 0.25)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          {/* Drawer Header */}
          <div
            style={{
              padding: '0.9rem 1.15rem',
              backgroundColor: '#11182B',
              background: '#11182B',
              borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <div
                style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  background: 'var(--color-brand)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '1rem',
                }}
              >
                🤖
              </div>
              <div>
                <strong style={{ fontSize: '0.88rem', color: '#F8FAFC', display: 'block' }}>
                  StockDNA AI Assistant
                </strong>
                <span style={{ fontSize: '0.72rem', color: 'var(--color-brand-light)' }}>
                  Active stock context: <strong style={{ color: 'white' }}>{currentTicker}</strong>
                </span>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <button
                type="button"
                onClick={() => onNavigate && onNavigate('chatbot', { ticker: currentTicker })}
                style={{
                  background: 'rgba(255, 255, 255, 0.08)',
                  border: '1px solid rgba(255, 255, 255, 0.15)',
                  borderRadius: 'var(--radius-sm)',
                  color: 'var(--text-secondary)',
                  fontSize: '0.72rem',
                  padding: '0.25rem 0.55rem',
                  cursor: 'pointer',
                }}
              >
                Full Page &rarr;
              </button>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-muted)',
                  fontSize: '1.1rem',
                  cursor: 'pointer',
                  padding: '0 0.25rem',
                }}
              >
                ✕
              </button>
            </div>
          </div>

          {/* Messages Scroll Area */}
          <div
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: '1.15rem',
              backgroundColor: '#0B101E',
              background: '#0B101E',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.95rem',
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
                      padding: '0.75rem 0.95rem',
                      borderRadius: isUser ? '14px 14px 2px 14px' : '14px 14px 14px 2px',
                      backgroundColor: isUser ? 'var(--color-brand)' : '#151D33',
                      background: isUser ? 'var(--color-brand)' : '#151D33',
                      border: isUser ? 'none' : '1px solid rgba(255, 255, 255, 0.09)',
                      fontSize: '0.84rem',
                      lineHeight: '1.5',
                      color: '#F8FAFC',
                      boxShadow: isUser
                        ? '0 4px 12px rgba(99, 102, 241, 0.3)'
                        : '0 4px 12px rgba(0, 0, 0, 0.35)',
                    }}
                  >
                    {m.text}
                  </div>

                  {!isUser && m.suggestedQuestions && m.suggestedQuestions.length > 0 && (
                    <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', marginTop: '0.3rem' }}>
                      {m.suggestedQuestions.map((sq, i) => (
                        <button
                          key={i}
                          type="button"
                          onClick={() => handleSend(sq)}
                          style={{
                            backgroundColor: '#151D33',
                            background: '#151D33',
                            border: '1px solid rgba(99, 102, 241, 0.3)',
                            borderRadius: '12px',
                            color: 'var(--text-secondary)',
                            fontSize: '0.72rem',
                            padding: '0.25rem 0.6rem',
                            cursor: 'pointer',
                            transition: 'all 0.15s ease',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = 'rgba(99, 102, 241, 0.25)';
                            e.currentTarget.style.color = 'white';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = '#151D33';
                            e.currentTarget.style.color = 'var(--text-secondary)';
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
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.75rem', padding: '0.25rem' }}>
                <span className="spinner" style={{ width: '1rem', height: '1rem' }} />
                <span>AI analyzing...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Drawer Input */}
          <div
            style={{
              padding: '0.85rem 1rem',
              borderTop: '1px solid rgba(255, 255, 255, 0.1)',
              backgroundColor: '#11182B',
              background: '#11182B',
              display: 'flex',
              gap: '0.65rem',
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
                padding: '0.65rem 0.85rem',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: '#1A233D',
                background: '#1A233D',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                color: '#F8FAFC',
                fontSize: '0.84rem',
              }}
            />
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => handleSend()}
              disabled={loading || !input.trim()}
              style={{ padding: '0.6rem 0.95rem', fontSize: '0.82rem' }}
            >
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
}

