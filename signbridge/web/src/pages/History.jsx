import { useState } from 'react';
import { getHistory, clearHistory, deleteHistory, speak } from '../services';

const DIR_COLORS = {
  'SIGN→TEXT':   'badge-green',
  'SPEECH→SIGN': 'badge-blue',
  'SPEECH→TEXT': 'badge-purple',
  'SIGN→SPEECH': 'badge-orange',
};

export default function History({ lang }) {
  const [history, setHistory] = useState(getHistory);

  const clear = () => { clearHistory(); setHistory([]); };
  const refresh = () => setHistory(getHistory());
  const remove = (id) => { deleteHistory(id); setHistory(getHistory()); };

  return (
    <div>
      <div className="page-title">📋 Translation History</div>
      <div className="page-sub">Recent translations stored locally in your browser — persists after refresh</div>

      <div style={{ display: 'flex', gap: 10, marginBottom: 24, alignItems: 'center' }}>
        <button className="btn btn-ghost btn-sm" onClick={refresh}>🔄 Refresh</button>
        {history.length > 0 && (
          <button className="btn btn-danger btn-sm" onClick={clear}>🗑 Clear All</button>
        )}
        <span style={{ marginLeft: 'auto', fontSize: '0.82rem', color: 'var(--text3)' }}>
          {history.length} item{history.length !== 1 ? 's' : ''}
        </span>
      </div>

      {history.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: 60, color: 'var(--text3)' }}>
          <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>📭</div>
          <div style={{ fontWeight: 600, marginBottom: 6, color: 'var(--text2)' }}>No history yet</div>
          <div style={{ fontSize: '0.88rem' }}>
            Use Camera, Translate, or Conversation to create translation history.
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {history.map(h => (
            <div key={h.id} style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: 14,
              padding: '12px 16px',
              transition: 'all 0.2s ease',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <span className={`badge ${DIR_COLORS[h.dir] || 'badge-purple'}`}>{h.dir}</span>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <span className="history-time">{h.ts}</span>
                  <button
                    className="btn btn-sm btn-ghost"
                    style={{ padding: '4px 8px' }}
                    onClick={() => speak(h.output, lang)}
                    title="Speak"
                  >🔊</button>
                  <button
                    className="btn btn-sm btn-danger"
                    style={{ padding: '4px 8px' }}
                    onClick={() => remove(h.id)}
                    title="Delete"
                  >✕</button>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: '0.85rem', color: 'var(--text2)' }}>{h.input}</div>
                <span style={{ color: 'var(--text3)', fontSize: '1.1rem' }}>→</span>
                <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#fff' }}>{h.output}</div>
              </div>
              {h.confidence && (
                <div style={{ marginTop: 6, fontSize: '0.75rem', color: 'var(--text3)' }}>
                  Confidence: {(h.confidence * 100).toFixed(1)}% · {h.type || 'ISL'}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
