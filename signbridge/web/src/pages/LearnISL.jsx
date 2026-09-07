import { useState } from 'react';
import { SIGN_ASSETS, ISL_WORDS, translateWord, formatISLWord, speak } from '../services';
import SignAnimation from '../components/SignAnimation';

function SignCard({ word, lang }) {
  const asset = SIGN_ASSETS[word];
  const [showTip, setShowTip] = useState(false);
  const [animKey, setAnimKey] = useState(0);

  if (!asset) return null;

  return (
    <div className="card" style={{ textAlign: 'center', cursor: 'default', padding: '16px 14px' }}>
      {/* Title */}
      <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#fff', marginBottom: 10 }}>
        {formatISLWord(word)}
      </div>

      {/* Real procedural illustrated hand animation */}
      <SignAnimation
        key={`${word}-${animKey}`}
        sign={word}
        word={word}
        showControls={true}
        maxHeight={210}
      />

      {/* Translated label */}
      {lang !== 'en-US' && (
        <div style={{ fontSize: '0.92rem', color: 'var(--accent)', marginTop: 8, fontWeight: 700 }}>
          {translateWord(word, lang)}
        </div>
      )}

      {/* Description */}
      {asset.description && (
        <div
          style={{
            fontSize: '0.82rem',
            color: 'var(--text2)',
            marginTop: 8,
            lineHeight: 1.45,
            textAlign: 'center',
            minHeight: 38,
          }}
        >
          {asset.description}
        </div>
      )}

      {/* Learning tip dropdown */}
      {showTip && (
        <div
          style={{
            background: 'rgba(0, 229, 255, 0.08)',
            border: '1px solid rgba(0, 229, 255, 0.25)',
            borderRadius: 8,
            padding: '10px 12px',
            fontSize: '0.82rem',
            color: 'var(--text1)',
            marginTop: 10,
            lineHeight: 1.5,
            textAlign: 'left',
          }}
        >
          <div style={{ fontWeight: 700, color: 'var(--cyan)', marginBottom: 2 }}>💡 Execution Tip:</div>
          {asset.tip || asset.description}
        </div>
      )}

      {/* Card Action Buttons: Replay, Speak Pronunciation, Learning Tip */}
      <div style={{ display: 'flex', gap: 6, justifyContent: 'center', flexWrap: 'wrap', marginTop: 12 }}>
        <button
          type="button"
          className="btn btn-sm btn-ghost"
          onClick={() => setAnimKey(k => k + 1)}
          title="Replay animation from start"
        >
          ⟳ Replay
        </button>
        <button
          type="button"
          className="btn btn-sm btn-success"
          onClick={() => speak(translateWord(word, lang), lang)}
        >
          🔊 Speak
        </button>
        <button
          type="button"
          className={`btn btn-sm ${showTip ? 'btn-primary' : 'btn-ghost'}`}
          onClick={() => setShowTip(p => !p)}
        >
          💡 {showTip ? 'Hide Tip' : 'Learning Tip'}
        </button>
      </div>
    </div>
  );
}

export default function LearnISL({ lang }) {
  const [filter, setFilter] = useState('all');

  const words = filter === 'all'
    ? ISL_WORDS
    : ISL_WORDS.filter(w => SIGN_ASSETS[w]?.type === filter);

  return (
    <div>
      <div className="page-title">📚 Learn ISL</div>
      <div className="page-sub">
        Explore Indian Sign Language dictionary — cards display verified procedural illustrated demonstrations, descriptions, and pronunciation
      </div>

      {/* Filter */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 24 }}>
        {['all', 'static', 'dynamic'].map(f => (
          <button
            key={f}
            className={`btn ${filter === f ? 'btn-primary' : 'btn-outline'} btn-sm`}
            onClick={() => setFilter(f)}
          >
            {f === 'all' ? `All Signs (${ISL_WORDS.length})` : f === 'static' ? '✋ Static Signs (4)' : '⚡ Dynamic Signs (4)'}
          </button>
        ))}
      </div>

      <div className="grid-4">
        {words.map(w => <SignCard key={w} word={w} lang={lang} />)}
      </div>

      {/* Educational Guidelines */}
      <div className="card" style={{ marginTop: 24, background: 'rgba(99,102,241,0.05)', border: '1px solid rgba(99,102,241,0.2)' }}>
        <div style={{ fontWeight: 700, color: '#fff', marginBottom: 8, fontSize: '0.98rem' }}>
          💡 ISL Demonstration & Learning Standards
        </div>
        <ul style={{ paddingLeft: 18, color: 'var(--text2)', fontSize: '0.875rem', lineHeight: 2 }}>
          <li><strong style={{ color: '#fff' }}>Static signs</strong>: HELLO, YES, NO, WATER — hold hand position steadily with clean palm and finger shapes.</li>
          <li><strong style={{ color: '#fff' }}>Dynamic signs</strong>: THANK YOU, PLEASE, SORRY, HELP — follow the natural physical movement of wrist and fingers.</li>
          <li>Click 🔊 <strong>Speak</strong> on any card to hear the pronunciation in your selected language.</li>
          <li>Click 💡 <strong>Learning Tip</strong> on each card to read the precise finger orientation and movement advice.</li>
          <li>All 8 core signs are illustrated with procedural 30-frame anatomical kinematics running smoothly locally.</li>
        </ul>
      </div>
    </div>
  );
}
