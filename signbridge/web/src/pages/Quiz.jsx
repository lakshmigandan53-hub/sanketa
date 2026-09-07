import { useState, useCallback } from 'react';
import { SIGN_ASSETS, ISL_WORDS, speak, translateWord, formatISLWord } from '../services';
import SignAnimation from '../components/SignAnimation';

function shuffle(a) { return [...a].sort(() => Math.random() - 0.5); }

function generateQuestion(mode) {
  const pool = ISL_WORDS;
  const word = pool[Math.floor(Math.random() * pool.length)];
  const others = shuffle(pool.filter(w => w !== word)).slice(0, 3);
  const options = shuffle([word, ...others]);
  return { word, options, mode };
}

export default function Quiz({ lang }) {
  const [q, setQ] = useState(() => generateQuestion('sign-to-word'));
  const [mode, setMode] = useState('sign-to-word');
  const [selected, setSelected] = useState(null);
  const [score, setScore] = useState({ right: 0, wrong: 0 });
  const [streak, setStreak] = useState(0);
  const [total, setTotal] = useState(0);

  const next = useCallback((m = mode) => {
    setQ(generateQuestion(m));
    setSelected(null);
  }, [mode]);

  const choose = (opt) => {
    if (selected) return;
    setSelected(opt);
    const correct = opt === q.word;
    setScore(s => ({ right: s.right + (correct ? 1 : 0), wrong: s.wrong + (correct ? 0 : 1) }));
    setStreak(s => correct ? s + 1 : 0);
    setTotal(t => t + 1);
    if (correct) speak('Correct!', lang);
    setTimeout(() => next(), 2200);
  };

  const switchMode = (m) => {
    setMode(m);
    setQ(generateQuestion(m));
    setSelected(null);
    setScore({ right: 0, wrong: 0 });
    setStreak(0);
    setTotal(0);
  };

  const accuracy = total > 0 ? Math.round((score.right / total) * 100) : 0;

  return (
    <div>
      <div className="page-title">🎯 ISL Quiz</div>
      <div className="page-sub">Test your Indian Sign Language recognition knowledge</div>

      {/* Score bar */}
      <div className="card-sm" style={{ display: 'flex', gap: 16, marginBottom: 24, flexWrap: 'wrap', alignItems: 'center' }}>
        <span className="badge badge-green">✅ {score.right} Correct</span>
        <span className="badge badge-red">❌ {score.wrong} Wrong</span>
        {total > 0 && <span className="badge badge-blue">📊 {accuracy}% Accuracy</span>}
        {streak > 1 && <span className="badge badge-orange">🔥 {streak} streak!</span>}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button
            className={`btn btn-sm ${mode === 'sign-to-word' ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => switchMode('sign-to-word')}
          >
            🤟 Sign → Word
          </button>
          <button
            className={`btn btn-sm ${mode === 'word-to-sign' ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => switchMode('word-to-sign')}
          >
            📝 Word → Gesture
          </button>
          <button className="btn btn-sm btn-ghost" onClick={() => switchMode(mode)}>
            🔄 Restart
          </button>
        </div>
      </div>

      {/* Question card */}
      <div className="card" style={{ maxWidth: 640, margin: '0 auto', textAlign: 'center' }}>
        {mode === 'sign-to-word' ? (
          <>
            <div style={{ fontSize: '0.85rem', color: 'var(--text3)', marginBottom: 8 }}>
              Which word matches this ISL sign demonstration?
            </div>
            <div style={{ marginBottom: 16, maxWidth: 300, margin: '0 auto 16px' }}>
              <SignAnimation sign={q.word} word={q.word} maxHeight={220} showControls={false} hideLabel={true} />
            </div>

            {/* Word options */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {q.options.map(opt => {
                let cls = 'quiz-option';
                if (selected) {
                  if (opt === q.word) cls += ' correct';
                  else if (opt === selected) cls += ' wrong';
                }
                return (
                  <button
                    key={opt}
                    className={cls}
                    onClick={() => choose(opt)}
                    style={{ padding: '14px', fontSize: '1rem', fontWeight: 600 }}
                  >
                    {formatISLWord(opt)}
                  </button>
                );
              })}
            </div>
          </>
        ) : (
          <>
            <div style={{ fontSize: '0.85rem', color: 'var(--text3)', marginBottom: 8 }}>
              Which gesture movement describes the ISL sign for:
            </div>
            <div style={{ fontSize: '2.4rem', fontWeight: 800, marginBottom: 4, color: 'var(--cyan)' }}>
              {formatISLWord(q.word)}
            </div>
            {lang !== 'en-US' && (
              <div style={{ fontSize: '1.1rem', color: 'var(--accent)', marginBottom: 16 }}>
                {translateWord(q.word, lang)}
              </div>
            )}

            {/* Gesture description options */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 10 }}>
              {q.options.map(opt => {
                const desc = SIGN_ASSETS[opt]?.description || opt;
                let borderStyle = '1px solid var(--border)';
                let bgStyle = '#0a0e1a';
                if (selected) {
                  if (opt === q.word) {
                    borderStyle = '2px solid var(--green, #22c55e)';
                    bgStyle = 'rgba(34, 197, 94, 0.15)';
                  } else if (opt === selected) {
                    borderStyle = '2px solid var(--red, #ef4444)';
                    bgStyle = 'rgba(239, 68, 68, 0.15)';
                  }
                }
                return (
                  <button
                    key={opt}
                    onClick={() => choose(opt)}
                    disabled={!!selected}
                    style={{
                      background: bgStyle,
                      border: borderStyle,
                      borderRadius: 10,
                      padding: '12px 14px',
                      cursor: selected ? 'default' : 'pointer',
                      textAlign: 'left',
                      fontSize: '0.88rem',
                      color: 'var(--text1)',
                      lineHeight: 1.4,
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <strong>{formatISLWord(opt)}:</strong> {desc}
                  </button>
                );
              })}
            </div>

            {/* Reveal demonstration card after answering */}
            {selected && (
              <div style={{ marginTop: 20, maxWidth: 300, margin: '20px auto 0' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text3)', marginBottom: 6 }}>
                  Demonstration:
                </div>
                <SignAnimation sign={q.word} word={q.word} maxHeight={190} showControls={false} />
              </div>
            )}
          </>
        )}

        {selected && (
          <div style={{ marginTop: 20, fontWeight: 700, fontSize: '1.15rem', color: selected === q.word ? 'var(--green)' : 'var(--red)' }}>
            {selected === q.word
              ? '✅ Correct!'
              : `❌ The correct sign was: ${formatISLWord(q.word)}`}
          </div>
        )}

        <button className="btn btn-ghost btn-sm" style={{ marginTop: 16 }} onClick={() => next()}>
          Next Question →
        </button>
      </div>

      {total > 0 && (
        <div style={{ textAlign: 'center', marginTop: 20, color: 'var(--text3)', fontSize: '0.82rem' }}>
          {total} question{total !== 1 ? 's' : ''} answered · {score.right}/{total} correct
        </div>
      )}
    </div>
  );
}
