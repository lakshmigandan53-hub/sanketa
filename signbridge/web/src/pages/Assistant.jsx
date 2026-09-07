import { useState } from 'react';
import { grammarAssist, wordSuggestions, sentenceImprove, translateWord, speak, ISL_WORDS, humanWord } from '../services';

export default function Assistant({ lang }) {
  const [input, setInput] = useState('');
  const [grammarResult, setGrammarResult] = useState(null);
  const [altWords, setAltWords] = useState(null);
  const [improveResult, setImproveResult] = useState(null);
  const [selectedISL, setSelectedISL] = useState('');
  const [activeTab, setActiveTab] = useState('grammar');

  const runGrammar = () => {
    if (!input.trim()) return;
    setGrammarResult(grammarAssist(input));
    setAltWords(null);
    setImproveResult(null);
  };

  const runAlternatives = () => {
    if (!selectedISL) return;
    setAltWords(wordSuggestions(selectedISL));
    setGrammarResult(null);
    setImproveResult(null);
  };

  const runImprove = () => {
    if (!input.trim()) return;
    setImproveResult(sentenceImprove(input));
    setGrammarResult(null);
    setAltWords(null);
  };

  const tabs = [
    { id: 'grammar',   label: '✏️ Grammar Correct',  desc: 'Fix grammar and capitalization of your sentence' },
    { id: 'improve',   label: '⬆️ Improve Sentence',  desc: 'Get suggestions to improve your message' },
    { id: 'alts',      label: '🔀 Alternative Words',  desc: 'Find alternative expressions for ISL words' },
  ];

  return (
    <div>
      <div className="page-title">🤖 AI Sentence Assistant</div>
      <div className="page-sub">Local rule-based grammar and sentence assistant — no internet required</div>

      {/* Tab selector */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button
            key={t.id}
            className={`btn ${activeTab === t.id ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => { setActiveTab(t.id); setGrammarResult(null); setAltWords(null); setImproveResult(null); }}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="card card-sm" style={{ marginBottom: 20, background: 'rgba(99,102,241,0.06)', border: '1px solid rgba(99,102,241,0.2)' }}>
        <div style={{ fontSize: '0.85rem', color: 'var(--text2)' }}>
          💡 <strong style={{ color: '#fff' }}>{tabs.find(t => t.id === activeTab)?.label}:</strong>{' '}
          {tabs.find(t => t.id === activeTab)?.desc}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
        {/* Input panel */}
        <div>
          {activeTab !== 'alts' ? (
            <div className="card">
              <div className="section-title">Your Sentence</div>
              <textarea
                className="input-field"
                placeholder={activeTab === 'grammar'
                  ? 'Type your sentence here… e.g. "i need water please"'
                  : 'Type your sentence here… e.g. "help i need"'}
                value={input}
                onChange={e => setInput(e.target.value)}
                rows={4}
              />
              <div style={{ display: 'flex', gap: 10, marginTop: 14, flexWrap: 'wrap' }}>
                {activeTab === 'grammar' && (
                  <button className="btn btn-primary" onClick={runGrammar} disabled={!input.trim()}>
                    ✏️ Correct Grammar
                  </button>
                )}
                {activeTab === 'improve' && (
                  <button className="btn btn-primary" onClick={runImprove} disabled={!input.trim()}>
                    ⬆️ Improve Sentence
                  </button>
                )}
                {input && (
                  <button className="btn btn-ghost" onClick={() => { setInput(''); setGrammarResult(null); setImproveResult(null); }}>
                    Clear
                  </button>
                )}
              </div>

              {/* Quick examples */}
              <div style={{ marginTop: 16 }}>
                <div style={{ fontSize: '0.78rem', color: 'var(--text3)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Quick examples
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {['i need water please', 'help me please', 'thank you very much', 'sorry i dont understand'].map(ex => (
                    <button
                      key={ex}
                      className="btn btn-ghost btn-sm"
                      onClick={() => setInput(ex)}
                    >
                      "{ex}"
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="card">
              <div className="section-title">Select an ISL Word</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {ISL_WORDS.map(w => (
                  <button
                    key={w}
                    className={`btn ${selectedISL === w ? 'btn-primary' : 'btn-outline'}`}
                    style={{ justifyContent: 'flex-start', gap: 12 }}
                    onClick={() => setSelectedISL(w)}
                  >
                    <span style={{ fontSize: '1.1rem' }}>
                      {w === 'HELLO' ? '👋' : w === 'THANK_YOU' ? '🙏' : w === 'YES' ? '✅' :
                       w === 'NO' ? '❌' : w === 'PLEASE' ? '🤲' : w === 'SORRY' ? '😔' :
                       w === 'HELP' ? '🆘' : '💧'}
                    </span>
                    {humanWord(w)}
                    {lang !== 'en-US' && (
                      <span style={{ marginLeft: 'auto', fontSize: '0.85rem', color: 'var(--accent)' }}>
                        {translateWord(w, lang)}
                      </span>
                    )}
                  </button>
                ))}
              </div>
              <button
                className="btn btn-primary"
                style={{ width: '100%', justifyContent: 'center', marginTop: 16 }}
                onClick={runAlternatives}
                disabled={!selectedISL}
              >
                🔀 Get Alternatives
              </button>
            </div>
          )}
        </div>

        {/* Result panel */}
        <div>
          {/* Grammar result */}
          {grammarResult && (
            <div className="card fade-in">
              <div className="section-title">Grammar Result</div>
              <div style={{ background: 'var(--bg3)', borderRadius: 12, padding: 16, marginBottom: 14 }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text3)', marginBottom: 6, textTransform: 'uppercase' }}>Original</div>
                <div style={{ color: 'var(--text2)', fontStyle: 'italic' }}>"{grammarResult.original}"</div>
              </div>
              <div style={{ background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.2)', borderRadius: 12, padding: 16, marginBottom: 14 }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--green)', marginBottom: 6, textTransform: 'uppercase' }}>Corrected</div>
                <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#fff' }}>"{grammarResult.corrected}"</div>
              </div>
              {grammarResult.changed
                ? <div className="badge badge-green">✅ Grammar was corrected</div>
                : <div className="badge badge-blue">✓ Sentence looks correct</div>
              }
              <div style={{ display: 'flex', gap: 10, marginTop: 14 }}>
                <button className="btn btn-success btn-sm" onClick={() => speak(grammarResult.corrected, lang)}>
                  🔊 Speak Result
                </button>
                <button className="btn btn-ghost btn-sm" onClick={() => setInput(grammarResult.corrected)}>
                  ↩ Use as Input
                </button>
              </div>
            </div>
          )}

          {/* Improve result */}
          {improveResult && (
            <div className="card fade-in">
              <div className="section-title">Improved Sentence</div>
              <div style={{ background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.2)', borderRadius: 12, padding: 16, marginBottom: 14 }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--green)', marginBottom: 6, textTransform: 'uppercase' }}>Improved</div>
                <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#fff' }}>"{improveResult.improved}"</div>
              </div>
              <div style={{ marginBottom: 14 }}>
                <div style={{ fontSize: '0.82rem', color: 'var(--text3)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Suggestions</div>
                {improveResult.suggestions.map((s, i) => (
                  <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 6, fontSize: '0.88rem', color: 'var(--text2)' }}>
                    <span style={{ color: 'var(--accent)', flexShrink: 0 }}>•</span>
                    {s}
                  </div>
                ))}
              </div>
              <button className="btn btn-success btn-sm" onClick={() => speak(improveResult.improved, lang)}>
                🔊 Speak Improved
              </button>
            </div>
          )}

          {/* Alternative words result */}
          {altWords !== null && (
            <div className="card fade-in">
              <div className="section-title">Alternatives for "{humanWord(selectedISL)}"</div>
              {altWords.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {altWords.map((w, i) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'var(--bg3)', borderRadius: 10, padding: '12px 14px' }}>
                      <div style={{ fontWeight: 600, fontSize: '1rem' }}>{w}</div>
                      <button className="btn btn-success btn-sm" onClick={() => speak(w, lang)}>
                        🔊
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ color: 'var(--text3)', fontSize: '0.9rem' }}>
                  No specific alternatives found. This is a core ISL word — use it as-is.
                </div>
              )}
            </div>
          )}

          {/* Placeholder */}
          {!grammarResult && !altWords && !improveResult && (
            <div className="card" style={{ textAlign: 'center', padding: '48px 24px', border: '2px dashed var(--border)' }}>
              <div style={{ fontSize: '3rem', marginBottom: 12 }}>🤖</div>
              <div style={{ fontWeight: 600, marginBottom: 6 }}>AI Assistant Ready</div>
              <div style={{ color: 'var(--text3)', fontSize: '0.88rem' }}>
                Select a feature and submit your input to see results here.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Info card */}
      <div className="card" style={{ marginTop: 24, background: 'rgba(99,102,241,0.05)', border: '1px solid rgba(99,102,241,0.15)' }}>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>ℹ️ About This Assistant</div>
        <div style={{ color: 'var(--text2)', fontSize: '0.88rem', lineHeight: 1.8 }}>
          This is a <strong style={{ color: '#fff' }}>local rule-based assistant</strong> — it works entirely offline with no internet or API key required.
          Grammar correction handles capitalization, punctuation, and common contractions.
          For advanced AI suggestions, you can optionally integrate an external API by setting <code style={{ color: 'var(--accent)' }}>REACT_APP_AI_API_KEY</code> in your environment.
        </div>
      </div>
    </div>
  );
}
