import { useState, useRef } from 'react';
import {
  speak,
  createRecognition,
  addHistory,
  translateWord,
  grammarAssist,
  formatISLWord,
  normalizeISLWord,
  extractISLSequence,
} from '../services';
import SignAnimation from '../components/SignAnimation';

export default function Conversation({ lang }) {
  const [messages, setMessages] = useState([]);
  const [personAListening, setPersonAListening] = useState(false);
  const [personBListening, setPersonBListening] = useState(false);
  const [srSupported] = useState(() => !!(window.SpeechRecognition || window.webkitSpeechRecognition));
  const bottomRef = useRef(null);

  const scrollDown = () => setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);

  const addMsg = (msg) => {
    setMessages(p => [
      ...p,
      {
        ...msg,
        id: Date.now() + Math.random(),
        ts: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
    addHistory({ input: msg.input, output: msg.output, dir: msg.dir });
    scrollDown();
  };

  // Person A: Sign/Speech → Text → Speech (deaf user)
  const startPersonA = () => {
    if (!srSupported) return;
    const rec = createRecognition(
      lang,
      (text) => {
        const ai = grammarAssist(text);
        const out = ai?.corrected || text;
        const matchWord = normalizeISLWord(text);
        speak(translateWord(out, lang), lang);
        addMsg({ who: 'A', input: 'Speech/Sign', output: out, dir: 'SIGN→TEXT', matchWord });
      },
      () => setPersonAListening(false),
      () => setPersonAListening(false)
    );
    if (!rec) return;
    setPersonAListening(true);
    try {
      rec.start();
    } catch (e) {
      setPersonAListening(false);
    }
  };

  // Person B: Speech → Text → Sign (hearing user)
  const startPersonB = () => {
    if (!srSupported) return;
    const rec = createRecognition(
      'en-US',
      (text) => {
        const seq = extractISLSequence(text);
        const single = normalizeISLWord(text);
        const matched = seq.length > 0 ? seq : (single ? [single] : []);
        addMsg({ who: 'B', input: text, output: text, dir: 'SPEECH→SIGN', matchWords: matched });
      },
      () => setPersonBListening(false),
      () => setPersonBListening(false)
    );
    if (!rec) return;
    setPersonBListening(true);
    try {
      rec.start();
    } catch (e) {
      setPersonBListening(false);
    }
  };

  return (
    <div>
      <div className="page-title">💬 Conversation Mode</div>
      <div className="page-sub">Two-way ISL ↔ Speech communication with verified sign demonstration panels</div>

      {!srSupported && (
        <div
          className="badge badge-orange"
          style={{ marginBottom: 20, padding: '10px 16px', display: 'block', borderRadius: 10 }}
        >
          ⚠️ Speech recognition not supported in this browser. Use Google Chrome / Edge for full voice conversation.
        </div>
      )}

      {/* Person panels */}
      <div className="grid-2" style={{ alignItems: 'start', marginBottom: 24 }}>
        {/* Person A — Deaf/HH */}
        <div className="card" style={{ border: '1px solid rgba(0,229,255,0.25)' }}>
          <div style={{ fontWeight: 700, marginBottom: 4, color: 'var(--primary)' }}>
            🤟 Person A — Deaf / Hard of Hearing
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text2)', marginBottom: 14 }}>
            Signs / Speaks → Text displayed → Spoken voice output for Person B
          </div>
          <button
            type="button"
            className={`btn btn-lg ${personAListening ? 'btn-danger' : 'btn-primary'}`}
            onClick={startPersonA}
            disabled={personAListening || !srSupported}
            style={{ width: '100%', justifyContent: 'center' }}
          >
            {personAListening ? (
              <>
                <span className="pulse">🎙</span> Listening…
              </>
            ) : srSupported ? (
              '🎤 Speak / Simulate Sign'
            ) : (
              '🎤 (Requires Chrome)'
            )}
          </button>
          <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {['HELLO', 'WATER', 'YES', 'NO'].map(w => (
              <button
                key={w}
                type="button"
                className="btn btn-xs btn-outline"
                style={{ fontSize: '0.72rem', padding: '3px 8px' }}
                onClick={() => {
                  const out = formatISLWord(w);
                  speak(translateWord(out, lang), lang);
                  addMsg({ who: 'A', input: 'Quick Sign', output: out, dir: 'SIGN→TEXT', matchWord: w });
                }}
              >
                🤟 {formatISLWord(w)}
              </button>
            ))}
          </div>
        </div>

        {/* Person B — Hearing */}
        <div className="card" style={{ border: '1px solid rgba(6,182,212,0.25)' }}>
          <div style={{ fontWeight: 700, marginBottom: 4, color: 'var(--cyan)' }}>
            🎤 Person B — Hearing
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text2)', marginBottom: 14 }}>
            Speaks → Text + verified ISL sign demonstration displayed for Person A
          </div>
          <button
            type="button"
            className={`btn btn-lg ${personBListening ? 'btn-danger' : ''}`}
            style={{
              width: '100%',
              justifyContent: 'center',
              background: personBListening ? 'var(--red)' : 'var(--cyan)',
              color: 'white',
              border: 'none',
            }}
            onClick={startPersonB}
            disabled={personBListening || !srSupported}
          >
            {personBListening ? (
              <>
                <span className="pulse">🎙</span> Listening…
              </>
            ) : srSupported ? (
              '🎤 Speak to Person A'
            ) : (
              '🎤 (Requires Chrome)'
            )}
          </button>
          <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {['THANK_YOU', 'PLEASE', 'HELP', 'SORRY'].map(w => (
              <button
                key={w}
                type="button"
                className="btn btn-xs btn-outline"
                style={{ fontSize: '0.72rem', padding: '3px 8px' }}
                onClick={() => {
                  addMsg({ who: 'B', input: formatISLWord(w), output: formatISLWord(w), dir: 'SPEECH→SIGN', matchWords: [w] });
                }}
              >
                💬 {formatISLWord(w)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Conversation bubbles */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div className="section-title" style={{ marginBottom: 0 }}>
            Conversation Log
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span className="badge badge-blue">{messages.length} messages</span>
            {messages.length > 0 && (
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => setMessages([])}>
                Clear
              </button>
            )}
          </div>
        </div>

        {messages.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--text3)', padding: '48px 0', fontSize: '0.9rem' }}>
            <div style={{ fontSize: '2rem', marginBottom: 8 }}>💬</div>
            Start speaking or tap a quick phrase — conversation messages will appear here
          </div>
        ) : null}

        <div className="chat-container">
          {messages.map(m => (
            <div
              key={m.id}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: m.who === 'A' ? 'flex-end' : 'flex-start',
                marginBottom: 14,
              }}
            >
              <div className={`bubble ${m.who === 'A' ? 'bubble-a' : 'bubble-b'}`} style={{ maxWidth: '82%' }}>
                <div
                  style={{
                    fontWeight: 600,
                    fontSize: '0.72rem',
                    marginBottom: 6,
                    color: m.who === 'A' ? 'var(--primary)' : 'var(--cyan)',
                  }}
                >
                  {m.who === 'A' ? '🤟 Person A (Deaf)' : '🎤 Person B (Hearing)'} · {m.dir}
                </div>
                <div style={{ fontSize: '1.05rem', fontWeight: 600 }}>"{m.output}"</div>

                {/* Show verified ISL animation demonstration for messages */}
                {(m.matchWords?.length > 0 || m.matchWord) && (
                  <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {(m.matchWords || [m.matchWord]).filter(Boolean).map((word, idx) => (
                      <div key={`${word}-${idx}`} style={{ maxWidth: 260 }}>
                        <SignAnimation word={word} sign={word} maxHeight={180} showControls={false} />
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <div
                className="bubble-meta"
                style={{
                  marginTop: 3,
                  marginLeft: m.who === 'A' ? 0 : 4,
                  marginRight: m.who === 'A' ? 4 : 0,
                }}
              >
                {m.ts}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  );
}
