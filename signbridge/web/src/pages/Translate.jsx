import { useState, useEffect, useRef } from 'react';
import {
  createRecognition,
  SIGN_ASSETS,
  ISL_WORDS,
  addHistory,
  speak,
  formatISLWord,
  normalizeISLWord,
  extractISLSequence,
  translateWord,
} from '../services';
import SignAnimation from '../components/SignAnimation';

// SignDisplay export maintained for backwards compatibility across older components
export function SignDisplay({ word, size = 'normal', showControls = true, lang = 'en-US' }) {
  const isLarge = size === 'large';
  const asset = word ? SIGN_ASSETS[word] : null;

  if (!word) return null;

  return (
    <div style={{ textAlign: 'center', width: '100%' }}>
      <SignAnimation
        word={word}
        showControls={showControls}
        maxHeight={isLarge ? 260 : 190}
      />
      {asset?.description && (
        <div
          style={{
            fontSize: '0.82rem',
            color: 'var(--text2, #94a3b8)',
            marginTop: 8,
            maxWidth: 340,
            margin: '8px auto 0',
            lineHeight: 1.4,
          }}
        >
          {asset.description}
        </div>
      )}
      {showControls && (
        <div style={{ marginTop: 10 }}>
          <button
            type="button"
            className="btn btn-sm btn-success"
            onClick={() => speak(translateWord(word, lang), lang)}
          >
            🔊 Speak Pronunciation
          </button>
        </div>
      )}
    </div>
  );
}

export default function Translate({ lang, setPage }) {
  const [mode, setMode] = useState('speech-to-sign');
  const [listening, setListening] = useState(false);
  const [spokenText, setSpokenText] = useState('');
  const [sequence, setSequence] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlayingAll, setIsPlayingAll] = useState(false);
  const [noMatch, setNoMatch] = useState(false);
  const [srSupported] = useState(() => !!(window.SpeechRecognition || window.webkitSpeechRecognition));

  const playAllTimerRef = useRef(null);

  // Stop auto-advance if sequence changes or unmounts
  useEffect(() => {
    return () => clearTimeout(playAllTimerRef.current);
  }, [sequence]);

  // Start Speech Recognition
  const startListening = () => {
    clearTimeout(playAllTimerRef.current);
    setIsPlayingAll(false);

    const rec = createRecognition(
      lang,
      (text) => {
        setSpokenText(text);
        const extracted = extractISLSequence(text);

        if (extracted.length > 0) {
          setSequence(extracted);
          setCurrentIndex(0);
          setNoMatch(false);
          addHistory({
            input: text,
            output: extracted.map(formatISLWord).join(' → '),
            dir: 'SPEECH→SIGN',
          });
          if (extracted.length > 1) {
            setTimeout(() => {
              handlePlayAll(extracted);
            }, 100);
          }
        } else {
          // Fallback single-word normalization
          const single = normalizeISLWord(text);
          if (single) {
            setSequence([single]);
            setCurrentIndex(0);
            setNoMatch(false);
            addHistory({ input: text, output: formatISLWord(single), dir: 'SPEECH→SIGN' });
          } else {
            setSequence([]);
            setCurrentIndex(0);
            setNoMatch(true);
          }
        }
      },
      () => setListening(false),
      () => setListening(false),
    );

    if (!rec) return;
    setListening(true);
    setSpokenText('');
    setSequence([]);
    setCurrentIndex(0);
    setNoMatch(false);

    try {
      rec.start();
    } catch (e) {
      setListening(false);
    }
  };

  const handleWordTap = (w) => {
    clearTimeout(playAllTimerRef.current);
    setIsPlayingAll(false);
    setSequence([w]);
    setCurrentIndex(0);
    setSpokenText(formatISLWord(w));
    setNoMatch(false);
    addHistory({ input: formatISLWord(w), output: formatISLWord(w), dir: 'SPEECH→SIGN' });
  };

  const activeWord = sequence[currentIndex] || null;

  // Next Word
  const handleNext = () => {
    if (currentIndex < sequence.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    }
  };

  // Previous Word
  const handlePrev = () => {
    if (currentIndex > 0) {
      setCurrentIndex((prev) => prev - 1);
    }
  };

  // Play All: auto-cycles through each word in the sequence
  const handlePlayAll = (customSeq = null) => {
    const activeSeq = customSeq || sequence;
    if (activeSeq.length <= 1) return;
    clearTimeout(playAllTimerRef.current);
    setIsPlayingAll(true);
    setCurrentIndex(0);

    const advance = (idx) => {
      if (idx < activeSeq.length - 1) {
        playAllTimerRef.current = setTimeout(() => {
          setCurrentIndex(idx + 1);
          advance(idx + 1);
        }, 2200);
      } else {
        setIsPlayingAll(false);
      }
    };
    advance(0);
  };

  return (
    <div>
      <div className="page-title">🔄 Translate</div>
      <div className="page-sub">Two-way Indian Sign Language translation with verified video demonstrations</div>

      {/* Mode toggle */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 24 }}>
        <button
          className={`btn ${mode === 'speech-to-sign' ? 'btn-primary' : 'btn-outline'}`}
          onClick={() => setMode('speech-to-sign')}
        >
          🎤 Speech → Sign
        </button>
        <button
          className={`btn ${mode === 'sign-to-speech' ? 'btn-primary' : 'btn-outline'}`}
          onClick={() => setMode('sign-to-speech')}
        >
          🤟 Sign → Speech
        </button>
      </div>

      {/* ── Speech → Sign ── */}
      {mode === 'speech-to-sign' && (
        <div className="grid-2" style={{ alignItems: 'start' }}>
          {/* Left Column: Voice input & Word Picker */}
          <div className="card">
            <div className="section-title">
              <span>Voice / Text Input</span>
            </div>

            {!srSupported && (
              <div
                className="badge badge-orange"
                style={{ marginBottom: 14, padding: '10px 14px', display: 'block', borderRadius: 10 }}
              >
                ⚠️ Speech recognition is not supported in this browser. Use Google Chrome / Edge or tap the word buttons below.
              </div>
            )}

            {srSupported && (
              <button
                className={`btn btn-lg ${listening ? 'btn-danger' : 'btn-primary'}`}
                onClick={startListening}
                disabled={listening}
                style={{ width: '100%', justifyContent: 'center', marginBottom: 16 }}
              >
                {listening ? (
                  <>
                    <span className="spin">🎙</span> Listening… Speak naturally
                  </>
                ) : (
                  '🎤 Tap to Speak'
                )}
              </button>
            )}

            {spokenText && (
              <div style={{ background: 'var(--bg3)', borderRadius: 10, padding: '14px 16px', marginBottom: 12 }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text3)', textTransform: 'uppercase', marginBottom: 4 }}>
                  Recognized Speech
                </div>
                <div style={{ fontSize: '1.15rem', fontWeight: 600, color: 'var(--primary)' }}>
                  "{spokenText}"
                </div>
              </div>
            )}

            {noMatch && (
              <div
                className="badge badge-orange"
                style={{ padding: '8px 12px', marginBottom: 12, display: 'block', borderRadius: 8 }}
              >
                No ISL signs detected in "{spokenText}". Try speaking "Hello", "Please", "Water", "Thank you", or select a word below:
              </div>
            )}

            {/* Quick Word Tap Grid */}
            <div style={{ marginTop: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text2)', fontWeight: 600 }}>
                  Supported ISL Signs:
                </span>
                {sequence.length > 0 && (
                  <button
                    className="btn btn-sm btn-ghost"
                    onClick={() => {
                      setSequence([]);
                      setSpokenText('');
                    }}
                  >
                    Clear Sequence
                  </button>
                )}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {ISL_WORDS.map((w) => {
                  const asset = SIGN_ASSETS[w];
                  const isSelected = activeWord === w;
                  return (
                    <button
                      key={w}
                      className={`btn btn-sm ${isSelected ? 'btn-primary' : 'btn-ghost'}`}
                      onClick={() => handleWordTap(w)}
                      title={`Select ${formatISLWord(w)}`}
                    >
                      {asset?.video ? '🎬' : '⚡'} {formatISLWord(w)}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Translation in selected language */}
            {activeWord && lang !== 'en-US' && (
              <div
                style={{
                  marginTop: 14,
                  padding: '10px 14px',
                  background: 'rgba(99,102,241,0.08)',
                  borderRadius: 8,
                  border: '1px solid rgba(99,102,241,0.2)',
                }}
              >
                <div style={{ fontSize: '0.75rem', color: 'var(--text3)', marginBottom: 4 }}>
                  LOCAL TRANSLATION
                </div>
                <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent)' }}>
                  {translateWord(activeWord, lang)}
                </div>
              </div>
            )}

            {activeWord && (
              <button
                className="btn btn-success"
                style={{ marginTop: 14 }}
                onClick={() => speak(translateWord(activeWord, lang), lang)}
              >
                🔊 Speak "{translateWord(activeWord, lang)}"
              </button>
            )}
          </div>

          {/* Right Column: Sign Demonstration Animation / Sequence Player */}
          <div>
            {sequence.length > 0 && activeWord ? (
              <div className="card">
                {/* Sequence Header & Indicators */}
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: 12,
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                    paddingBottom: 8,
                  }}
                >
                  <div>
                    {sequence.length > 1 ? (
                      <div style={{ fontSize: '0.8rem', color: 'var(--primary)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                        Playing {currentIndex + 1} / {sequence.length}
                      </div>
                    ) : (
                      <div style={{ fontSize: '0.8rem', color: 'var(--text3)', fontWeight: 600 }}>
                        ISL Sign
                      </div>
                    )}
                    <div style={{ fontWeight: 800, fontSize: '1.25rem', color: '#fff' }}>
                      {formatISLWord(activeWord)}
                    </div>
                  </div>
                  {sequence.length > 1 && (
                    <span className="badge badge-purple" style={{ fontSize: '0.85rem', fontWeight: 700, padding: '4px 10px' }}>
                      {currentIndex + 1} of {sequence.length}
                    </span>
                  )}
                </div>

                {/* Sequence Word Chips */}
                {sequence.length > 1 && (
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 14 }}>
                    {sequence.map((w, idx) => (
                      <button
                        key={`${w}-${idx}`}
                        className={`badge ${idx === currentIndex ? 'badge-blue' : 'badge-ghost'}`}
                        style={{
                          cursor: 'pointer',
                          border: idx === currentIndex ? '1px solid var(--primary)' : '1px solid rgba(255,255,255,0.1)',
                          padding: '4px 10px',
                          fontWeight: idx === currentIndex ? 700 : 500,
                        }}
                        onClick={() => {
                          clearTimeout(playAllTimerRef.current);
                          setIsPlayingAll(false);
                          setCurrentIndex(idx);
                        }}
                      >
                        {idx + 1}. {formatISLWord(w)}
                      </button>
                    ))}
                  </div>
                )}

                {/* Reusable Real Sign Animation Component */}
                <SignAnimation
                  sign={activeWord}
                  word={activeWord}
                  showControls={true}
                  maxHeight={280}
                />

                {/* Description and Tip */}
                {SIGN_ASSETS[activeWord]?.description && (
                  <div
                    style={{
                      fontSize: '0.85rem',
                      color: 'var(--text2)',
                      marginTop: 12,
                      lineHeight: 1.45,
                    }}
                  >
                    <strong>Execution:</strong> {SIGN_ASSETS[activeWord].description}
                  </div>
                )}

                {/* Sequence Navigation Controls: PREVIOUS, PAUSE, NEXT, REPLAY, PLAY ALL */}
                {sequence.length > 1 && (
                  <div
                    style={{
                      display: 'flex',
                      gap: 8,
                      marginTop: 16,
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      flexWrap: 'wrap',
                    }}
                  >
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={handlePrev}
                      disabled={currentIndex === 0}
                    >
                      ◀ Previous
                    </button>

                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      <button
                        className={`btn btn-sm ${isPlayingAll ? 'btn-danger' : 'btn-primary'}`}
                        onClick={isPlayingAll ? () => { clearTimeout(playAllTimerRef.current); setIsPlayingAll(false); } : () => handlePlayAll()}
                      >
                        {isPlayingAll ? '⏸ Pause' : '▶ Play All'}
                      </button>

                      <button
                        className="btn btn-sm btn-ghost"
                        onClick={() => {
                          clearTimeout(playAllTimerRef.current);
                          handlePlayAll();
                        }}
                      >
                        ⟳ Replay All
                      </button>
                    </div>

                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={handleNext}
                      disabled={currentIndex === sequence.length - 1}
                    >
                      Next ▶
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="sign-placeholder" style={{ minHeight: 340 }}>
                <div className="sign-icon">🤟</div>
                <div style={{ fontWeight: 600, fontSize: '1.1rem', marginBottom: 4 }}>
                  Real ISL Sign Animation
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text3)', maxWidth: 280 }}>
                  Speak a phrase like "Hello please help" or tap any sign button to see the authentic ISL demonstration.
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Sign → Speech ── */}
      {mode === 'sign-to-speech' && (
        <div>
          {/* Real-Time Live Webcam Banner */}
          <div
            className="card"
            style={{
              marginBottom: 20,
              background: 'rgba(0,229,255,0.05)',
              border: '1px solid rgba(0,229,255,0.15)',
            }}
          >
            <div style={{ fontWeight: 600, fontSize: '1.05rem', marginBottom: 6 }}>
              📷 Real-Time Live Webcam Recognition
            </div>
            <div style={{ color: 'var(--text2)', fontSize: '0.9rem', marginBottom: 14 }}>
              Open the Camera page to use your webcam for live 21-landmark skeleton tracking and sign recognition with automatic text-to-speech.
            </div>
            <button className="btn btn-primary btn-lg" onClick={() => setPage('camera')}>
              Open Live Camera →
            </button>
          </div>

          {/* Quick manual sign selection cards */}
          <div className="section-title">
            <span>Or select an ISL sign to preview demonstration and audio:</span>
          </div>
          <div className="grid-4" style={{ marginTop: 14 }}>
            {ISL_WORDS.map((w) => {
              const asset = SIGN_ASSETS[w];
              return (
                <div
                  key={w}
                  className="card card-sm"
                  style={{
                    textAlign: 'center',
                    cursor: 'pointer',
                    transition: 'transform 0.15s ease',
                  }}
                  onClick={() => speak(translateWord(w, lang), lang)}
                >
                  <div
                    style={{
                      height: 120,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: '#070b14',
                      borderRadius: 8,
                      overflow: 'hidden',
                      marginBottom: 8,
                    }}
                  >
                    {asset?.video ? (
                      <video
                        src={asset.video}
                        autoPlay
                        loop
                        muted
                        playsInline
                        style={{ height: '100%', width: '100%', objectFit: 'cover' }}
                      />
                    ) : (
                      <img
                        src={asset?.gif || asset?.image || asset?.fallbackImage}
                        alt={formatISLWord(w)}
                        style={{ maxHeight: 110, maxWidth: '100%', objectFit: 'contain' }}
                      />
                    )}
                  </div>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem', color: '#fff' }}>
                    {formatISLWord(w)}
                  </div>
                  {lang !== 'en-US' && (
                    <div style={{ fontSize: '0.8rem', color: 'var(--accent)', marginTop: 2 }}>
                      {translateWord(w, lang)}
                    </div>
                  )}
                  <div style={{ fontSize: '0.75rem', color: 'var(--cyan)', marginTop: 4 }}>
                    Tap to Speak 🔊
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
