import { useState, useEffect, useRef, useCallback } from 'react';
import {
  apiHealth,
  apiModelInfo,
  apiTranslateImage,
  formatISLWord,
  confColor,
  SIGN_ASSETS,
  ISL_WORDS,
} from '../services';
import SignAnimation from '../components/SignAnimation';

export default function Home({ setPage }) {
  const [health, setHealth] = useState(null);
  const [model, setModel] = useState(null);

  // Live Demo mini-preview state
  const [demoCameraActive, setDemoCameraActive] = useState(false);
  const [demoStarting, setDemoStarting] = useState(false);
  const [demoError, setDemoError] = useState('');
  const [demoPrediction, setDemoPrediction] = useState(null);
  const [demoHandDetected, setDemoHandDetected] = useState(false);
  const [selectedHomeSign, setSelectedHomeSign] = useState('HELLO');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const loopRef = useRef(null);
  const isProcessingRef = useRef(false);

  useEffect(() => {
    apiHealth().then(setHealth).catch(() => setHealth({ status: 'offline' }));
    apiModelInfo().then(setModel).catch(() => setModel({ model_loaded: false }));
  }, []);

  const online = health?.status === 'healthy';
  const modelLoaded = model?.model_loaded;

  // Stop Demo Camera
  const stopDemoCamera = useCallback(() => {
    if (loopRef.current) {
      clearInterval(loopRef.current);
      loopRef.current = null;
    }
    isProcessingRef.current = false;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        try { track.stop(); } catch (e) {}
      });
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setDemoCameraActive(false);
    setDemoStarting(false);
    setDemoPrediction(null);
    setDemoHandDetected(false);
  }, []);

  // Process a single demo frame (~4 req/sec)
  const processDemoFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isProcessingRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0 || video.paused || video.ended) return;

    isProcessingRef.current = true;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    // Draw canonical unmirrored frame for backend recognition (preview is mirrored via CSS)
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      async (blob) => {
        if (!blob) {
          isProcessingRef.current = false;
          return;
        }
        try {
          const data = await apiTranslateImage(blob);
          if (data.success && data.word && data.word !== 'uncertain') {
            setDemoHandDetected(true);
            setDemoPrediction({
              word: data.word,
              confidence: data.confidence || 0,
              type: data.type || 'static',
            });
          } else {
            setDemoHandDetected(false);
          }
        } catch (e) {
          // Backend unreachable or network error
        } finally {
          isProcessingRef.current = false;
        }
      },
      'image/jpeg',
      0.80
    );
  }, []);

  // Start Demo Camera
  const startDemoCamera = async () => {
    stopDemoCamera();
    setDemoError('');
    setDemoStarting(true);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Webcam is not supported in this browser.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 480 }, height: { ideal: 360 } },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setDemoCameraActive(true);
        setDemoStarting(false);

        // Start throttled frame loop (~260ms = ~3.8 FPS)
        loopRef.current = setInterval(processDemoFrame, 260);
      }
    } catch (err) {
      console.error('Demo camera error:', err);
      setDemoStarting(false);
      setDemoCameraActive(false);
      setDemoError(err.name === 'NotAllowedError' ? 'Camera permission was denied.' : 'No camera detected.');
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopDemoCamera();
    };
  }, [stopDemoCamera]);

  const features = [
    {
      id: 'camera',
      icon: '📷',
      title: 'SIGN → TEXT',
      subtitle: 'Real-time webcam recognition',
      desc: 'MediaPipe 63-landmark tracking combined with a lightweight neural network for instant gesture-to-text conversion.',
      color: 'var(--primary)',
    },
    {
      id: 'camera',
      icon: '🔊',
      title: 'SIGN → SPEECH',
      subtitle: 'Recognized signs spoken aloud',
      desc: 'Automatic text-to-speech synthesis translates detected hand gestures into clear spoken voice in English, Tamil, or Hindi.',
      color: 'var(--cyan)',
    },
    {
      id: 'translate',
      icon: '🤟',
      title: 'SPEECH → SIGN',
      subtitle: 'Voice converted to ISL demonstration',
      desc: 'Browser speech recognition breaks down spoken sentences into verified Indian Sign Language video animations.',
      color: 'var(--accent)',
    },
    {
      id: 'learn',
      icon: '📚',
      title: 'LEARN ISL',
      subtitle: 'Interactive sign learning',
      desc: 'Interactive visual dictionary with authentic demonstration videos, execution tips, and audio pronunciation guides.',
      color: '#f59e0b',
    },
    {
      id: 'conversation',
      icon: '💬',
      title: 'CONVERSATION',
      subtitle: 'Two-way communication',
      desc: 'Bridge deaf and hearing speakers with simultaneous sign-to-speech and speech-to-sign dialogue channels.',
      color: '#10b981',
    },
    {
      id: 'emergency',
      icon: '🚨',
      title: 'EMERGENCY',
      subtitle: 'Fast emergency communication',
      desc: 'High-visibility rapid-access tiles with medical, police, and urgent SOS phrases spoken with priority.',
      color: '#ef4444',
    },
  ];

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto' }}>
      {/* ── HERO SECTION ──────────────────────────────────────────────────────── */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1.2fr 1fr',
          gap: 36,
          alignItems: 'center',
          padding: '28px 0 42px 0',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          marginBottom: 42,
        }}
        className="hero-grid"
      >
        {/* Left Column: Value Proposition & CTAs */}
        <div>
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              background: 'rgba(0, 229, 255, 0.08)',
              border: '1px solid rgba(0, 229, 255, 0.25)',
              borderRadius: 20,
              padding: '4px 14px',
              fontSize: '0.78rem',
              color: 'var(--primary)',
              fontWeight: 700,
              letterSpacing: '0.5px',
              textTransform: 'uppercase',
              marginBottom: 16,
            }}
          >
            <span>🇮🇳</span> SIH Hackathon Demo • Indian Sign Language
          </div>

          <h1
            style={{
              fontSize: '2.9rem',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              lineHeight: 1.15,
              margin: '0 0 16px 0',
              color: '#ffffff',
            }}
          >
            SANKETA
          </h1>

          <p
            style={{
              fontSize: '1.25rem',
              fontWeight: 600,
              color: 'var(--text1)',
              lineHeight: 1.4,
              margin: '0 0 12px 0',
            }}
          >
            Indian Sign Language Translation Platform
          </p>

          <p
            style={{
              fontSize: '0.96rem',
              color: 'var(--text2)',
              lineHeight: 1.6,
              maxWidth: 480,
              margin: '0 0 20px 0',
            }}
          >
            Real-time local ISL translation for Sign ↔ Text ↔ Speech. Powered by on-device computer vision, verified educational sign demonstrations, and lightweight neural networks.
          </p>

          {/* Three Core Pillars */}
          <div
            style={{
              display: 'flex',
              gap: 12,
              flexWrap: 'wrap',
              marginBottom: 24,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.82rem', color: 'var(--cyan)', fontWeight: 600, background: 'rgba(6,182,212,0.1)', padding: '4px 10px', borderRadius: 8, border: '1px solid rgba(6,182,212,0.2)' }}>
              <span>📷</span> Sign → Text
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.82rem', color: 'var(--primary)', fontWeight: 600, background: 'rgba(0,229,255,0.1)', padding: '4px 10px', borderRadius: 8, border: '1px solid rgba(0,229,255,0.2)' }}>
              <span>🤟</span> Speech → Sign
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.82rem', color: '#10b981', fontWeight: 600, background: 'rgba(16,185,129,0.1)', padding: '4px 10px', borderRadius: 8, border: '1px solid rgba(16,185,129,0.2)' }}>
              <span>💬</span> Real-time Conversation
            </div>
          </div>

          {/* Action CTAs */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 28 }}>
            <button
              className="btn btn-primary btn-lg"
              onClick={() => setPage('camera')}
              style={{ padding: '12px 22px', fontSize: '0.92rem', fontWeight: 700, letterSpacing: '0.02em' }}
            >
              START TRANSLATING →
            </button>
            <button
              className="btn btn-secondary btn-lg"
              onClick={() => setPage('learn')}
              style={{ padding: '12px 20px', fontSize: '0.92rem', fontWeight: 700, letterSpacing: '0.02em' }}
            >
              LEARN ISL 📚
            </button>
            <button
              className="btn btn-outline btn-lg"
              onClick={startDemoCamera}
              style={{ padding: '12px 20px', fontSize: '0.92rem', fontWeight: 600 }}
            >
              TRY LIVE DEMO 📷
            </button>
          </div>

          {/* System Status Indicators */}
          <div
            style={{
              display: 'flex',
              gap: 16,
              alignItems: 'center',
              flexWrap: 'wrap',
              fontSize: '0.82rem',
              color: 'var(--text2)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className={`dot ${online ? 'dot-green' : 'dot-red'}`} />
              <span>Backend {online ? 'Online' : 'Offline'}</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className={`dot ${modelLoaded ? 'dot-green' : 'dot-orange'}`} />
              <span>AI Model {modelLoaded ? 'Ready' : 'Missing'}</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="dot dot-green" />
              <span>Camera Ready</span>
            </div>
          </div>
        </div>

        {/* Right Column: Interactive Live Demo Preview Card */}
        <div>
          <div
            style={{
              background: '#090d16',
              borderRadius: 16,
              border: '1px solid rgba(0, 229, 255, 0.22)',
              overflow: 'hidden',
              boxShadow: '0 16px 36px rgba(0, 0, 0, 0.45)',
            }}
          >
            {/* Demo Card Top Header */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '12px 16px',
                background: 'rgba(15, 23, 42, 0.85)',
                borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#fff', letterSpacing: '0.5px' }}>
                  LIVE ISL DEMO
                </span>
              </div>
              <span
                className={`badge ${demoCameraActive ? 'badge-green' : 'badge-ghost'}`}
                style={{ fontSize: '0.72rem', padding: '2px 8px' }}
              >
                {demoCameraActive ? '● Camera Live' : 'Camera Ready'}
              </span>
            </div>

            {/* Video Preview / Inactive View */}
            <div
              style={{
                position: 'relative',
                height: 240,
                background: '#04060a',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
              }}
            >
              <video
                ref={videoRef}
                muted
                playsInline
                autoPlay
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  display: demoCameraActive ? 'block' : 'none',
                }}
              />
              <canvas ref={canvasRef} style={{ display: 'none' }} />

              {!demoCameraActive && (
                <div style={{ textAlign: 'center', padding: 20 }}>
                  <div style={{ fontSize: '2.4rem', marginBottom: 10 }}>📹</div>
                  <div style={{ fontWeight: 600, color: '#fff', fontSize: '0.95rem', marginBottom: 4 }}>
                    Camera Ready
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text3)', maxWidth: 220, marginBottom: 14 }}>
                    Click below to preview live real-time ISL recognition directly on this card.
                  </div>
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={startDemoCamera}
                    disabled={demoStarting}
                  >
                    {demoStarting ? 'Starting…' : '▶ Start Live Demo'}
                  </button>
                </div>
              )}

              {demoError && (
                <div
                  style={{
                    position: 'absolute',
                    bottom: 10,
                    left: 10,
                    right: 10,
                    background: 'rgba(239, 68, 68, 0.9)',
                    color: '#fff',
                    padding: '6px 10px',
                    borderRadius: 6,
                    fontSize: '0.75rem',
                    textAlign: 'center',
                  }}
                >
                  {demoError}
                </div>
              )}
            </div>

            {/* Real Detection Output Footer */}
            <div
              style={{
                padding: '12px 16px',
                background: 'rgba(10, 15, 29, 0.95)',
                borderTop: '1px solid rgba(255, 255, 255, 0.06)',
                fontSize: '0.85rem',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text3)', textTransform: 'uppercase' }}>
                    Detected Sign:
                  </div>
                  <div style={{ fontWeight: 700, fontSize: '1.05rem', color: demoPrediction?.word ? 'var(--primary)' : 'var(--text2)' }}>
                    {demoCameraActive
                      ? demoHandDetected && demoPrediction?.word
                        ? formatISLWord(demoPrediction.word)
                        : 'Searching hand…'
                      : 'HELLO (Sample)'}
                  </div>
                </div>

                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text3)', textTransform: 'uppercase' }}>
                    Confidence:
                  </div>
                  <div
                    style={{
                      fontWeight: 700,
                      fontSize: '1.05rem',
                      color: demoPrediction?.confidence ? confColor(demoPrediction.confidence) : 'var(--cyan)',
                    }}
                  >
                    {demoCameraActive
                      ? demoHandDetected && demoPrediction?.confidence
                        ? `${(demoPrediction.confidence * 100).toFixed(0)}%`
                        : '--%'
                      : '94%'}
                  </div>
                </div>
              </div>

              {demoCameraActive && (
                <div style={{ marginTop: 10, display: 'flex', justifyContent: 'flex-end' }}>
                  <button
                    type="button"
                    className="btn btn-xs btn-ghost"
                    onClick={stopDemoCamera}
                    style={{ fontSize: '0.75rem', padding: '3px 8px' }}
                  >
                    Stop Demo Feed
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ── CLEAN FEATURE GRID ─────────────────────────────────────────────────── */}
      <div style={{ marginBottom: 36 }}>
        <div style={{ textAlign: 'center', marginBottom: 28 }}>
          <div style={{ fontSize: '0.78rem', color: 'var(--primary)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1px' }}>
            Comprehensive Capabilities
          </div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 800, margin: '6px 0 0 0', color: '#fff' }}>
            Real-Time Assistive Communication
          </h2>
        </div>

        <div className="grid-3" style={{ gap: 20 }}>
          {features.map((f, idx) => (
            <div
              key={`${f.title}-${idx}`}
              className="card"
              onClick={() => setPage(f.id)}
              style={{
                cursor: 'pointer',
                padding: '22px 20px',
                transition: 'all 0.18s ease',
                display: 'flex',
                flexDirection: 'column',
                border: '1px solid rgba(255, 255, 255, 0.07)',
                background: 'rgba(15, 23, 42, 0.5)',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.transform = 'translateY(-3px)';
                e.currentTarget.style.borderColor = f.color;
              }}
              onMouseLeave={e => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.07)';
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                <span style={{ fontSize: '1.6rem' }}>{f.icon}</span>
                <div>
                  <div style={{ fontWeight: 700, fontSize: '0.95rem', color: '#fff' }}>{f.title}</div>
                  <div style={{ fontSize: '0.75rem', color: f.color, fontWeight: 600 }}>{f.subtitle}</div>
                </div>
              </div>

              <p style={{ fontSize: '0.84rem', color: 'var(--text2)', lineHeight: 1.5, margin: 0, flex: 1 }}>
                {f.desc}
              </p>

              <div
                style={{
                  marginTop: 14,
                  fontSize: '0.75rem',
                  color: 'var(--text3)',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                }}
              >
                <span>Launch module</span> →
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── INTERACTIVE SUPPORTED ISL SIGNS SECTION ─────────────────────────── */}
      <div
        className="card"
        style={{
          padding: '28px 24px',
          background: 'rgba(13, 19, 34, 0.75)',
          border: '1px solid rgba(0, 229, 255, 0.22)',
          borderRadius: 20,
          marginBottom: 40,
        }}
      >
        <div style={{ textAlign: 'center', marginBottom: 24 }}>
          <div
            style={{
              fontSize: '0.78rem',
              color: 'var(--primary)',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '1px',
            }}
          >
            Supported ISL Vocabulary
          </div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 800, margin: '6px 0 8px 0', color: '#fff' }}>
            Interactive Sign Demonstrations
          </h2>
          <p style={{ fontSize: '0.92rem', color: 'var(--text2)', maxWidth: 580, margin: '0 auto' }}>
            Click any supported ISL gesture to preview its authentic animation, execution technique, and training model classification.
          </p>
        </div>

        <div className="grid-2" style={{ gap: 24, alignItems: 'start' }}>
          {/* Left Column: Interactive 8-Sign Button Grid */}
          <div>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 10 }}>
              Select a Sign to Preview:
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 10,
              }}
            >
              {ISL_WORDS.map((w) => {
                const asset = SIGN_ASSETS[w] || {};
                const isSelected = selectedHomeSign === w;
                const isDynamic = asset.type === 'dynamic';

                return (
                  <div
                    key={w}
                    onClick={() => setSelectedHomeSign(w)}
                    style={{
                      cursor: 'pointer',
                      padding: '12px 14px',
                      borderRadius: 12,
                      background: isSelected
                        ? 'rgba(0, 229, 255, 0.12)'
                        : 'rgba(15, 23, 42, 0.65)',
                      border: isSelected
                        ? '1px solid var(--primary)'
                        : '1px solid rgba(255, 255, 255, 0.08)',
                      transition: 'all 0.16s ease',
                      boxShadow: isSelected ? '0 0 16px rgba(0, 229, 255, 0.2)' : 'none',
                    }}
                    onMouseEnter={(e) => {
                      if (!isSelected) {
                        e.currentTarget.style.borderColor = 'rgba(0, 229, 255, 0.4)';
                        e.currentTarget.style.transform = 'translateY(-2px)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isSelected) {
                        e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)';
                        e.currentTarget.style.transform = 'translateY(0)';
                      }
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                      <span style={{ fontWeight: 800, fontSize: '0.98rem', color: isSelected ? 'var(--primary)' : '#fff' }}>
                        {formatISLWord(w)}
                      </span>
                      <span
                        style={{
                          fontSize: '0.65rem',
                          fontWeight: 800,
                          padding: '2px 6px',
                          borderRadius: 4,
                          background: isDynamic ? 'rgba(168, 85, 247, 0.2)' : 'rgba(6, 182, 212, 0.2)',
                          color: isDynamic ? '#c084fc' : '#22d3ee',
                          border: isDynamic ? '1px solid rgba(168, 85, 247, 0.4)' : '1px solid rgba(6, 182, 212, 0.4)',
                        }}
                      >
                        {isDynamic ? 'DYNAMIC' : 'STATIC'}
                      </span>
                    </div>
                    <div
                      style={{
                        fontSize: '0.74rem',
                        color: 'var(--text3)',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {asset.description || 'Verified gesture'}
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{ marginTop: 16, display: 'flex', gap: 10 }}>
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={() => setPage('learn')}
                style={{ flex: 1, justifyContent: 'center' }}
              >
                Open Full Dictionary 📚
              </button>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setPage('translate')}
                style={{ flex: 1, justifyContent: 'center' }}
              >
                Speech → Sign 🤟
              </button>
            </div>
          </div>

          {/* Right Column: Live Interactive SignAnimation Viewport */}
          <div>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 10 }}>
              Demonstration Viewport:
            </div>
            <SignAnimation
              key={selectedHomeSign}
              sign={selectedHomeSign}
              word={selectedHomeSign}
              maxHeight={260}
              showControls={true}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
