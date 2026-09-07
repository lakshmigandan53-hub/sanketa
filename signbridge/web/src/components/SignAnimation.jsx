import { useState, useRef, useEffect, useCallback } from 'react';
import {
  SIGN_FRAMES_COUNT,
  getSignFrameData,
  getSignMeta,
  normalizeSignKey,
} from './signAnimations';
import { SIGN_ASSETS } from '../config/signAssets';

/**
 * Procedural SVG Hand Illustration Component
 * Renders an anatomically structured, illustrated hand and forearm
 * with continuous 30-frame kinematics across 8 authentic ISL gestures.
 */
function HandIllustration({ signKey, pose }) {
  if (!pose) return null;

  const {
    wristX = 0,
    wristY = 0,
    wristAngle = 0,
    scale = 1.0,
    thumb = { curl: 0, angle: 0, spread: 0 },
    index = { curl: 0, angle: 0, spread: 0 },
    middle = { curl: 0, angle: 0, spread: 0 },
    ring = { curl: 0, angle: 0, spread: 0 },
    pinky = { curl: 0, angle: 0, spread: 0 },
    supportHand = null,
  } = pose;

  // Finger parametric generator
  const renderFinger = (config) => {
    const {
      name,
      baseX,
      baseY,
      baseAngle,
      length,
      width,
      curl,
      spread,
      angleOffset,
    } = config;

    const totalAngleDeg = baseAngle + spread + angleOffset;
    const totalAngleRad = (totalAngleDeg * Math.PI) / 180;
    const c = Math.max(0, Math.min(1.0, curl));

    // When curl is low, finger is extended outward.
    // When curl is high, finger folds downward into palm.
    const l1 = length * 0.44;
    const l2 = length * 0.56;

    // Segment 1 (Proximal)
    const theta1 = totalAngleRad + c * 0.22;
    const s1 = 1 - c * 0.25;
    const p1x = baseX + Math.sin(theta1) * l1 * s1;
    const p1y = baseY - Math.cos(theta1) * l1 * s1;

    // Segment 2 (Intermediate + Distal)
    // Curled finger loops down toward palm
    const curlBending = c * 2.3; // Bends up to ~130 degrees when curled
    const theta2 = totalAngleRad + curlBending;
    const s2 = 1 - c * 0.52;
    const p2x = p1x + Math.sin(theta2) * l2 * s2;
    const p2y = p1y - Math.cos(theta2) * l2 * s2;

    // Widths
    const halfW0 = width * 0.5;
    const halfW1 = width * 0.44;
    const halfW2 = width * 0.36;

    // Normal vectors for thickness
    const nx1 = Math.cos(theta1);
    const ny1 = Math.sin(theta1);
    const nx2 = Math.cos(theta2);
    const ny2 = Math.sin(theta2);

    // Outline path
    const pathD = [
      `M ${baseX - nx1 * halfW0} ${baseY - ny1 * halfW0}`,
      `L ${p1x - nx1 * halfW1} ${p1y - ny1 * halfW1}`,
      `L ${p2x - nx2 * halfW2} ${p2y - ny2 * halfW2}`,
      // Rounded tip
      `A ${halfW2} ${halfW2} 0 0 1 ${p2x + nx2 * halfW2} ${p2y + ny2 * halfW2}`,
      `L ${p1x + nx1 * halfW1} ${p1y + ny1 * halfW1}`,
      `L ${baseX + nx1 * halfW0} ${baseY + ny1 * halfW0}`,
      'Z',
    ].join(' ');

    return (
      <g key={name} className={`finger finger-${name}`}>
        {/* Main finger segment body */}
        <path
          d={pathD}
          fill="url(#skinBodyGrad)"
          stroke="#b46d43"
          strokeWidth="1.6"
          strokeLinejoin="round"
        />

        {/* Highlight inner glow */}
        <path
          d={`M ${baseX} ${baseY} Q ${p1x} ${p1y} ${p2x} ${p2y}`}
          stroke="#ffe5d2"
          strokeWidth={width * 0.3}
          strokeLinecap="round"
          fill="none"
          opacity="0.65"
        />

        {/* Joint Creases */}
        {c < 0.65 ? (
          <>
            {/* PIP joint line */}
            <line
              x1={p1x - nx1 * (halfW1 * 0.7)}
              y1={p1y - ny1 * (halfW1 * 0.7)}
              x2={p1x + nx1 * (halfW1 * 0.7)}
              y2={p1y + ny1 * (halfW1 * 0.7)}
              stroke="#a65e36"
              strokeWidth="1.2"
              strokeLinecap="round"
              opacity="0.8"
            />
            {/* Fingernail at tip */}
            <ellipse
              cx={p2x - Math.sin(theta2) * (halfW2 * 0.6)}
              cy={p2y + Math.cos(theta2) * (halfW2 * 0.6)}
              rx={halfW2 * 0.65}
              ry={halfW2 * 0.8}
              transform={`rotate(${totalAngleDeg}, ${p2x}, ${p2y})`}
              fill="#fff4ee"
              stroke="#d58b62"
              strokeWidth="0.8"
              opacity="0.9"
            />
          </>
        ) : (
          /* Folded knuckle dome crease when curled */
          <path
            d={`M ${p1x - nx1 * halfW1 * 0.85} ${p1y - ny1 * halfW1 * 0.85} Q ${p1x} ${p1y - 2} ${p1x + nx1 * halfW1 * 0.85} ${p1y + ny1 * halfW1 * 0.85}`}
            stroke="#8d4520"
            strokeWidth="1.8"
            strokeLinecap="round"
            fill="none"
            opacity="0.9"
          />
        )}
      </g>
    );
  };

  // Thumb generator
  const renderThumb = () => {
    const isHelp = signKey === 'HELP';
    const c = Math.max(0, Math.min(1.0, thumb.curl));
    const angleOffset = thumb.angle || 0;

    // Special Thumbs-up for HELP
    if (isHelp) {
      // Thumb held firmly upright pointing toward sky
      const baseTx = 118;
      const baseTy = 160;
      const tipTx = 112;
      const tipTy = 96;

      return (
        <g key="thumb-help">
          <path
            d={`M ${baseTx} ${baseTy + 12} C 104 165, 96 142, 98 122 C 100 102, 106 94, 114 94 C 122 94, 126 106, 126 126 L 128 160 Z`}
            fill="url(#skinBodyGrad)"
            stroke="#b46d43"
            strokeWidth="1.8"
            strokeLinejoin="round"
          />
          {/* Thumb highlight */}
          <line
            x1={112}
            y1={baseTy}
            x2={tipTx}
            y2={tipTy + 8}
            stroke="#ffe5d2"
            strokeWidth="5"
            strokeLinecap="round"
            opacity="0.6"
          />
          {/* Thumbnail */}
          <ellipse
            cx={114}
            cy={103}
            rx={5.5}
            ry={7}
            fill="#fff4ee"
            stroke="#d58b62"
            strokeWidth="0.8"
          />
          {/* Joint crease */}
          <line
            x1={103}
            y1={130}
            x2={124}
            y2={132}
            stroke="#a65e36"
            strokeWidth="1.4"
            strokeLinecap="round"
          />
        </g>
      );
    }

    // Standard Thumb kinematics
    const baseTx = 118;
    const baseTy = 166;
    const baseAngleDeg = -42 + angleOffset + (thumb.spread || 0);
    const baseAngleRad = (baseAngleDeg * Math.PI) / 180;

    if (c < 0.5) {
      // Extended thumb
      const tLen = 52;
      const tipX = baseTx + Math.sin(baseAngleRad) * tLen;
      const tipY = baseTy - Math.cos(baseAngleRad) * tLen;

      return (
        <g key="thumb-extended">
          <path
            d={`M 124 186 C 114 182, 102 174, 96 160 L ${tipX - 8} ${tipY + 6} A 7.5 7.5 0 0 1 ${tipX + 6} ${tipY - 8} L 126 156 Z`}
            fill="url(#skinBodyGrad)"
            stroke="#b46d43"
            strokeWidth="1.8"
            strokeLinejoin="round"
          />
          <line
            x1={baseTx - 2}
            y1={baseTy + 6}
            x2={tipX}
            y2={tipY}
            stroke="#ffe5d2"
            strokeWidth="5"
            strokeLinecap="round"
            opacity="0.6"
          />
          {/* Thumbnail */}
          <ellipse
            cx={tipX}
            cy={tipY}
            rx={5}
            ry={6.5}
            transform={`rotate(${baseAngleDeg}, ${tipX}, ${tipY})`}
            fill="#fff4ee"
            stroke="#d58b62"
            strokeWidth="0.8"
          />
        </g>
      );
    } else {
      // Curled thumb wrapping across fingers / fist
      const foldX = 142 + (angleOffset * 0.4);
      const foldY = 162;

      return (
        <g key="thumb-curled">
          {/* Thenar pad base */}
          <path
            d="M 116 188 C 104 180, 102 165, 112 154 L 126 150 Z"
            fill="url(#skinBodyGrad)"
            stroke="#b46d43"
            strokeWidth="1.6"
          />
          {/* Wrapped thumb bar across fist */}
          <path
            d={`M 112 156 C 118 150, 134 148, ${foldX} ${foldY - 4} A 8 8 0 0 1 ${foldX - 2} ${foldY + 12} C 130 168, 116 168, 112 166 Z`}
            fill="url(#skinBodyGrad)"
            stroke="#a65e36"
            strokeWidth="1.8"
            strokeLinejoin="round"
          />
          {/* Thumb knuckle crease */}
          <line
            x1={120}
            y1={154}
            x2={126}
            y2={164}
            stroke="#8d4520"
            strokeWidth="1.4"
            strokeLinecap="round"
          />
        </g>
      );
    }
  };

  // Support Hand for HELP
  const renderSupportHand = () => {
    if (!supportHand?.active) return null;
    const sy = 220 + (supportHand.y || 0);
    const sx = 160 + (supportHand.x || 0);

    return (
      <g key="support-hand" transform={`translate(${sx - 160}, 0)`}>
        {/* Shadow cast under dominant hand */}
        <ellipse cx={160} cy={sy + 10} rx={60} ry={12} fill="rgba(0,0,0,0.12)" />

        {/* Support forearm entering from lower left */}
        <path
          d={`M 50 280 L 105 ${sy + 20} L 125 ${sy + 16} L 70 280 Z`}
          fill="url(#skinForearmGrad)"
          stroke="#b46d43"
          strokeWidth="1.6"
        />

        {/* Flat Open Support Palm */}
        <path
          d={`M 105 ${sy + 20} C 115 ${sy + 24}, 145 ${sy + 25}, 185 ${sy + 23} C 215 ${sy + 21}, 235 ${sy + 14}, 242 ${sy + 8} C 245 ${sy + 2}, 238 ${sy - 2}, 225 ${sy} L 180 ${sy + 6} C 145 ${sy + 8}, 122 ${sy + 6}, 112 ${sy + 12} Z`}
          fill="url(#skinBodyGrad)"
          stroke="#a65e36"
          strokeWidth="1.8"
          strokeLinejoin="round"
        />

        {/* Fingers of support hand held flat together */}
        <g stroke="#a65e36" strokeWidth="1.2" opacity="0.75">
          <line x1={175} y1={sy + 6} x2={226} y2={sy + 1} />
          <line x1={180} y1={sy + 10} x2={234} y2={sy + 4} />
          <line x1={182} y1={sy + 14} x2={236} y2={sy + 8} />
        </g>
      </g>
    );
  };

  return (
    <svg
      viewBox="0 0 320 300"
      style={{
        width: '100%',
        height: '100%',
        display: 'block',
        overflow: 'visible',
      }}
      aria-label={`ISL Sign Animation demonstration for ${signKey}`}
    >
      <defs>
        {/* Soft realistic skin gradients */}
        <linearGradient id="skinBodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#fee0cb" />
          <stop offset="50%" stopColor="#f7bc95" />
          <stop offset="100%" stopColor="#e59869" />
        </linearGradient>

        <linearGradient id="skinForearmGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#e49566" />
          <stop offset="40%" stopColor="#f9c19b" />
          <stop offset="70%" stopColor="#fcd3b4" />
          <stop offset="100%" stopColor="#df8d5b" />
        </linearGradient>

        <radialGradient id="palmContourGrad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#ffe9dc" />
          <stop offset="75%" stopColor="#f4b288" />
          <stop offset="100%" stopColor="#df8b59" />
        </radialGradient>

        <filter id="handShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="6" stdDeviation="6" floodColor="#091e42" floodOpacity="0.10" />
        </filter>
      </defs>

      {/* Support hand (underneath dominant hand) */}
      {renderSupportHand()}

      {/* Main Dominant Hand Group with dynamic Wrist Kinematics */}
      <g
        transform={`translate(${wristX}, ${wristY}) rotate(${wristAngle}, 160, 200) scale(${scale})`}
        filter="url(#handShadow)"
      >
        {/* ── Forearm ── */}
        <path
          d="M 134 290 L 137 208 C 137 202, 140 197, 145 195 L 175 195 C 180 197, 183 202, 183 208 L 186 290 Z"
          fill="url(#skinForearmGrad)"
          stroke="#b46d43"
          strokeWidth="1.8"
          strokeLinejoin="round"
        />

        {/* Forearm base subtle shadow */}
        <ellipse cx={160} cy={286} rx={26} ry={5} fill="#d57e4b" opacity="0.3" />

        {/* ── Palm ── */}
        <path
          d="M 145 195 C 128 190, 114 176, 116 160 C 117 150, 125 142, 132 134 C 138 127, 150 123, 160 122 C 172 121, 184 125, 192 130 C 200 135, 207 142, 206 154 C 205 170, 194 188, 175 195 Z"
          fill="url(#palmContourGrad)"
          stroke="#b46d43"
          strokeWidth="2.0"
          strokeLinejoin="round"
        />

        {/* Anatomical Palm Creases (Heart Line & Head Line) */}
        <g fill="none" stroke="#a65e36" strokeWidth="1.3" strokeLinecap="round" opacity="0.75">
          <path d="M 132 152 Q 155 158 184 148" />
          <path d="M 128 164 Q 152 174 186 165" />
          <path d="M 120 162 Q 138 180 152 192" />
        </g>

        {/* ── 4 Fingers (Ordered from Pinky to Index for natural depth) ── */}
        {renderFinger({
          name: 'pinky',
          baseX: 198,
          baseY: 137,
          baseAngle: 12,
          length: 53,
          width: 12,
          curl: pinky.curl,
          spread: pinky.spread || 0,
          angleOffset: pinky.angle || 0,
        })}

        {renderFinger({
          name: 'ring',
          baseX: 180,
          baseY: 128,
          baseAngle: 5,
          length: 66,
          width: 14,
          curl: ring.curl,
          spread: ring.spread || 0,
          angleOffset: ring.angle || 0,
        })}

        {renderFinger({
          name: 'middle',
          baseX: 160,
          baseY: 123,
          baseAngle: 0,
          length: 75,
          width: 15,
          curl: middle.curl,
          spread: middle.spread || 0,
          angleOffset: middle.angle || 0,
        })}

        {renderFinger({
          name: 'index',
          baseX: 138,
          baseY: 131,
          baseAngle: -6,
          length: 65,
          width: 14.5,
          curl: index.curl,
          spread: index.spread || 0,
          angleOffset: index.angle || 0,
        })}

        {/* ── Knuckle Base Domes ── */}
        <g fill="#f3af84" opacity="0.5">
          <circle cx={138} cy={133} r={4.5} />
          <circle cx={160} cy={125} r={5} />
          <circle cx={180} cy={130} r={4.5} />
          <circle cx={198} cy={139} r={3.8} />
        </g>

        {/* ── Thumb (Rendered over palm / fingers as appropriate) ── */}
        {renderThumb()}
      </g>
    </svg>
  );
}

/**
 * Main Universal SignAnimation Component
 * Accepts sign/word, autoPlay/autoplay, loop, maxHeight, showControls, onEnded.
 */
export default function SignAnimation({
  sign,
  word,
  autoplay = true,
  autoPlay = true,
  loop = true,
  showControls = true,
  maxHeight = 270,
  className = '',
  style = {},
  onEnded = null,
  onFrameChange = null,
  hideLabel = false,
}) {
  const targetSign = normalizeSignKey(sign || word || 'HELLO');
  const meta = getSignMeta(targetSign);
  const asset = SIGN_ASSETS[targetSign] || {};
  const shouldAutoPlay = autoplay ?? autoPlay ?? true;

  const localGif = asset.gif || null;
  const localVideo = asset.video || null;
  const hasRealAsset = Boolean(localGif || localVideo);

  const [viewMode, setViewMode] = useState(hasRealAsset ? 'media' : 'illustration');
  const [mediaError, setMediaError] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(shouldAutoPlay);

  const videoElemRef = useRef(null);
  const reqIdRef = useRef(null);
  const lastTimeRef = useRef(0);
  const frameRateMs = 52; // ~19.2 fps for authentic, readable gesture comprehension

  // Reset media error and initial view mode when target sign changes
  useEffect(() => {
    setMediaError(false);
    setViewMode(hasRealAsset ? 'media' : 'illustration');
    setCurrentFrame(0);
    setIsPlaying(shouldAutoPlay);
  }, [targetSign, hasRealAsset, shouldAutoPlay]);

  const showMedia = hasRealAsset && !mediaError && viewMode === 'media';

  // Step to a specific frame
  const goToFrame = useCallback((f) => {
    const clamped = Math.max(0, Math.min(SIGN_FRAMES_COUNT - 1, f));
    setCurrentFrame(clamped);
    if (onFrameChange) onFrameChange(clamped);
  }, [onFrameChange]);

  // Replay from start
  const handleReplay = useCallback(() => {
    if (showMedia && videoElemRef.current) {
      videoElemRef.current.currentTime = 0;
      videoElemRef.current.play().catch(() => {});
      setIsPlaying(true);
    } else {
      setCurrentFrame(0);
      setIsPlaying(true);
      if (onFrameChange) onFrameChange(0);
    }
  }, [showMedia, onFrameChange]);

  // Toggle play/pause
  const togglePlay = useCallback(() => {
    setIsPlaying((prev) => {
      const nextState = !prev;
      if (showMedia && videoElemRef.current) {
        if (nextState) {
          videoElemRef.current.play().catch(() => {});
        } else {
          videoElemRef.current.pause();
        }
      }
      return nextState;
    });
  }, [showMedia]);

  // Smooth requestAnimationFrame ticker across 30 logical frames for procedural hand
  useEffect(() => {
    if (showMedia || !isPlaying) {
      if (reqIdRef.current) cancelAnimationFrame(reqIdRef.current);
      return;
    }

    const tick = (time) => {
      if (!lastTimeRef.current) lastTimeRef.current = time;
      const delta = time - lastTimeRef.current;

      if (delta >= frameRateMs) {
        lastTimeRef.current = time;
        setCurrentFrame((prevFrame) => {
          if (prevFrame >= SIGN_FRAMES_COUNT - 1) {
            if (loop) {
              if (onFrameChange) onFrameChange(0);
              return 0;
            } else {
              setIsPlaying(false);
              if (onEnded) onEnded();
              return prevFrame;
            }
          }
          const next = prevFrame + 1;
          if (onFrameChange) onFrameChange(next);
          return next;
        });
      }

      reqIdRef.current = requestAnimationFrame(tick);
    };

    reqIdRef.current = requestAnimationFrame(tick);
    return () => {
      if (reqIdRef.current) cancelAnimationFrame(reqIdRef.current);
    };
  }, [showMedia, isPlaying, loop, onEnded, onFrameChange]);

  // Get current frame's interpolated kinematic pose
  const pose = getSignFrameData(targetSign, currentFrame);

  return (
    <div
      className={`sign-animation-card ${className}`}
      style={{
        borderRadius: 16,
        overflow: 'hidden',
        background: '#0d1322',
        border: '1px solid rgba(255, 255, 255, 0.12)',
        boxShadow: '0 8px 30px rgba(0, 0, 0, 0.4)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        width: '100%',
        position: 'relative',
        userSelect: 'none',
        ...style,
      }}
    >
      {/* ── Educational Viewport: Clean light neutral surface ── */}
      <div
        style={{
          width: '100%',
          height: maxHeight,
          maxHeight: maxHeight,
          background: 'linear-gradient(180deg, #ffffff 0%, #f4f6f9 100%)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
          padding: '10px 14px',
          boxSizing: 'border-box',
          overflow: 'hidden',
        }}
      >
        {/* Top Header: Sign Name Badge & Real-Time Status */}
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: 12,
            right: 12,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            zIndex: 5,
            pointerEvents: 'none',
          }}
        >
          {/* Sign Label Badge + Static/Dynamic Pill */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div
              style={{
                background: 'rgba(15, 23, 42, 0.92)',
                color: '#ffffff',
                padding: '4px 12px',
                borderRadius: 8,
                fontSize: '0.80rem',
                fontWeight: 800,
                letterSpacing: '0.6px',
                boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              <span style={{ color: '#38bdf8' }}>ISL</span>
              <span>{hideLabel ? 'SIGN DEMO' : meta.label.toUpperCase()}</span>
            </div>

            {/* Static / Dynamic Badge */}
            <span
              style={{
                background: meta.type === 'dynamic' ? 'rgba(147, 51, 234, 0.90)' : 'rgba(8, 145, 178, 0.90)',
                color: '#ffffff',
                padding: '3px 8px',
                borderRadius: 6,
                fontSize: '0.70rem',
                fontWeight: 800,
                letterSpacing: '0.5px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.15)',
              }}
            >
              {meta.type ? meta.type.toUpperCase() : 'STATIC'}
            </span>
          </div>

          {/* Right Mode Indicator: Real Sign Asset vs Educational Illustration */}
          <div>
            {showMedia ? (
              <span
                style={{
                  background: 'rgba(15, 23, 42, 0.88)',
                  color: '#22c55e',
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                  padding: '4px 10px',
                  borderRadius: 8,
                  fontSize: '0.72rem',
                  fontWeight: 700,
                  letterSpacing: '0.5px',
                  boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
                  border: '1px solid rgba(34, 197, 94, 0.3)',
                }}
              >
                📹 Real Sign Asset
              </span>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span
                  style={{
                    background: 'rgba(15, 23, 42, 0.88)',
                    color: '#38bdf8',
                    padding: '4px 10px',
                    borderRadius: 8,
                    fontSize: '0.72rem',
                    fontWeight: 700,
                    letterSpacing: '0.4px',
                    boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
                    border: '1px solid rgba(56, 189, 248, 0.25)',
                  }}
                >
                  Educational sign illustration
                </span>
                <span
                  style={{
                    background: 'rgba(15, 23, 42, 0.88)',
                    color: '#94a3b8',
                    fontFamily: 'ui-monospace, monospace',
                    padding: '4px 8px',
                    borderRadius: 8,
                    fontSize: '0.70rem',
                    fontWeight: 700,
                    border: '1px solid rgba(255,255,255,0.1)',
                  }}
                >
                  F {String(currentFrame + 1).padStart(2, '0')}/{SIGN_FRAMES_COUNT}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Viewport Content: Real Media vs Procedural SVG Hand */}
        <div
          style={{
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            paddingTop: 24,
            paddingBottom: 16,
            boxSizing: 'border-box',
          }}
        >
          {showMedia ? (
            localGif ? (
              <img
                key={`gif-${targetSign}`}
                src={localGif}
                alt={meta.label}
                onError={() => {
                  setMediaError(true);
                  setViewMode('illustration');
                }}
                style={{
                  maxHeight: '100%',
                  maxWidth: '100%',
                  objectFit: 'contain',
                  borderRadius: 8,
                  filter: 'drop-shadow(0 4px 12px rgba(0,0,0,0.12))',
                }}
              />
            ) : (
              <video
                key={`vid-${targetSign}`}
                ref={videoElemRef}
                src={localVideo}
                autoPlay={shouldAutoPlay}
                loop={loop}
                muted
                playsInline
                onError={() => {
                  setMediaError(true);
                  setViewMode('illustration');
                }}
                style={{
                  maxHeight: '100%',
                  maxWidth: '100%',
                  objectFit: 'contain',
                  borderRadius: 8,
                  filter: 'drop-shadow(0 4px 12px rgba(0,0,0,0.12))',
                }}
              />
            )
          ) : (
            <HandIllustration signKey={targetSign} pose={pose} />
          )}
        </div>

        {/* Micro-Motion Guide Hint at bottom of viewport */}
        <div
          style={{
            position: 'absolute',
            bottom: 8,
            left: 12,
            right: 12,
            display: 'flex',
            justifyContent: 'center',
            pointerEvents: 'none',
            zIndex: 4,
          }}
        >
          <span
            style={{
              background: 'rgba(255, 255, 255, 0.90)',
              color: '#334155',
              backdropFilter: 'blur(4px)',
              padding: '3px 12px',
              borderRadius: 12,
              fontSize: '0.74rem',
              fontWeight: 600,
              boxShadow: '0 1px 4px rgba(0,0,0,0.10)',
              border: '1px solid rgba(0,0,0,0.06)',
            }}
          >
            {showMedia ? meta.description : (pose?.actionText || meta.description)}
          </span>
        </div>
      </div>

      {/* ── Card Footer Controls inside Dark SANKETA UI ── */}
      {showControls && (
        <div
          style={{
            width: '100%',
            padding: '10px 14px',
            background: '#090d16',
            borderTop: '1px solid rgba(255, 255, 255, 0.08)',
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            boxSizing: 'border-box',
          }}
        >
          {/* Frame Progress Scrubber (when in illustration mode) */}
          {!showMedia && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <input
                type="range"
                min={0}
                max={SIGN_FRAMES_COUNT - 1}
                value={currentFrame}
                onChange={(e) => {
                  setIsPlaying(false);
                  goToFrame(Number(e.target.value));
                }}
                aria-label="Frame scrubber"
                style={{
                  width: '100%',
                  accentColor: 'var(--primary, #00e5ff)',
                  cursor: 'pointer',
                  height: 4,
                }}
              />
            </div>
          )}

          {/* Action Buttons: Play/Pause, Replay, Mode Switcher */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '0.82rem',
              flexWrap: 'wrap',
              gap: 8,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontWeight: 700, color: '#f1f5f9' }}>
                {hideLabel ? 'ISL Sign' : meta.label}
              </span>

              {/* Toggle between real media and procedural hand illustration when both available */}
              {hasRealAsset && !mediaError && (
                <button
                  type="button"
                  className="btn btn-xs btn-outline"
                  onClick={() => setViewMode((m) => (m === 'media' ? 'illustration' : 'media'))}
                  style={{ fontSize: '0.70rem', padding: '2px 8px' }}
                >
                  {viewMode === 'media' ? '✋ View Hand Skeleton' : '📹 View Real Media'}
                </button>
              )}
            </div>

            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              {/* Step Prev Frame (in illustration mode) */}
              {!showMedia && (
                <button
                  type="button"
                  className="btn btn-sm btn-ghost"
                  onClick={() => {
                    setIsPlaying(false);
                    goToFrame(currentFrame > 0 ? currentFrame - 1 : SIGN_FRAMES_COUNT - 1);
                  }}
                  title="Previous frame"
                  style={{ padding: '3px 8px', fontSize: '0.75rem' }}
                >
                  ⏮
                </button>
              )}

              {/* Play / Pause Toggle */}
              <button
                type="button"
                className="btn btn-sm btn-ghost"
                onClick={togglePlay}
                title={isPlaying ? 'Pause animation' : 'Play animation'}
                style={{
                  padding: '3px 10px',
                  fontSize: '0.78rem',
                  fontWeight: 600,
                  color: isPlaying ? '#38bdf8' : '#f8fafc',
                }}
              >
                {isPlaying ? '⏸ Pause' : '▶ Play'}
              </button>

              {/* Replay Button */}
              <button
                type="button"
                className="btn btn-sm btn-ghost"
                onClick={handleReplay}
                title="Replay from start"
                style={{ padding: '3px 10px', fontSize: '0.78rem', fontWeight: 600 }}
              >
                ⟳ Replay
              </button>

              {/* Step Next Frame (in illustration mode) */}
              {!showMedia && (
                <button
                  type="button"
                  className="btn btn-sm btn-ghost"
                  onClick={() => {
                    setIsPlaying(false);
                    goToFrame(currentFrame < SIGN_FRAMES_COUNT - 1 ? currentFrame + 1 : 0);
                  }}
                  title="Next frame"
                  style={{ padding: '3px 8px', fontSize: '0.75rem' }}
                >
                  ⏭
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
