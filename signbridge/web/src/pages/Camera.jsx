import { useState, useRef, useEffect, useCallback } from 'react';
import {
  apiTranslateLandmarks,
  apiTranslateSequence,
  apiTranslateBurst,
  apiResetBuffer,
  apiModelInfo,
  speak,
  SIGN_ASSETS,
  addHistory,
  getHistory,
  clearHistory,
  confColor,
  translateWord,
  formatISLWord,
  getSettings,
  API_BASE,
} from '../services';

// Official MediaPipe Hand Connections topology (21 landmark pairs)
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [9, 10], [10, 11], [11, 12],
  // Ring finger
  [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm connections
  [5, 9], [9, 13], [13, 17],
];

// Fingertip landmark indices (Thumb, Index, Middle, Ring, Pinky)
const FINGERTIP_INDICES = [4, 8, 12, 16, 20];

export default function Camera({ lang }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Camera lifecycle state: 'OFFLINE' | 'STARTING' | 'LIVE' | 'PERMISSION_REQUIRED' | 'ERROR'
  const [cameraStatus, setCameraStatus] = useState('OFFLINE');
  const [cameraError, setCameraError] = useState('');
  const [isMirrored, setIsMirrored] = useState(true);

  // MediaPipe state
  const [mediaPipeStatus, setMediaPipeStatus] = useState('INITIALIZING'); // 'INITIALIZING' | 'READY' | 'ERROR'
  const [handsDetectedCount, setHandsDetectedCount] = useState(0);
  const [handednessLabel, setHandednessLabel] = useState('');
  const [trackingFps, setTrackingFps] = useState(0);
  const [debugMode, setDebugMode] = useState(true);

  // Backend model info & status
  const [backendStatus, setBackendStatus] = useState('CHECKING');
  const [modelInfo, setModelInfo] = useState(null);

  // Live recognition pipeline state
  const [liveRecognitionActive, setLiveRecognitionActive] = useState(true);
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [lastApiError, setLastApiError] = useState('');
  const [stablePrediction, setStablePrediction] = useState('');
  const [stabilityState, setStabilityState] = useState('IDLE'); // 'IDLE' | 'TRACKING' | 'STABLE' | 'NO_HAND'
  const [stabilityRatio, setStabilityRatio] = useState(0);

  // Dynamic sequence recorder state
  const [recordingDynamic, setRecordingDynamic] = useState(false);
  const [recordProgress, setRecordProgress] = useState(0);

  // Sentence & History State
  const [sentenceWords, setSentenceWords] = useState([]);
  const [historyList, setHistoryList] = useState([]);

  // Loop & Tracking Refs
  const streamRef = useRef(null);
  const handsRef = useRef(null);
  const animFrameIdRef = useRef(null);
  const lastInferenceTimeRef = useRef(0);
  const isInferringRef = useRef(false);

  // FPS calculation refs
  const fpsFrameCountRef = useRef(0);
  const fpsLastTimeRef = useRef(performance.now());

  // Handlers refs to avoid stale closures and satisfy exhaustive-deps
  const handleResultsRef = useRef(null);
  const dispatchInferenceRef = useRef(null);

  // Landmark smoothing ref (EMA per hand)
  // smoothedLandmarksRef.current[handIndex] = Array(21).fill({x, y, z})
  const smoothedLandmarksRef = useRef([]);
  const latestLandmarksRef = useRef(null);

  // Stability & Cooldown refs
  const rollingBufferRef = useRef([]); // Last 10 predictions
  const lastAcceptedWordRef = useRef('');
  const lastAcceptedTimeRef = useRef(0);

  // ── Load Backend Model Info & History ───────────────────────────────────────
  useEffect(() => {
    setHistoryList(getHistory());

    apiModelInfo()
      .then((info) => {
        setModelInfo(info);
        setBackendStatus('ONLINE');
      })
      .catch((err) => {
        console.warn('Backend model info check failed:', err);
        setBackendStatus('OFFLINE');
      });
  }, []);

  // ── Initialize MediaPipe Hands Instance ─────────────────────────────────────
  useEffect(() => {
    let isCancelled = false;

    const initMediaPipe = async () => {
      try {
        const HandsConstructor = window.Hands;
        if (!HandsConstructor) {
          throw new Error('MediaPipe Hands library not loaded on window.');
        }

        const hands = new HandsConstructor({
          locateFile: (file) => {
            // Local assets served from public/mediapipe/
            return `/mediapipe/${file}`;
          },
        });

        hands.setOptions({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        hands.onResults((results) => {
          if (!isCancelled && handleResultsRef.current) {
            handleResultsRef.current(results);
          }
        });

        handsRef.current = hands;
        setMediaPipeStatus('READY');
      } catch (err) {
        console.warn('Local MediaPipe init failed, trying CDN fallback:', err);
        // Fallback to CDN if local files encounter an issue
        try {
          const HandsConstructor = window.Hands;
          if (!HandsConstructor) throw err;
          const hands = new HandsConstructor({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
          });
          hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
          hands.onResults((results) => {
            if (!isCancelled && handleResultsRef.current) {
              handleResultsRef.current(results);
            }
          });
          handsRef.current = hands;
          setMediaPipeStatus('READY');
        } catch (cdnErr) {
          console.error('MediaPipe initialization completely failed:', cdnErr);
          setMediaPipeStatus('ERROR');
        }
      }
    };

    initMediaPipe();

    return () => {
      isCancelled = true;
      if (handsRef.current) {
        try {
          handsRef.current.close();
        } catch (e) {}
        handsRef.current = null;
      }
    };
  }, []);

  // ── Stop Camera Stream & Tracking Loop ──────────────────────────────────────
  const stopCamera = useCallback(() => {
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
      animFrameIdRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (e) {}
      });
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    smoothedLandmarksRef.current = [];
    rollingBufferRef.current = [];
    isInferringRef.current = false;

    setCameraStatus('OFFLINE');
    setHandsDetectedCount(0);
    setHandednessLabel('');
    setCurrentPrediction(null);
    setStabilityState('IDLE');
    setStabilityRatio(0);
  }, []);

  // ── Start Camera Stream ─────────────────────────────────────────────────────
  const startCamera = async () => {
    stopCamera();
    setCameraError('');
    setCameraStatus('STARTING');

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Webcam API is not supported in this browser. Please use Google Chrome, Edge, or Firefox.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        setCameraStatus('LIVE');
        setCameraError('');
        await apiResetBuffer().catch(() => {});

        // Start continuous MediaPipe tracking loop
        startTrackingLoop();
      }
    } catch (err) {
      console.error('Camera startup error:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setCameraStatus('PERMISSION_REQUIRED');
        setCameraError('Camera permission required. Please click the lock or camera icon in your address bar and allow camera access.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setCameraStatus('ERROR');
        setCameraError('No webcam hardware detected. Please connect a webcam and try again.');
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        setCameraStatus('ERROR');
        setCameraError('Webcam is currently in use by another application. Please close other camera apps and retry.');
      } else {
        setCameraStatus('ERROR');
        setCameraError(err.message || 'Failed to start webcam.');
      }
    }
  };

  // Clean up stream and animation frames on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  // ── Continuous MediaPipe Tracking Loop (Native ~30-60 FPS) ──────────────────
  const startTrackingLoop = useCallback(() => {
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
    }

    const loop = async () => {
      const video = videoRef.current;
      const hands = handsRef.current;

      if (
        video &&
        hands &&
        video.readyState >= 2 &&
        !video.paused &&
        !video.ended &&
        video.videoWidth > 0
      ) {
        try {
          await hands.send({ image: video });
        } catch (err) {
          // Frame dropped or busy
        }
      }

      // Calculate live FPS
      fpsFrameCountRef.current += 1;
      const now = performance.now();
      if (now - fpsLastTimeRef.current >= 1000) {
        setTrackingFps(fpsFrameCountRef.current);
        fpsFrameCountRef.current = 0;
        fpsLastTimeRef.current = now;
      }

      animFrameIdRef.current = requestAnimationFrame(loop);
    };

    animFrameIdRef.current = requestAnimationFrame(loop);
  }, []);

  // ── MediaPipe Results Handler & Canvas Skeleton Drawing ─────────────────────
  const handleMediaPipeResults = useCallback(
    (results) => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;

      // Match canvas internal resolution to video's actual dimensions
      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, width, height);

      const rawHands = results.multiHandLandmarks || [];
      const numHands = rawHands.length;
      setHandsDetectedCount(numHands);

      if (numHands === 0) {
        setHandednessLabel('');
        setStabilityState('NO_HAND');
        setStabilityRatio(0);
        setStablePrediction('');
        smoothedLandmarksRef.current = [];
        latestLandmarksRef.current = null;
        return;
      }

      // Store canonical 63 landmark coordinates [lm0.x, lm0.y, lm0.z, ...] for recognition
      if (rawHands.length > 0 && rawHands[0].length === 21) {
        const lms63 = [];
        for (let i = 0; i < 21; i++) {
          const pt = rawHands[0][i];
          // When mirrored (default selfie webcam preview), invert x so it matches
          // the training coordinate space recorded with OpenCV cv2.flip(frame, 1)
          const xCoord = isMirrored ? (1.0 - pt.x) : pt.x;
          lms63.push(xCoord, pt.y, pt.z);
        }
        latestLandmarksRef.current = lms63;
      } else {
        latestLandmarksRef.current = null;
      }

      // Format Handedness label (e.g. "RIGHT HAND" or "LEFT HAND" or "2 HANDS")
      const handednessArray = results.multiHandedness || [];
      if (handednessArray.length === 1) {
        const hLabel = handednessArray[0]?.label || 'Hand';
        setHandednessLabel(`${hLabel.toUpperCase()} HAND`);
      } else if (handednessArray.length >= 2) {
        setHandednessLabel('LEFT & RIGHT HANDS');
      }

      // Landmark smoothing with Exponential Moving Average (EMA)
      const smoothedHands = rawHands.map((landmarks, hIdx) => {
        let prev = smoothedLandmarksRef.current[hIdx];
        if (!prev || prev.length !== 21) {
          prev = landmarks;
        }

        const alpha = 0.82; // 0.82 = fast responsiveness with high jitter suppression
        const smoothed = landmarks.map((curr, idx) => ({
          x: alpha * curr.x + (1 - alpha) * prev[idx].x,
          y: alpha * curr.y + (1 - alpha) * prev[idx].y,
          z: alpha * curr.z + (1 - alpha) * prev[idx].z,
        }));

        smoothedLandmarksRef.current[hIdx] = smoothed;
        return smoothed;
      });

      // ── Draw Complete 21-Landmark Hand Skeleton for Each Detected Hand ───────
      smoothedHands.forEach((landmarks, hIdx) => {
        const isSecondHand = hIdx === 1;
        const mainColor = isSecondHand ? '#a855f7' : '#00e5ff';
        const jointColor = isSecondHand ? '#d8b4fe' : '#e0f2fe';
        const tipColor = isSecondHand ? '#ec4899' : '#10b981';

        // 1. Draw Skeleton Connections
        ctx.lineWidth = 3.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = mainColor;
        ctx.shadowColor = mainColor;
        ctx.shadowBlur = 8;

        HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
          const p1 = landmarks[startIdx];
          const p2 = landmarks[endIdx];

          // Screen display coordinates (mirrored horizontally to match mirrored video preview)
          const x1 = isMirrored ? (1 - p1.x) * width : p1.x * width;
          const y1 = p1.y * height;
          const x2 = isMirrored ? (1 - p2.x) * width : p2.x * width;
          const y2 = p2.y * height;

          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        });

        // 2. Draw 21 Landmark Nodes
        landmarks.forEach((pt, idx) => {
          const x = isMirrored ? (1 - pt.x) * width : pt.x * width;
          const y = pt.y * height;
          const isFingertip = FINGERTIP_INDICES.includes(idx);
          const isWrist = idx === 0;

          // Node radius
          const radius = isWrist ? 6.5 : isFingertip ? 5.5 : 4.0;

          // Outer halo
          ctx.beginPath();
          ctx.arc(x, y, radius + 2, 0, 2 * Math.PI);
          ctx.fillStyle = 'rgba(15, 23, 42, 0.6)';
          ctx.fill();

          // Core node
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, 2 * Math.PI);
          ctx.fillStyle = isFingertip ? tipColor : isWrist ? '#38bdf8' : jointColor;
          ctx.shadowBlur = isFingertip ? 12 : 6;
          ctx.fill();
        });

        // Reset shadow
        ctx.shadowBlur = 0;
      });

      // ── Trigger Throttled Backend Recognition (every ~160ms) ────────────────
      const now = performance.now();
      if (
        liveRecognitionActive &&
        !recordingDynamic &&
        !isInferringRef.current &&
        now - lastInferenceTimeRef.current >= 160
      ) {
        lastInferenceTimeRef.current = now;
        if (dispatchInferenceRef.current) {
          dispatchInferenceRef.current();
        }
      }
    },
    [isMirrored, liveRecognitionActive, recordingDynamic]
  );

  handleResultsRef.current = handleMediaPipeResults;

  // ── Dispatch Recognition to Local Backend Model ─────────────────────────────
  const dispatchInference = async () => {
    if (isInferringRef.current) return;
    const lms = latestLandmarksRef.current;
    if (!lms || lms.length !== 63) return;

    isInferringRef.current = true;
    setLastApiError('');

    try {
      // ── Step 1: Log outgoing landmarks for debugging ──
      console.debug(
        '[SANKETA] Sending landmarks to /translate/landmarks | count:', lms.length,
        '| first 6:', lms.slice(0, 6).map(v => v.toFixed(4))
      );

      // ── Step 2: Call /translate/landmarks with 63-feature vector ──
      let data = null;
      let httpError = null;
      try {
        data = await apiTranslateLandmarks(lms);
        console.debug('[SANKETA] /translate/landmarks response:', data);
      } catch (lmsErr) {
        httpError = lmsErr.message || 'Network error';
        console.error('[SANKETA] /translate/landmarks failed:', lmsErr);
      }

      // ── Step 3: If API call failed, show error and mark backend offline ──
      if (!data) {
        const errMsg = httpError || 'No response from backend';
        setLastApiError(errMsg);
        setBackendStatus('OFFLINE');
        // Show an error prediction state so the UI doesn't stay stuck on "Analyzing"
        setCurrentPrediction({
          word: '__error__',
          confidence: 0,
          type: 'static',
          model: 'isl_model.keras',
          modelName: 'SANKETA Landmark Model',
          message: `Backend error: ${errMsg}`,
          error: true,
        });
        return;
      }

      setBackendStatus('ONLINE');

      // ── Step 4: Handle response ──
      const rawWord = (data.success && data.word) ? data.word : 'uncertain';
      const conf = typeof data.confidence === 'number' ? data.confidence : 0;
      const nowTime = Date.now();

      console.debug(`[SANKETA] Prediction: ${rawWord} | conf: ${(conf * 100).toFixed(1)}% | type: ${data.type}`);

      // Always update current prediction so HUD shows real-time feedback
      setCurrentPrediction({
        word: rawWord,
        confidence: conf,
        type: data.type || 'static',
        model: data.model || 'isl_model.keras',
        modelName: 'SANKETA Landmark Model',
        message: data.message || 'Hand sign recognized',
        probabilities: data.probabilities || null,
        error: false,
      });

      // ── Step 5: Temporal stability gating ──
      if (rawWord !== 'uncertain' && conf >= 0.70) {
        const buf = rollingBufferRef.current;
        buf.push({ word: rawWord, conf, time: nowTime });
        if (buf.length > 8) buf.shift();

        const freq = {};
        let sumConf = 0;
        buf.forEach((item) => {
          freq[item.word] = (freq[item.word] || 0) + 1;
          sumConf += item.conf;
        });

        let topWord = '';
        let topCount = 0;
        for (const [w, count] of Object.entries(freq)) {
          if (count > topCount) { topCount = count; topWord = w; }
        }

        const ratio = topCount / buf.length;
        setStabilityRatio(Math.round(ratio * 100));

        // Require 4+ frames, ≥70% same word, avg conf ≥70%
        const isStable = buf.length >= 4 && ratio >= 0.70 && (sumConf / buf.length) >= 0.70;

        if (isStable) {
          setStabilityState('STABLE');
          setStablePrediction(topWord);

          // 2.5s cooldown before repeating same word
          if (
            lastAcceptedWordRef.current === topWord &&
            nowTime - lastAcceptedTimeRef.current < 2500
          ) {
            return;
          }

          lastAcceptedWordRef.current = topWord;
          lastAcceptedTimeRef.current = nowTime;
          rollingBufferRef.current = [];

          const display = translateWord(topWord, lang);
          const signType = data.type || 'static';

          setSentenceWords((prev) => {
            if (prev[prev.length - 1] === display) return prev;
            return [...prev, display];
          });

          addHistory({
            input: `Camera (${signType.toUpperCase()})`,
            output: display,
            rawWord: topWord,
            type: signType,
            confidence: conf,
            dir: 'SIGN→TEXT',
          });
          setHistoryList(getHistory());

          const settings = getSettings();
          if (settings.autoSpeak) {
            speak(display, lang);
          }
        } else {
          setStabilityState('TRACKING');
        }
      } else {
        // Low confidence or uncertain — decay the buffer
        if (rollingBufferRef.current.length > 0) rollingBufferRef.current.shift();
        setStabilityState('TRACKING');
        setStabilityRatio(0);
      }

    } catch (err) {
      console.error('[SANKETA] dispatchInference unexpected error:', err);
      setLastApiError(err.message || 'Unexpected error');
      setBackendStatus('OFFLINE');
      setCurrentPrediction({
        word: '__error__',
        confidence: 0,
        type: 'static',
        model: 'isl_model.keras',
        modelName: 'SANKETA Landmark Model',
        message: `Error: ${err.message}`,
        error: true,
      });
    } finally {
      isInferringRef.current = false;
    }
  };

  dispatchInferenceRef.current = dispatchInference;

  // ── Record Dynamic Sequence (30 frames at 30 FPS) ──────────────────────────
  const recordDynamicSequence = async () => {
    if (
      cameraStatus !== 'LIVE' ||
      recordingDynamic ||
      !videoRef.current ||
      handsDetectedCount === 0
    ) {
      return;
    }

    const video = videoRef.current;
    setRecordingDynamic(true);
    setRecordProgress(0);

    const offCanvas = document.createElement('canvas');
    offCanvas.width = 320;
    offCanvas.height = 240;
    const offCtx = offCanvas.getContext('2d');

    const frames = [];
    const landmarkSeq = [];
    const totalFrames = 30;
    const intervalMs = 33; // ~30 FPS -> 1.0 second capture

    try {
      for (let i = 0; i < totalFrames; i++) {
        // Collect current 63-landmark vector if valid hand is detected
        if (latestLandmarksRef.current && latestLandmarksRef.current.length === 63) {
          landmarkSeq.push(latestLandmarksRef.current);
        }

        // Draw unmirrored frame as backup
        offCtx.drawImage(video, 0, 0, 320, 240);
        const blob = await new Promise((res) => offCanvas.toBlob(res, 'image/jpeg', 0.82));
        if (blob) frames.push(blob);
        setRecordProgress(Math.round(((i + 1) / totalFrames) * 100));
        await new Promise((res) => setTimeout(res, intervalMs));
      }

      setRecordingDynamic(false);
      let data = null;

      // Prioritize direct 30-frame MediaPipe sequence
      if (landmarkSeq.length === 30) {
        try {
          data = await apiTranslateSequence(landmarkSeq);
        } catch (seqErr) {
          data = null;
        }
      }

      // Fallback to burst image pipeline if sequence call was unavailable
      if (!data && frames.length >= 10) {
        data = await apiTranslateBurst(frames);
      }

      if (data.success && data.word && data.word !== 'uncertain') {
        const rawWord = data.word;
        const conf = data.confidence || 0;
        const nowTime = Date.now();

        setCurrentPrediction({
          word: rawWord,
          confidence: conf,
          type: 'dynamic',
          message: data.message || 'Dynamic sequence recognized',
        });

        if (conf >= 0.65) {
          if (
            lastAcceptedWordRef.current === rawWord &&
            nowTime - lastAcceptedTimeRef.current < 3000
          ) {
            await apiResetBuffer().catch(() => {});
            return;
          }

          lastAcceptedWordRef.current = rawWord;
          lastAcceptedTimeRef.current = nowTime;

          const display = translateWord(rawWord, lang);
          setSentenceWords((prev) => {
            if (prev[prev.length - 1] === display) return prev;
            return [...prev, display];
          });

          addHistory({
            input: 'Camera (DYNAMIC)',
            output: display,
            rawWord: rawWord,
            type: 'dynamic',
            confidence: conf,
            dir: 'SIGN→TEXT',
          });
          setHistoryList(getHistory());

          const settings = getSettings();
          if (settings.autoSpeak) {
            speak(display, lang);
          }
        }

        await apiResetBuffer().catch(() => {});
      }
    } catch (err) {
      console.error('Dynamic sequence record error:', err);
    } finally {
      setRecordingDynamic(false);
    }
  };

  // ── Sentence Operations ─────────────────────────────────────────────────────
  const sentenceText = sentenceWords.join(' ');

  const handleClearSentence = () => {
    setSentenceWords([]);
  };

  const handleSpeakSentence = () => {
    if (sentenceText) {
      speak(sentenceText, lang);
    }
  };

  const handleClearHistory = () => {
    clearHistory();
    setHistoryList([]);
  };

  // activeWord: null = no prediction yet, '__error__' = backend error, 'uncertain' = low conf
  const activeWord = currentPrediction?.word || null;
  const activeConf = currentPrediction?.confidence || 0;
  const activeError = currentPrediction?.error || false;
  const activeRealWord = activeWord && activeWord !== 'uncertain' && activeWord !== '__error__' ? activeWord : null;
  const activeAsset = activeRealWord ? SIGN_ASSETS[activeRealWord] : null;

  return (
    <div>
      {/* Header & Status Ribbon */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <div className="page-title">
            <span>📷</span> Live ISL Camera Recognition
          </div>
          <div className="page-sub">
            Real-time MediaPipe Hand Detection + Local Neural Network Inference Pipeline
          </div>
        </div>

        {/* Global Pipeline Badges */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <span className={`badge ${backendStatus === 'ONLINE' ? 'badge-green' : 'badge-red'}`}>
            BACKEND {backendStatus}
          </span>
          <span className={`badge ${mediaPipeStatus === 'READY' ? 'badge-blue' : mediaPipeStatus === 'INITIALIZING' ? 'badge-orange' : 'badge-red'}`}>
            MEDIAPIPE {mediaPipeStatus}
          </span>
          <span
            className={`badge ${
              cameraStatus === 'LIVE'
                ? 'badge-green'
                : cameraStatus === 'STARTING'
                ? 'badge-orange'
                : cameraStatus === 'PERMISSION_REQUIRED'
                ? 'badge-orange'
                : cameraStatus === 'ERROR'
                ? 'badge-red'
                : 'badge-ghost'
            }`}
          >
            CAMERA {cameraStatus}
          </span>
        </div>
      </div>

      <div className="grid-2" style={{ alignItems: 'start', marginTop: 16 }}>
        {/* Left Column: Live Webcam Viewport & Controls */}
        <div>
          <div
            className="camera-wrapper"
            style={{
              position: 'relative',
              overflow: 'hidden',
              borderRadius: 16,
              background: '#0a0e1a',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.45)',
            }}
          >
            {/* Live Video Preview Element (Mirrored horizontally via CSS for natural user experience) */}
            <video
              ref={videoRef}
              className="camera-preview"
              muted
              playsInline
              autoPlay
              style={{
                transform: isMirrored ? 'scaleX(-1)' : 'none',
                width: '100%',
                display: cameraStatus === 'LIVE' ? 'block' : 'none',
                objectFit: 'contain',
              }}
            />

            {/* Real-time Hand Skeleton Overlay Canvas (Positioned exactly over webcam) */}
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                display: cameraStatus === 'LIVE' ? 'block' : 'none',
                zIndex: 2,
              }}
            />

            {/* LIVE Overlay Badges */}
            {cameraStatus === 'LIVE' && (
              <>
                <div
                  style={{
                    position: 'absolute',
                    top: 12,
                    left: 12,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 6,
                    zIndex: 4,
                  }}
                >
                  <div
                    className="badge"
                    style={{
                      background: 'rgba(15, 23, 42, 0.88)',
                      backdropFilter: 'blur(8px)',
                      color: '#22c55e',
                      border: '1px solid rgba(34, 197, 94, 0.3)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      fontWeight: 700,
                    }}
                  >
                    <div className="pulse-dot" style={{ background: '#22c55e' }} />
                    <span>CAMERA CONNECTED</span>
                  </div>

                  {/* Real-Time Hand Status Badge */}
                  <div
                    className="badge"
                    style={{
                      background: 'rgba(15, 23, 42, 0.88)',
                      backdropFilter: 'blur(8px)',
                      color: handsDetectedCount > 0 ? '#38bdf8' : '#94a3b8',
                      border: handsDetectedCount > 0 ? '1px solid rgba(56, 189, 248, 0.4)' : '1px solid rgba(255, 255, 255, 0.12)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      fontWeight: 700,
                    }}
                  >
                    <span>{handsDetectedCount > 0 ? '● HAND DETECTED' : '○ NO HAND DETECTED'}</span>
                    {handednessLabel && (
                      <span style={{ color: '#f8fafc', fontWeight: 600 }}>({handednessLabel})</span>
                    )}
                  </div>

                  {/* 21 LANDMARKS Active Badge */}
                  {handsDetectedCount > 0 && (
                    <div
                      className="badge"
                      style={{
                        background: 'rgba(15, 23, 42, 0.88)',
                        backdropFilter: 'blur(8px)',
                        color: '#a855f7',
                        border: '1px solid rgba(168, 85, 247, 0.4)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                        fontWeight: 700,
                      }}
                    >
                      <span>✨ 21 LANDMARKS TRACKED</span>
                    </div>
                  )}

                  {/* Active Model Indicator Badge */}
                  <div
                    className="badge"
                    style={{
                      background: 'rgba(15, 23, 42, 0.88)',
                      backdropFilter: 'blur(8px)',
                      color: '#c084fc',
                      border: '1px solid rgba(192, 132, 252, 0.4)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      fontSize: '0.72rem',
                      fontWeight: 700,
                    }}
                  >
                    <span>MODEL: LANDMARK (ISL)</span>
                  </div>
                </div>

                {/* Right Top HUD: Hands Count & Live Tracking FPS */}
                <div
                  style={{
                    position: 'absolute',
                    top: 12,
                    right: 12,
                    display: 'flex',
                    gap: 6,
                    zIndex: 4,
                  }}
                >
                  <span
                    style={{
                      background: 'rgba(15, 23, 42, 0.88)',
                      backdropFilter: 'blur(8px)',
                      color: '#38bdf8',
                      border: '1px solid rgba(56, 189, 248, 0.3)',
                      padding: '3px 10px',
                      borderRadius: 8,
                      fontSize: '0.74rem',
                      fontFamily: 'monospace',
                      fontWeight: 700,
                    }}
                  >
                    {trackingFps} FPS
                  </span>

                  <span
                    style={{
                      background: 'rgba(15, 23, 42, 0.88)',
                      backdropFilter: 'blur(8px)',
                      color: handsDetectedCount > 0 ? '#10b981' : '#94a3b8',
                      border: '1px solid rgba(255, 255, 255, 0.15)',
                      padding: '3px 10px',
                      borderRadius: 8,
                      fontSize: '0.74rem',
                      fontWeight: 700,
                    }}
                  >
                    {handsDetectedCount} {handsDetectedCount === 1 ? 'HAND' : 'HANDS'}
                  </span>
                </div>
              </>
            )}

            {/* Offline / Starting / Error Placeholder Display */}
            {cameraStatus !== 'LIVE' && (
              <div
                style={{
                  minHeight: 400,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'rgba(10, 14, 26, 0.95)',
                  color: 'var(--text2)',
                  padding: 28,
                  textAlign: 'center',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                }}
              >
                {cameraStatus === 'STARTING' ? (
                  <>
                    <div style={{ fontSize: '3rem', marginBottom: 12 }} className="spin">
                      ⏳
                    </div>
                    <div style={{ fontWeight: 700, color: '#fff', fontSize: '1.25rem', marginBottom: 6 }}>
                      Connecting to Webcam…
                    </div>
                    <div style={{ fontSize: '0.9rem', maxWidth: 340, color: 'var(--text2)', lineHeight: 1.5 }}>
                      Requesting camera device and preparing local MediaPipe tracking pipeline.
                    </div>
                  </>
                ) : cameraStatus === 'PERMISSION_REQUIRED' ? (
                  <>
                    <div style={{ fontSize: '3.2rem', marginBottom: 12 }}>🔒</div>
                    <div style={{ fontWeight: 800, color: '#f59e0b', fontSize: '1.3rem', marginBottom: 8 }}>
                      Camera Permission Required
                    </div>
                    <div style={{ fontSize: '0.9rem', color: 'var(--text1)', maxWidth: 400, marginBottom: 20, lineHeight: 1.5 }}>
                      {cameraError}
                    </div>
                    <button className="btn btn-primary btn-lg" onClick={startCamera}>
                      🔄 Allow Camera & Retry
                    </button>
                  </>
                ) : cameraStatus === 'ERROR' ? (
                  <>
                    <div style={{ fontSize: '3.2rem', marginBottom: 12 }}>⚠️</div>
                    <div style={{ fontWeight: 800, color: '#ef4444', fontSize: '1.3rem', marginBottom: 8 }}>
                      Camera Access Failed
                    </div>
                    <div style={{ fontSize: '0.9rem', color: 'var(--text1)', maxWidth: 380, marginBottom: 20, lineHeight: 1.5 }}>
                      {cameraError}
                    </div>
                    <button className="btn btn-primary" onClick={startCamera}>
                      🔄 Retry Camera
                    </button>
                  </>
                ) : (
                  <>
                    <div style={{ fontSize: '3.6rem', marginBottom: 14 }}>📹</div>
                    <div style={{ fontWeight: 800, color: '#fff', fontSize: '1.35rem', marginBottom: 6 }}>
                      Webcam Offline
                    </div>
                    <div style={{ fontSize: '0.92rem', maxWidth: 380, marginBottom: 22, lineHeight: 1.5 }}>
                      Click <strong>Start Camera</strong> to activate your webcam with real-time 21-landmark MediaPipe hand tracking and live ISL recognition.
                    </div>
                    <button className="btn btn-primary btn-lg" onClick={startCamera}>
                      ▶ Start Camera
                    </button>
                  </>
                )}
              </div>
            )}
          </div>

          {/* Primary Action Buttons */}
          <div style={{ display: 'flex', gap: 10, marginTop: 14, flexWrap: 'wrap' }}>
            {cameraStatus === 'LIVE' ? (
              <>
                <button
                  type="button"
                  className={`btn btn-lg ${liveRecognitionActive ? 'btn-danger' : 'btn-success'}`}
                  onClick={() => setLiveRecognitionActive((p) => !p)}
                  disabled={recordingDynamic}
                >
                  {liveRecognitionActive ? '⏸ Pause Tracking' : '▶ Resume Tracking'}
                </button>

                <button
                  type="button"
                  className="btn btn-lg"
                  style={{
                    background: 'linear-gradient(135deg, #7928ca, #ff0080)',
                    color: '#fff',
                    border: 'none',
                  }}
                  onClick={recordDynamicSequence}
                  disabled={recordingDynamic || handsDetectedCount === 0}
                  title="Record 1-second 30-frame sequence for dynamic gesture"
                >
                  {recordingDynamic ? `📹 Recording (${recordProgress}%)…` : '⚡ Record Dynamic (1s)'}
                </button>

                <button
                  type="button"
                  className="btn btn-secondary btn-lg"
                  onClick={() => setIsMirrored((p) => !p)}
                  title="Toggle horizontal mirror"
                >
                  🪞 Mirror: {isMirrored ? 'ON' : 'OFF'}
                </button>

                <button
                  type="button"
                  className="btn btn-secondary btn-lg"
                  onClick={() => setDebugMode((p) => !p)}
                  title="Toggle Developer Debug Panel"
                >
                  🛠 Debug: {debugMode ? 'ON' : 'OFF'}
                </button>

                <button
                  type="button"
                  className="btn btn-danger btn-lg"
                  onClick={stopCamera}
                  title="Stop camera and turn off webcam"
                >
                  ⏹ Stop Camera
                </button>
              </>
            ) : (
              <button
                type="button"
                className="btn btn-primary btn-lg"
                onClick={startCamera}
                disabled={cameraStatus === 'STARTING'}
              >
                ▶ Start Camera
              </button>
            )}
          </div>

          {/* Developer / Debug Diagnostics Panel */}
          {debugMode && (
            <div
              className="card"
              style={{
                marginTop: 14,
                background: 'rgba(15, 23, 42, 0.95)',
                border: '1px solid rgba(56, 189, 248, 0.35)',
                padding: '16px 20px',
                borderRadius: 12,
                fontSize: '0.88rem',
              }}
            >
              <div style={{ fontWeight: 800, color: '#38bdf8', marginBottom: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ letterSpacing: '0.5px' }}>🛠 DEVELOPER DEBUG PANEL (TESTING)</span>
                <span style={{ color: cameraStatus === 'LIVE' ? '#22c55e' : '#94a3b8', fontSize: '0.78rem' }}>
                  ● {cameraStatus === 'LIVE' ? 'LIVE TESTING' : 'CAMERA OFFLINE'}
                </span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px 20px', color: 'var(--text2)' }}>
                <div><strong>Hand detected:</strong> <span style={{ color: handsDetectedCount > 0 ? '#22c55e' : '#ef4444', fontWeight: 700 }}>{handsDetectedCount > 0 ? 'YES' : 'NO'}</span></div>
                <div><strong>Landmarks:</strong> <span style={{ color: handsDetectedCount > 0 ? '#38bdf8' : '#94a3b8', fontWeight: 700 }}>{handsDetectedCount > 0 ? '21' : '0'}</span></div>
                <div><strong>Features:</strong> <span style={{ color: handsDetectedCount > 0 ? '#38bdf8' : '#94a3b8', fontWeight: 700 }}>{handsDetectedCount > 0 ? '63' : '0'}</span></div>
                <div><strong>Raw prediction:</strong> <span style={{ color: activeWord ? '#facc15' : '#94a3b8', fontWeight: 800, fontSize: '1.05rem' }}>{activeWord || 'None'}</span></div>
                <div><strong>Raw confidence:</strong> <span style={{ color: confColor(activeConf), fontWeight: 700 }}>{Math.round(activeConf * 100)}%</span></div>
                <div><strong>Stable prediction:</strong> <span style={{ color: stablePrediction ? '#22c55e' : '#94a3b8', fontWeight: 800, fontSize: '1.05rem' }}>{stablePrediction || 'None'}</span></div>
                <div><strong>Model:</strong> <span style={{ color: '#e0f2fe', fontFamily: 'monospace' }}>isl_model.keras</span></div>
                <div><strong>Model classes:</strong> <span style={{ color: '#e0f2fe', fontWeight: 700 }}>{modelInfo?.supported_classes || 11}</span></div>
              </div>
            </div>
          )}

          {/* Dynamic Recording Progress */}
          {recordingDynamic && (
            <div style={{ marginTop: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: 'var(--text2)', marginBottom: 4 }}>
                <span>Recording 30-Frame Gesture Sequence…</span>
                <strong>{recordProgress}%</strong>
              </div>
              <div className="conf-bar" style={{ height: 8 }}>
                <div
                  className="conf-fill"
                  style={{ width: `${recordProgress}%`, background: 'linear-gradient(90deg, #7928ca, #ff0080)' }}
                />
              </div>
            </div>
          )}

          {/* Sentence Builder Accumulator */}
          <div className="sentence-box" style={{ marginTop: 20 }}>
            <div className="sentence-header">
              <span>Translated Sentence (Accumulator)</span>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  className="btn btn-sm btn-ghost"
                  onClick={handleClearSentence}
                  disabled={sentenceWords.length === 0}
                >
                  Clear
                </button>
                <button
                  className="btn btn-sm btn-success"
                  onClick={handleSpeakSentence}
                  disabled={sentenceWords.length === 0}
                >
                  🔊 Speak Sentence
                </button>
              </div>
            </div>

            <div className="sentence-text">
              {sentenceText || (
                <span style={{ color: 'var(--text3)', fontSize: '0.92rem' }}>
                  Hold a sign steadily in front of the camera to build a sentence here…
                </span>
              )}
            </div>

            {sentenceWords.length > 0 && (
              <div className="sentence-chips">
                {sentenceWords.map((w, idx) => (
                  <span key={idx} className="word-chip">
                    {w}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Real-Time Prediction & Stabilization */}
        <div>
          {/* Active Prediction Card */}
          <div className={`card ${activeWord ? 'card-glow' : ''}`}>
            <div className="section-title">
              <span>Live Recognition Status</span>
              {activeWord && (
                <span
                  className={`badge ${
                    stabilityState === 'STABLE'
                      ? 'badge-green'
                      : stabilityState === 'TRACKING'
                      ? 'badge-blue'
                      : 'badge-ghost'
                  }`}
                >
                  {stabilityState === 'STABLE'
                    ? '● STABLE'
                    : stabilityState === 'TRACKING'
                    ? `TRACKING (${stabilityRatio}%)`
                    : 'WAITING'}
                </span>
              )}
            </div>

            {cameraStatus !== 'LIVE' ? (
              <div style={{ color: 'var(--text3)', fontSize: '0.92rem', padding: '24px 0', textAlign: 'center' }}>
                Camera is inactive. Click "Start Camera" to begin detection.
              </div>
            ) : handsDetectedCount === 0 ? (
              <div style={{ padding: '24px 0', textAlign: 'center' }}>
                <div style={{ fontSize: '2.5rem', marginBottom: 10 }}>✋</div>
                <div style={{ fontWeight: 700, color: 'var(--text2)', marginBottom: 4 }}>
                  No Hand Detected
                </div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text3)', maxWidth: 280, margin: '0 auto' }}>
                  Position one or two hands clearly within the camera frame for real-time skeleton tracking and ISL recognition.
                </div>
              </div>
            ) : activeWord ? (
              <div id="recognition-result-panel">
                {/* HUD Metadata Tags: HAND DETECTED, LEFT/RIGHT HAND, 21 LANDMARKS, MODEL NAME */}
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
                  <span
                    className="badge"
                    style={{
                      background: 'rgba(34, 197, 94, 0.15)',
                      color: '#22c55e',
                      border: '1px solid rgba(34, 197, 94, 0.35)',
                      fontSize: '0.72rem',
                      fontWeight: 700,
                    }}
                  >
                    ● HAND DETECTED {handednessLabel ? `(${handednessLabel})` : ''}
                  </span>

                  <span
                    className="badge"
                    style={{
                      background: 'rgba(168, 85, 247, 0.15)',
                      color: '#c084fc',
                      border: '1px solid rgba(168, 85, 247, 0.35)',
                      fontSize: '0.72rem',
                      fontWeight: 700,
                    }}
                  >
                    ✨ 21 LANDMARKS
                  </span>

                  <span
                    className="badge"
                    style={{
                      background: 'rgba(234, 179, 8, 0.15)',
                      color: '#facc15',
                      border: '1px solid rgba(234, 179, 8, 0.35)',
                      fontSize: '0.72rem',
                      fontWeight: 700,
                    }}
                  >
                    MODEL: {currentPrediction?.modelName || 'SANKETA Landmark Model'}
                  </span>
                </div>

                {/* Backend error state */}
                {activeError ? (
                  <div style={{ padding: '16px 14px', background: 'rgba(239, 68, 68, 0.10)', borderRadius: 10, border: '1px solid rgba(239, 68, 68, 0.35)', marginBottom: 12 }}>
                    <div style={{ fontSize: '0.85rem', color: '#f87171', fontWeight: 800, marginBottom: 6 }}>
                      🔴 BACKEND OFFLINE / ERROR
                    </div>
                    <div style={{ fontSize: '0.80rem', color: '#fca5a5', fontFamily: 'monospace', wordBreak: 'break-all' }}>
                      {currentPrediction?.message || `Cannot reach backend at ${API_BASE}`}
                    </div>
                    <div style={{ fontSize: '0.76rem', color: 'var(--text3)', marginTop: 8 }}>
                      Start the backend: <code style={{ color: '#38bdf8' }}>uvicorn app.main:app --reload</code> inside <code style={{ color: '#38bdf8' }}>signbridge/backend/</code>
                    </div>
                  </div>
                ) : activeRealWord ? (
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                    <div>
                      <div style={{ fontSize: '0.78rem', color: '#22c55e', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.8px', display: 'flex', alignItems: 'center', gap: 5 }}>
                        <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#22c55e' }}></span>
                        SIGN DETECTED
                      </div>
                      <div style={{ fontSize: '2.4rem', fontWeight: 800, color: '#fff', lineHeight: 1.1, marginTop: 4 }}>
                        {formatISLWord(activeRealWord)}
                      </div>
                      {lang !== 'en-US' && (
                        <div style={{ fontSize: '1.05rem', color: 'var(--accent)', fontWeight: 700, marginTop: 4 }}>
                          {translateWord(activeRealWord, lang)}
                        </div>
                      )}
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '2.0rem', fontWeight: 800, color: confColor(activeConf) }}>
                        {Math.round(activeConf * 100)}%
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text3)', fontWeight: 600 }}>
                        Confidence
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ padding: '16px 14px', background: 'rgba(245, 158, 11, 0.08)', borderRadius: 10, border: '1px solid rgba(245, 158, 11, 0.25)', marginBottom: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <div style={{ fontSize: '0.82rem', color: '#f59e0b', fontWeight: 800, letterSpacing: '0.5px' }}>
                          ⚠️ UNCERTAIN SIGN
                        </div>
                        <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f8fafc', lineHeight: 1.1, marginTop: 4 }}>
                          Uncertain sign
                        </div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '1.6rem', fontWeight: 800, color: confColor(activeConf) }}>
                          {Math.round(activeConf * 100)}%
                        </div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text3)' }}>
                          Confidence &lt; 70%
                        </div>
                      </div>
                    </div>
                    <div style={{ fontSize: '0.80rem', color: 'var(--text2)', marginTop: 8 }}>
                      Hold hand clearly within frame. Required confidence threshold is &ge; 70%.
                    </div>
                  </div>
                )}

                {/* Confidence Meter Bar */}
                <div className="conf-bar" style={{ height: 8, marginBottom: 12 }}>
                  <div
                    className="conf-fill"
                    style={{
                      width: `${Math.round(activeConf * 100)}%`,
                      background: activeError ? '#ef4444' : confColor(activeConf),
                    }}
                  />
                </div>

                {/* Developer Debug Telemetry Panel */}
                <div
                  style={{
                    marginBottom: 12,
                    padding: '10px 14px',
                    background: 'rgba(15, 23, 42, 0.95)',
                    borderRadius: 8,
                    border: '1px solid rgba(56, 189, 248, 0.25)',
                    fontFamily: 'monospace',
                    fontSize: '0.76rem',
                  }}
                >
                  <div style={{ color: '#38bdf8', fontWeight: 800, marginBottom: 6, fontSize: '0.70rem', letterSpacing: '0.6px' }}>
                    DEVELOPER DEBUG TELEMETRY
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 10px' }}>
                    <div><span style={{ color: '#94a3b8' }}>HAND DETECTED: </span><strong style={{ color: handsDetectedCount > 0 ? '#22c55e' : '#ef4444' }}>{handsDetectedCount > 0 ? 'YES' : 'NO'}</strong></div>
                    <div><span style={{ color: '#94a3b8' }}>LANDMARKS: </span><strong style={{ color: '#38bdf8' }}>{handsDetectedCount > 0 ? '21' : '0'}</strong></div>
                    <div><span style={{ color: '#94a3b8' }}>FEATURES: </span><strong style={{ color: '#c084fc' }}>{handsDetectedCount > 0 ? '63' : '0'}</strong></div>
                    <div><span style={{ color: '#94a3b8' }}>MODEL: </span><strong style={{ color: '#facc15' }}>isl_model.keras</strong></div>
                    <div><span style={{ color: '#94a3b8' }}>PREDICTION: </span><strong style={{ color: activeRealWord ? '#22c55e' : activeError ? '#ef4444' : '#f59e0b' }}>{activeError ? 'ERROR' : activeRealWord || 'Uncertain'}</strong></div>
                    <div><span style={{ color: '#94a3b8' }}>CONFIDENCE: </span><strong style={{ color: confColor(activeConf) }}>{Math.round(activeConf * 100)}%</strong></div>
                    {lastApiError ? <div style={{ gridColumn: '1/-1' }}><span style={{ color: '#94a3b8' }}>API ERROR: </span><strong style={{ color: '#f87171', wordBreak: 'break-all' }}>{lastApiError}</strong></div> : null}
                  </div>
                </div>

                {/* Gesture Description */}
                {activeAsset?.description && (
                  <div
                    style={{
                      fontSize: '0.82rem',
                      color: 'var(--text2)',
                      background: 'rgba(255, 255, 255, 0.04)',
                      padding: '10px 12px',
                      borderRadius: 8,
                      marginBottom: 12,
                      lineHeight: 1.45,
                    }}
                  >
                    <strong>Execution:</strong> {activeAsset.description}
                  </div>
                )}

                {/* Pronunciation Speaker Button - only for real recognized word */}
                {activeRealWord && (
                  <button
                    type="button"
                    className="btn btn-sm btn-success"
                    style={{ width: '100%', justifyContent: 'center' }}
                    onClick={() => speak(translateWord(activeRealWord, lang), lang)}
                  >
                    🔊 Speak "{translateWord(activeRealWord, lang)}"
                  </button>
                )}
              </div>
            ) : (
              <div style={{ padding: '24px 0', textAlign: 'center' }}>
                <div style={{ fontSize: '2.2rem', marginBottom: 8 }} className="pulse">
                  ⚡
                </div>
                <div style={{ fontWeight: 700, color: 'var(--primary)' }}>
                  Analyzing Hand Pose…
                </div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text3)', marginTop: 4 }}>
                  Hold your hand steady to classify the ISL sign.
                </div>
              </div>
            )}
          </div>

          {/* Translation History Log */}
          <div className="card" style={{ marginTop: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <div className="section-title" style={{ marginBottom: 0 }}>
                Recent History
              </div>
              {historyList.length > 0 && (
                <button className="btn btn-xs btn-ghost" onClick={handleClearHistory}>
                  Clear History
                </button>
              )}
            </div>

            {historyList.length === 0 ? (
              <div style={{ color: 'var(--text3)', fontSize: '0.85rem', textAlign: 'center', padding: '16px 0' }}>
                No history yet. Recognized signs will appear here.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 220, overflowY: 'auto' }}>
                {historyList.slice(0, 8).map((item, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      background: 'rgba(255, 255, 255, 0.03)',
                      padding: '8px 12px',
                      borderRadius: 8,
                      fontSize: '0.84rem',
                    }}
                  >
                    <div>
                      <span style={{ fontWeight: 700, color: '#f1f5f9' }}>{item.output}</span>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text3)', marginLeft: 8 }}>
                        {item.input}
                      </span>
                    </div>
                    {item.confidence && (
                      <span style={{ fontSize: '0.75rem', fontWeight: 600, color: confColor(item.confidence) }}>
                        {Math.round(item.confidence * 100)}%
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
