// SANKETA — Local Procedural ISL Hand Animation Kinematics
// web/src/components/signAnimations.js

export const SIGN_FRAMES_COUNT = 30;

export const SIGN_ANIMATION_KEYS = [
  'HELLO',
  'YES',
  'NO',
  'WATER',
  'THANK_YOU',
  'PLEASE',
  'SORRY',
  'HELP',
  'OKAY',
  'I_LOVE_YOU',
];

/**
 * Normalizes sign keys (e.g. "hello" -> "HELLO", "thank you" -> "THANK_YOU")
 */
export function normalizeSignKey(raw) {
  if (!raw || typeof raw !== 'string') return 'HELLO';
  const clean = raw.trim().toUpperCase().replace(/[\s\-_]+/g, '_');
  if (clean === 'THANKYOU') return 'THANK_YOU';
  if (clean === 'ILOVEYOU' || clean === 'ILY') return 'I_LOVE_YOU';
  if (SIGN_ANIMATION_KEYS.includes(clean)) return clean;
  // Fallbacks for known aliases
  if (['HI', 'HEY', 'GREETINGS'].includes(clean)) return 'HELLO';
  if (['YEAH', 'YEP'].includes(clean)) return 'YES';
  if (['OK', 'OKAY'].includes(clean)) return 'OKAY';
  if (['I_LOVE_YOU', 'ILOVEYOU', 'I LOVE YOU'].includes(clean)) return 'I_LOVE_YOU';
  if (['NAH', 'NOPE'].includes(clean)) return 'NO';
  if (['DRINK', 'THIRSTY'].includes(clean)) return 'WATER';
  if (['THANKS', 'THX'].includes(clean)) return 'THANK_YOU';
  if (['PLZ', 'PLS'].includes(clean)) return 'PLEASE';
  if (['APOLOGIZE', 'MY_BAD'].includes(clean)) return 'SORRY';
  if (['ASSIST', 'ASSISTANCE', 'EMERGENCY'].includes(clean)) return 'HELP';
  return clean;
}

/**
 * Metadata for all 8 supported ISL signs
 */
export const SIGN_METADATA = {
  HELLO: {
    key: 'HELLO',
    label: 'Hello',
    type: 'static',
    description: 'Open palm held at head height, waving gracefully side-to-side with natural wrist oscillation.',
    tip: 'Keep palm open, fingers slightly separated, and wave smoothly from the wrist.',
  },
  YES: {
    key: 'YES',
    label: 'Yes',
    type: 'static',
    description: 'Closed fist held in front of chest, nodding forward and down from the wrist like a head nod.',
    tip: 'Form an "S" fist and nod forward and down twice smoothly.',
  },
  NO: {
    key: 'NO',
    label: 'No',
    type: 'static',
    description: 'Index and middle fingers extended together, snapping down firmly onto the thumb.',
    tip: 'Extend index and middle fingers, then snap them down onto the thumb twice in a pinching motion.',
  },
  WATER: {
    key: 'WATER',
    label: 'Water',
    type: 'static',
    description: '"W" handshape (three middle fingers upright) tapping lightly against the chin twice.',
    tip: 'Form a "W" with index, middle, and ring fingers upright; tap near your chin twice.',
  },
  THANK_YOU: {
    key: 'THANK_YOU',
    label: 'Thank You',
    type: 'dynamic',
    description: 'Flat hand starting with fingertips near chin/lips, extending forward and downward toward the listener.',
    tip: 'Start with fingers near your chin, then extend your flat open palm forward smoothly.',
  },
  PLEASE: {
    key: 'PLEASE',
    label: 'Please',
    type: 'dynamic',
    description: 'Flat open palm rubbed in a smooth clockwise circular motion over the center of the chest.',
    tip: 'Place flat hand on your chest and trace a smooth circle clockwise.',
  },
  SORRY: {
    key: 'SORRY',
    label: 'Sorry',
    type: 'dynamic',
    description: 'Closed "A" fist rubbed in a circular motion on the chest over the heart with an apologetic expression.',
    tip: 'Make a fist with thumb upright alongside fingers and circle gently over your chest.',
  },
  HELP: {
    key: 'HELP',
    label: 'Help',
    type: 'dynamic',
    description: 'Dominant fist in thumbs-up position resting upon the flat open palm of the support hand, lifted upward together.',
    tip: 'Rest your dominant thumbs-up fist upon your flat open palm and lift both hands upward in unison.',
  },
};

export function getSignMeta(signKey) {
  const norm = normalizeSignKey(signKey);
  return SIGN_METADATA[norm] || SIGN_METADATA.HELLO;
}

// ─────────────────────────────────────────────────────────────────────────────
// Interpolation Helpers
// ─────────────────────────────────────────────────────────────────────────────

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function easeInOutQuad(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

/**
 * Linearly interpolates between keyframe objects
 */
function interpolateKeyframes(keyframes, frameIndex) {
  if (!keyframes || keyframes.length === 0) return null;
  const f = Math.max(0, Math.min(SIGN_FRAMES_COUNT - 1, frameIndex));

  // Find surrounding keyframes
  if (f <= keyframes[0].frame) return keyframes[0].pose;
  if (f >= keyframes[keyframes.length - 1].frame) {
    return keyframes[keyframes.length - 1].pose;
  }

  for (let i = 0; i < keyframes.length - 1; i++) {
    const k1 = keyframes[i];
    const k2 = keyframes[i + 1];
    if (f >= k1.frame && f <= k2.frame) {
      const span = k2.frame - k1.frame;
      const rawT = span === 0 ? 0 : (f - k1.frame) / span;
      const t = easeInOutQuad(rawT);
      return interpolatePose(k1.pose, k2.pose, t);
    }
  }

  return keyframes[0].pose;
}

function interpolatePose(p1, p2, t) {
  return {
    wristX: lerp(p1.wristX || 0, p2.wristX || 0, t),
    wristY: lerp(p1.wristY || 0, p2.wristY || 0, t),
    wristAngle: lerp(p1.wristAngle || 0, p2.wristAngle || 0, t),
    scale: lerp(p1.scale || 1, p2.scale || 1, t),
    actionText: t < 0.5 ? p1.actionText : p2.actionText,
    thumb: {
      curl: lerp(p1.thumb?.curl || 0, p2.thumb?.curl || 0, t),
      angle: lerp(p1.thumb?.angle || 0, p2.thumb?.angle || 0, t),
      spread: lerp(p1.thumb?.spread || 0, p2.thumb?.spread || 0, t),
    },
    index: {
      curl: lerp(p1.index?.curl || 0, p2.index?.curl || 0, t),
      angle: lerp(p1.index?.angle || 0, p2.index?.angle || 0, t),
      spread: lerp(p1.index?.spread || 0, p2.index?.spread || 0, t),
    },
    middle: {
      curl: lerp(p1.middle?.curl || 0, p2.middle?.curl || 0, t),
      angle: lerp(p1.middle?.angle || 0, p2.middle?.angle || 0, t),
      spread: lerp(p1.middle?.spread || 0, p2.middle?.spread || 0, t),
    },
    ring: {
      curl: lerp(p1.ring?.curl || 0, p2.ring?.curl || 0, t),
      angle: lerp(p1.ring?.angle || 0, p2.ring?.angle || 0, t),
      spread: lerp(p1.ring?.spread || 0, p2.ring?.spread || 0, t),
    },
    pinky: {
      curl: lerp(p1.pinky?.curl || 0, p2.pinky?.curl || 0, t),
      angle: lerp(p1.pinky?.angle || 0, p2.pinky?.angle || 0, t),
      spread: lerp(p1.pinky?.spread || 0, p2.pinky?.spread || 0, t),
    },
    supportHand: p1.supportHand ? {
      active: p1.supportHand.active,
      x: lerp(p1.supportHand.x || 0, p2.supportHand?.x || 0, t),
      y: lerp(p1.supportHand.y || 0, p2.supportHand?.y || 0, t),
      angle: lerp(p1.supportHand.angle || 0, p2.supportHand?.angle || 0, t),
    } : null,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 30-Frame Kinematic Definitions for All 8 ISL Signs
// ─────────────────────────────────────────────────────────────────────────────

// 1. HELLO: Open palm waving side-to-side
const HELLO_KEYFRAMES = [
  {
    frame: 0,
    pose: {
      wristX: 0, wristY: 0, wristAngle: 0, scale: 1.0, actionText: 'Palm open facing forward',
      thumb: { curl: 0.1, angle: -32, spread: 22 },
      index: { curl: 0.05, angle: -4, spread: 0 },
      middle: { curl: 0.02, angle: 0, spread: 0 },
      ring: { curl: 0.05, angle: 4, spread: 0 },
      pinky: { curl: 0.08, angle: 10, spread: 12 },
    },
  },
  {
    frame: 7,
    pose: {
      wristX: -18, wristY: 3, wristAngle: -19, scale: 1.02, actionText: 'Wave left from wrist',
      thumb: { curl: 0.12, angle: -36, spread: 26 },
      index: { curl: 0.04, angle: -5, spread: 0 },
      middle: { curl: 0.02, angle: -1, spread: 0 },
      ring: { curl: 0.05, angle: 3, spread: 0 },
      pinky: { curl: 0.08, angle: 9, spread: 14 },
    },
  },
  {
    frame: 15,
    pose: {
      wristX: 18, wristY: 3, wristAngle: 19, scale: 1.02, actionText: 'Wave right across center',
      thumb: { curl: 0.08, angle: -28, spread: 18 },
      index: { curl: 0.05, angle: -3, spread: 0 },
      middle: { curl: 0.02, angle: 1, spread: 0 },
      ring: { curl: 0.05, angle: 5, spread: 0 },
      pinky: { curl: 0.08, angle: 11, spread: 10 },
    },
  },
  {
    frame: 22,
    pose: {
      wristX: -12, wristY: 2, wristAngle: -14, scale: 1.01, actionText: 'Wave left gentle oscillation',
      thumb: { curl: 0.1, angle: -34, spread: 24 },
      index: { curl: 0.05, angle: -4, spread: 0 },
      middle: { curl: 0.02, angle: 0, spread: 0 },
      ring: { curl: 0.05, angle: 4, spread: 0 },
      pinky: { curl: 0.08, angle: 10, spread: 12 },
    },
  },
  {
    frame: 29,
    pose: {
      wristX: 0, wristY: 0, wristAngle: 0, scale: 1.0, actionText: 'Return to center',
      thumb: { curl: 0.1, angle: -32, spread: 22 },
      index: { curl: 0.05, angle: -4, spread: 0 },
      middle: { curl: 0.02, angle: 0, spread: 0 },
      ring: { curl: 0.05, angle: 4, spread: 0 },
      pinky: { curl: 0.08, angle: 10, spread: 12 },
    },
  },
];

// 2. YES: Closed fist nodding forward and down twice
const YES_KEYFRAMES = [
  {
    frame: 0,
    pose: {
      wristX: 0, wristY: -4, wristAngle: 0, scale: 1.0, actionText: 'Hold "S" fist upright',
      thumb: { curl: 0.95, angle: 15, spread: 0 },
      index: { curl: 0.98, angle: 0, spread: 0 },
      middle: { curl: 0.98, angle: 0, spread: 0 },
      ring: { curl: 0.98, angle: 0, spread: 0 },
      pinky: { curl: 0.98, angle: 0, spread: 0 },
    },
  },
  {
    frame: 6,
    pose: {
      wristX: 0, wristY: 18, wristAngle: 24, scale: 1.04, actionText: 'Nod fist down (1st nod)',
      thumb: { curl: 0.96, angle: 18, spread: 0 },
      index: { curl: 1.0, angle: 0, spread: 0 },
      middle: { curl: 1.0, angle: 0, spread: 0 },
      ring: { curl: 1.0, angle: 0, spread: 0 },
      pinky: { curl: 1.0, angle: 0, spread: 0 },
    },
  },
  {
    frame: 12,
    pose: {
      wristX: 0, wristY: -2, wristAngle: 0, scale: 1.0, actionText: 'Lift back to upright',
      thumb: { curl: 0.95, angle: 15, spread: 0 },
      index: { curl: 0.98, angle: 0, spread: 0 },
      middle: { curl: 0.98, angle: 0, spread: 0 },
      ring: { curl: 0.98, angle: 0, spread: 0 },
      pinky: { curl: 0.98, angle: 0, spread: 0 },
    },
  },
  {
    frame: 18,
    pose: {
      wristX: 0, wristY: 16, wristAngle: 22, scale: 1.03, actionText: 'Nod fist down (2nd nod)',
      thumb: { curl: 0.96, angle: 18, spread: 0 },
      index: { curl: 1.0, angle: 0, spread: 0 },
      middle: { curl: 1.0, angle: 0, spread: 0 },
      ring: { curl: 1.0, angle: 0, spread: 0 },
      pinky: { curl: 1.0, angle: 0, spread: 0 },
    },
  },
  {
    frame: 24,
    pose: {
      wristX: 0, wristY: -2, wristAngle: -2, scale: 1.0, actionText: 'Slight rebound',
      thumb: { curl: 0.95, angle: 15, spread: 0 },
      index: { curl: 0.98, angle: 0, spread: 0 },
      middle: { curl: 0.98, angle: 0, spread: 0 },
      ring: { curl: 0.98, angle: 0, spread: 0 },
      pinky: { curl: 0.98, angle: 0, spread: 0 },
    },
  },
  {
    frame: 29,
    pose: {
      wristX: 0, wristY: -4, wristAngle: 0, scale: 1.0, actionText: 'Hold ready position',
      thumb: { curl: 0.95, angle: 15, spread: 0 },
      index: { curl: 0.98, angle: 0, spread: 0 },
      middle: { curl: 0.98, angle: 0, spread: 0 },
      ring: { curl: 0.98, angle: 0, spread: 0 },
      pinky: { curl: 0.98, angle: 0, spread: 0 },
    },
  },
];

// 3. NO: Index and middle fingers snapping down onto the thumb twice
const NO_KEYFRAMES = [
  {
    frame: 0,
    pose: {
      wristX: 0, wristY: 0, wristAngle: -8, scale: 1.0, actionText: 'Index & middle fingers open',
      thumb: { curl: 0.15, angle: -42, spread: 35 },
      index: { curl: 0.05, angle: -8, spread: 0 },
      middle: { curl: 0.05, angle: -4, spread: 0 },
      ring: { curl: 0.98, angle: 6, spread: 0 },
      pinky: { curl: 0.98, angle: 12, spread: 0 },
    },
  },
  {
    frame: 6,
    pose: {
      wristX: -4, wristY: 8, wristAngle: -4, scale: 1.02, actionText: 'Snap fingers down to thumb (1st snap)',
      thumb: { curl: 0.45, angle: -20, spread: 15 },
      index: { curl: 0.72, angle: -12, spread: 0 },
      middle: { curl: 0.72, angle: -10, spread: 0 },
      ring: { curl: 0.98, angle: 6, spread: 0 },
      pinky: { curl: 0.98, angle: 12, spread: 0 },
    },
  },
  {
    frame: 11,
    pose: {
      wristX: 0, wristY: -2, wristAngle: -7, scale: 1.0, actionText: 'Open fingers back up',
      thumb: { curl: 0.18, angle: -38, spread: 30 },
      index: { curl: 0.12, angle: -8, spread: 0 },
      middle: { curl: 0.12, angle: -4, spread: 0 },
      ring: { curl: 0.98, angle: 6, spread: 0 },
      pinky: { curl: 0.98, angle: 12, spread: 0 },
    },
  },
  {
    frame: 17,
    pose: {
      wristX: -4, wristY: 8, wristAngle: -4, scale: 1.02, actionText: 'Snap fingers down to thumb (2nd snap)',
      thumb: { curl: 0.45, angle: -20, spread: 15 },
      index: { curl: 0.72, angle: -12, spread: 0 },
      middle: { curl: 0.72, angle: -10, spread: 0 },
      ring: { curl: 0.98, angle: 6, spread: 0 },
      pinky: { curl: 0.98, angle: 12, spread: 0 },
    },
  },
  {
    frame: 24,
    pose: {
      wristX: 0, wristY: 0, wristAngle: -7, scale: 1.0, actionText: 'Settle back open',
      thumb: { curl: 0.18, angle: -38, spread: 30 },
      index: { curl: 0.1, angle: -8, spread: 0 },
      middle: { curl: 0.1, angle: -4, spread: 0 },
      ring: { curl: 0.98, angle: 6, spread: 0 },
      pinky: { curl: 0.98, angle: 12, spread: 0 },
    },
  },
  {
    frame: 29,
    pose: {
      wristX: 0, wristY: 0, wristAngle: -8, scale: 1.0, actionText: 'Ready position',
      thumb: { curl: 0.15, angle: -42, spread: 35 },
      index: { curl: 0.05, angle: -8, spread: 0 },
      middle: { curl: 0.05, angle: -4, spread: 0 },
      ring: { curl: 0.98, angle: 6, spread: 0 },
      pinky: { curl: 0.98, angle: 12, spread: 0 },
    },
  },
];

// 4. WATER: "W" handshape tapping chin twice
const WATER_KEYFRAMES = [
  {
    frame: 0,
    pose: {
      wristX: 0, wristY: 4, wristAngle: 4, scale: 1.0, actionText: 'Form "W" shape (3 fingers extended)',
      thumb: { curl: 0.82, angle: 10, spread: 0 },
      index: { curl: 0.05, angle: -14, spread: 10 },
      middle: { curl: 0.02, angle: 0, spread: 0 },
      ring: { curl: 0.05, angle: 14, spread: 10 },
      pinky: { curl: 0.96, angle: 8, spread: 0 },
    },
  },
  {
    frame: 6,
    pose: {
      wristX: 14, wristY: -18, wristAngle: -7, scale: 1.04, actionText: 'Tap chin with "W" index side (1st tap)',
      thumb: { curl: 0.82, angle: 10, spread: 0 },
      index: { curl: 0.05, angle: -14, spread: 10 },
      middle: { curl: 0.02, angle: 0, spread: 0 },
      ring: { curl: 0.05, angle: 14, spread: 10 },
      pinky: { curl: 0.96, angle: 8, spread: 0 },
    },
  },
  {
    frame: 11,
    pose: {
      wristX: 4, wristY: -4, wristAngle: 0, scale: 1.01, actionText: 'Recoil slightly',
      thumb: { curl: 0.82, angle: 10, spread: 0 },
      index: { curl: 0.05, angle: -14, spread: 10 },
      middle: { curl: 0.02, angle: 0, spread: 0 },
      ring: { curl: 0.05, angle: 14, spread: 10 },
      pinky: { curl: 0.96, angle: 8, spread: 0 },
    },
  },
  {
    frame: 17,
    pose: {
      wristX: 14, wristY: -18, wristAngle: -7, scale: 1.04, actionText: 'Tap chin with "W" index side (2nd tap)',
      thumb: { curl: 0.82, angle: 10, spread: 0 },
      index: { curl: 0.05, angle: -14, spread: 10 },
      middle: { curl: 0.02, angle: 0, spread: 0 },
      ring: { curl: 0.05, angle: 14, spread: 10 },
      pinky: { curl: 0.96, angle: 8, spread: 0 },
    },
  },
  {
    frame: 24,
    pose: {
      wristX: 4, wristY: 0, wristAngle: 2, scale: 1.0, actionText: 'Settle downward',
      thumb: { curl: 0.82, angle: 10, spread: 0 },
      index: { curl: 0.05, angle: -14, spread: 10 },
      middle: { curl: 0.02, angle: 0, spread: 0 },
      ring: { curl: 0.05, angle: 14, spread: 10 },
      pinky: { curl: 0.96, angle: 8, spread: 0 },
    },
  },
  {
    frame: 29,
    pose: {
      wristX: 0, wristY: 4, wristAngle: 4, scale: 1.0, actionText: 'Ready position',
      thumb: { curl: 0.82, angle: 10, spread: 0 },
      index: { curl: 0.05, angle: -14, spread: 10 },
      middle: { curl: 0.02, angle: 0, spread: 0 },
      ring: { curl: 0.05, angle: 14, spread: 10 },
      pinky: { curl: 0.96, angle: 8, spread: 0 },
    },
  },
];

// 5. THANK_YOU: Flat hand moving forward and down from chin
const THANK_YOU_KEYFRAMES = [
  {
    frame: 0,
    pose: {
      wristX: -6, wristY: -22, wristAngle: 8, scale: 0.92, actionText: 'Touch fingertips near lips/chin',
      thumb: { curl: 0.15, angle: -24, spread: 8 },
      index: { curl: 0.08, angle: 2, spread: 0 },
      middle: { curl: 0.06, angle: 0, spread: 0 },
      ring: { curl: 0.08, angle: -2, spread: 0 },
      pinky: { curl: 0.1, angle: -4, spread: 0 },
    },
  },
  {
    frame: 7,
    pose: {
      wristX: -2, wristY: -10, wristAngle: 4, scale: 0.98, actionText: 'Begin moving forward from chin',
      thumb: { curl: 0.12, angle: -26, spread: 10 },
      index: { curl: 0.04, angle: 1, spread: 0 },
      middle: { curl: 0.02, angle: 0, spread: 0 },
      ring: { curl: 0.04, angle: -1, spread: 0 },
      pinky: { curl: 0.06, angle: -2, spread: 0 },
    },
  },
  {
    frame: 17,
    pose: {
      wristX: 14, wristY: 22, wristAngle: -10, scale: 1.12, actionText: 'Extend flat palm forward to listener',
      thumb: { curl: 0.08, angle: -32, spread: 16 },
      index: { curl: 0.02, angle: -2, spread: 0 },
      middle: { curl: 0.0, angle: 0, spread: 0 },
      ring: { curl: 0.02, angle: 2, spread: 0 },
      pinky: { curl: 0.05, angle: 5, spread: 2 },
    },
  },
  {
    frame: 23,
    pose: {
      wristX: 14, wristY: 22, wristAngle: -10, scale: 1.12, actionText: 'Hold presentation of gratitude',
      thumb: { curl: 0.08, angle: -32, spread: 16 },
      index: { curl: 0.02, angle: -2, spread: 0 },
      middle: { curl: 0.0, angle: 0, spread: 0 },
      ring: { curl: 0.02, angle: 2, spread: 0 },
      pinky: { curl: 0.05, angle: 5, spread: 2 },
    },
  },
  {
    frame: 29,
    pose: {
      wristX: -6, wristY: -22, wristAngle: 8, scale: 0.92, actionText: 'Return to starting chin position',
      thumb: { curl: 0.15, angle: -24, spread: 8 },
      index: { curl: 0.08, angle: 2, spread: 0 },
      middle: { curl: 0.06, angle: 0, spread: 0 },
      ring: { curl: 0.08, angle: -2, spread: 0 },
      pinky: { curl: 0.1, angle: -4, spread: 0 },
    },
  },
];

// 6. PLEASE: Flat open palm rubbing clockwise circle on chest
function getPleasePose(frame) {
  const f = Math.max(0, Math.min(SIGN_FRAMES_COUNT - 1, frame));
  const angleRad = (f / (SIGN_FRAMES_COUNT - 1)) * 2 * Math.PI;

  const radiusX = 22;
  const radiusY = 16;
  const wx = Math.sin(angleRad) * radiusX;
  const wy = -Math.cos(angleRad) * radiusY;
  const wAngle = Math.sin(angleRad) * 11;

  let micro = 'Trace clockwise circle on chest';
  if (f < 8) micro = 'Circle top-right across chest';
  else if (f < 16) micro = 'Circle downward across chest';
  else if (f < 23) micro = 'Circle bottom-left across chest';
  else micro = 'Circle upward back to center';

  return {
    wristX: wx,
    wristY: wy,
    wristAngle: wAngle,
    scale: 1.02,
    actionText: micro,
    thumb: { curl: 0.12, angle: -26, spread: 12 },
    index: { curl: 0.03, angle: -2, spread: 0 },
    middle: { curl: 0.02, angle: 0, spread: 0 },
    ring: { curl: 0.03, angle: 2, spread: 0 },
    pinky: { curl: 0.06, angle: 5, spread: 0 },
  };
}

// 7. SORRY: Closed "A" fist rubbing circular motion on chest/heart
function getSorryPose(frame) {
  const f = Math.max(0, Math.min(SIGN_FRAMES_COUNT - 1, frame));
  const angleRad = (f / (SIGN_FRAMES_COUNT - 1)) * 2 * Math.PI;

  const radiusX = 20;
  const radiusY = 15;
  const wx = Math.sin(angleRad) * radiusX;
  const wy = -Math.cos(angleRad) * radiusY;
  const wAngle = Math.sin(angleRad) * 9;

  let micro = 'Rub "A" fist circularly on chest';
  if (f < 8) micro = 'Circular fist motion across chest';
  else if (f < 16) micro = 'Downward circular rub';
  else if (f < 23) micro = 'Leftward circular rub';
  else micro = 'Complete circular rub';

  return {
    wristX: wx,
    wristY: wy,
    wristAngle: wAngle,
    scale: 1.01,
    actionText: micro,
    thumb: { curl: 0.22, angle: -10, spread: 0 }, // Upright thumb against index side
    index: { curl: 0.98, angle: 0, spread: 0 },
    middle: { curl: 0.98, angle: 0, spread: 0 },
    ring: { curl: 0.98, angle: 0, spread: 0 },
    pinky: { curl: 0.98, angle: 0, spread: 0 },
  };
}

// 8. HELP: Thumbs-up fist resting on flat support hand, lifting upward together
const HELP_KEYFRAMES = [
  {
    frame: 0,
    pose: {
      wristX: 0, wristY: 16, wristAngle: 0, scale: 1.0, actionText: 'Thumbs-up fist rests on base palm',
      thumb: { curl: 0.0, angle: 0, spread: 0 }, // Thumb straight upright
      index: { curl: 1.0, angle: 0, spread: 0 },
      middle: { curl: 1.0, angle: 0, spread: 0 },
      ring: { curl: 1.0, angle: 0, spread: 0 },
      pinky: { curl: 1.0, angle: 0, spread: 0 },
      supportHand: { active: true, x: 0, y: 16, angle: 0 },
    },
  },
  {
    frame: 6,
    pose: {
      wristX: 0, wristY: 10, wristAngle: 0, scale: 1.01, actionText: 'Begin lifting both hands together',
      thumb: { curl: 0.0, angle: 0, spread: 0 },
      index: { curl: 1.0, angle: 0, spread: 0 },
      middle: { curl: 1.0, angle: 0, spread: 0 },
      ring: { curl: 1.0, angle: 0, spread: 0 },
      pinky: { curl: 1.0, angle: 0, spread: 0 },
      supportHand: { active: true, x: 0, y: 10, angle: 0 },
    },
  },
  {
    frame: 16,
    pose: {
      wristX: 0, wristY: -26, wristAngle: 0, scale: 1.05, actionText: 'Elevate thumbs-up fist on support palm',
      thumb: { curl: 0.0, angle: 0, spread: 0 },
      index: { curl: 1.0, angle: 0, spread: 0 },
      middle: { curl: 1.0, angle: 0, spread: 0 },
      ring: { curl: 1.0, angle: 0, spread: 0 },
      pinky: { curl: 1.0, angle: 0, spread: 0 },
      supportHand: { active: true, x: 0, y: -26, angle: 0 },
    },
  },
  {
    frame: 22,
    pose: {
      wristX: 0, wristY: -26, wristAngle: 0, scale: 1.05, actionText: 'Hold elevated help gesture',
      thumb: { curl: 0.0, angle: 0, spread: 0 },
      index: { curl: 1.0, angle: 0, spread: 0 },
      middle: { curl: 1.0, angle: 0, spread: 0 },
      ring: { curl: 1.0, angle: 0, spread: 0 },
      pinky: { curl: 1.0, angle: 0, spread: 0 },
      supportHand: { active: true, x: 0, y: -26, angle: 0 },
    },
  },
  {
    frame: 29,
    pose: {
      wristX: 0, wristY: 16, wristAngle: 0, scale: 1.0, actionText: 'Lower back to starting position',
      thumb: { curl: 0.0, angle: 0, spread: 0 },
      index: { curl: 1.0, angle: 0, spread: 0 },
      middle: { curl: 1.0, angle: 0, spread: 0 },
      ring: { curl: 1.0, angle: 0, spread: 0 },
      pinky: { curl: 1.0, angle: 0, spread: 0 },
      supportHand: { active: true, x: 0, y: 16, angle: 0 },
    },
  },
];

// 9. OKAY: Index touches thumb forming "O", middle, ring, pinky upright
const OKAY_KEYFRAMES = [
  {
    frame: 0,
    pose: {
      wristX: 0, wristY: 0, wristAngle: 0, scale: 1.0, actionText: 'Form "O" with thumb and index, other fingers upright',
      thumb: { curl: 0.6, angle: -12, spread: 15 },
      index: { curl: 0.65, angle: 10, spread: 0 },
      middle: { curl: 0.05, angle: 0, spread: 8 },
      ring: { curl: 0.08, angle: 8, spread: 12 },
      pinky: { curl: 0.12, angle: 16, spread: 18 },
    },
  },
  {
    frame: 14,
    pose: {
      wristX: 0, wristY: -6, wristAngle: 0, scale: 1.03, actionText: 'Reassuring gentle forward pulse',
      thumb: { curl: 0.62, angle: -12, spread: 15 },
      index: { curl: 0.67, angle: 10, spread: 0 },
      middle: { curl: 0.04, angle: 0, spread: 8 },
      ring: { curl: 0.07, angle: 8, spread: 12 },
      pinky: { curl: 0.11, angle: 16, spread: 18 },
    },
  },
  {
    frame: 29,
    pose: {
      wristX: 0, wristY: 0, wristAngle: 0, scale: 1.0, actionText: 'Hold clear Okay handshape',
      thumb: { curl: 0.6, angle: -12, spread: 15 },
      index: { curl: 0.65, angle: 10, spread: 0 },
      middle: { curl: 0.05, angle: 0, spread: 8 },
      ring: { curl: 0.08, angle: 8, spread: 12 },
      pinky: { curl: 0.12, angle: 16, spread: 18 },
    },
  },
];

// 10. I_LOVE_YOU: Thumb, index, pinky upright, middle and ring curled
const I_LOVE_YOU_KEYFRAMES = [
  {
    frame: 0,
    pose: {
      wristX: 0, wristY: 0, wristAngle: 0, scale: 1.0, actionText: 'Extend thumb, index, and pinky upright',
      thumb: { curl: 0.05, angle: -38, spread: 28 },
      index: { curl: 0.05, angle: -4, spread: 0 },
      middle: { curl: 1.0, angle: 0, spread: 0 },
      ring: { curl: 1.0, angle: 0, spread: 0 },
      pinky: { curl: 0.05, angle: 14, spread: 18 },
    },
  },
  {
    frame: 14,
    pose: {
      wristX: 0, wristY: -8, wristAngle: -4, scale: 1.04, actionText: 'Gentle expressive tilt forward',
      thumb: { curl: 0.05, angle: -40, spread: 30 },
      index: { curl: 0.04, angle: -4, spread: 0 },
      middle: { curl: 1.0, angle: 0, spread: 0 },
      ring: { curl: 1.0, angle: 0, spread: 0 },
      pinky: { curl: 0.04, angle: 16, spread: 20 },
    },
  },
  {
    frame: 29,
    pose: {
      wristX: 0, wristY: 0, wristAngle: 0, scale: 1.0, actionText: 'Hold I Love You sign steady',
      thumb: { curl: 0.05, angle: -38, spread: 28 },
      index: { curl: 0.05, angle: -4, spread: 0 },
      middle: { curl: 1.0, angle: 0, spread: 0 },
      ring: { curl: 1.0, angle: 0, spread: 0 },
      pinky: { curl: 0.05, angle: 14, spread: 18 },
    },
  },
];

/**
 * Returns exact kinematic pose parameters for any sign at frame [0..29]
 */
export function getSignFrameData(signKey, frameIndex = 0) {
  const key = normalizeSignKey(signKey);
  const f = Math.max(0, Math.min(SIGN_FRAMES_COUNT - 1, Math.round(frameIndex)));

  switch (key) {
    case 'HELLO':
      return interpolateKeyframes(HELLO_KEYFRAMES, f);
    case 'YES':
      return interpolateKeyframes(YES_KEYFRAMES, f);
    case 'NO':
      return interpolateKeyframes(NO_KEYFRAMES, f);
    case 'WATER':
      return interpolateKeyframes(WATER_KEYFRAMES, f);
    case 'THANK_YOU':
      return interpolateKeyframes(THANK_YOU_KEYFRAMES, f);
    case 'PLEASE':
      return getPleasePose(f);
    case 'SORRY':
      return getSorryPose(f);
    case 'HELP':
      return interpolateKeyframes(HELP_KEYFRAMES, f);
    case 'OKAY':
      return interpolateKeyframes(OKAY_KEYFRAMES, f);
    case 'I_LOVE_YOU':
      return interpolateKeyframes(I_LOVE_YOU_KEYFRAMES, f);
    default:
      return interpolateKeyframes(HELLO_KEYFRAMES, f);
  }
}
