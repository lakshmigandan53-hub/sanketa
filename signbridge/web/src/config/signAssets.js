// SANKETA — Centralized Indian Sign Language (ISL) Asset Registry
// web/src/config/signAssets.js

/**
 * Central configuration of verified ISL signs.
 * Used across Home, Camera, Translate, Learn ISL, Quiz, and Conversation.
 * 
 * Strict Validation Policy:
 * - verifiedIllustrated: true ONLY if the asset is a verified illustrated ISL hand-motion animation
 *   matching the reference style (clean illustrated human hand/forearm on a simple light background).
 * - MediaPipe skeleton/landmark renders, emojis, and ASL videos are strictly excluded from being
 *   passed off as verified illustrated ISL animations.
 */
export const SIGN_ASSETS = {
  HELLO: {
    word: 'HELLO',
    label: 'Hello',
    type: 'static',
    video: '/signs/hello.mp4',
    gif: null,
    image: '/signs/HELLO.png',
    description: 'Open palm facing forward at head level, waving gently side to side with natural wrist movement.',
    tip: 'Keep your palm open, fingers together, and wave smoothly from the wrist.',
    available: true,
  },
  YES: {
    word: 'YES',
    label: 'Yes',
    type: 'static',
    video: '/signs/yes.mp4',
    gif: null,
    image: '/signs/YES.png',
    description: 'Closed fist held in front of chest, nodding up and down from the wrist like a head nod.',
    tip: 'Form an "S" fist and nod forward and down twice smoothly.',
    available: true,
  },
  NO: {
    word: 'NO',
    label: 'No',
    type: 'static',
    video: '/signs/no.mp4',
    gif: null,
    image: '/signs/NO.png',
    description: 'Index and middle fingers extended together, snapping down onto the thumb.',
    tip: 'Extend thumb, index, and middle finger, then tap them firmly shut together.',
    available: true,
  },
  WATER: {
    word: 'WATER',
    label: 'Water',
    type: 'static',
    video: null,
    gif: null,
    image: '/signs/WATER.png',
    description: '"W" handshape (three middle fingers upright) tapping lightly against the side of the chin twice.',
    tip: 'Form a "W" with index, middle, and ring fingers upright; tap your chin lightly.',
    available: true,
  },
  THANK_YOU: {
    word: 'THANK_YOU',
    label: 'Thank You',
    type: 'dynamic',
    video: '/signs/thank-you.mp4',
    gif: '/signs/THANK_YOU.gif',
    image: '/signs/THANK_YOU.png',
    description: 'Flat hand with fingertips touching chin or lips, extending forward and slightly downward toward the listener.',
    tip: 'Start with fingertips touching your chin, then move your flat palm forward smoothly.',
    available: true,
  },
  PLEASE: {
    word: 'PLEASE',
    label: 'Please',
    type: 'dynamic',
    video: null,
    gif: '/signs/PLEASE.gif',
    image: '/signs/PLEASE.png',
    description: 'Flat open palm rubbed in a smooth clockwise circular motion over the center of the chest.',
    tip: 'Place flat hand on the center of your chest and circle smoothly 2-3 times.',
    available: true,
  },
  SORRY: {
    word: 'SORRY',
    label: 'Sorry',
    type: 'dynamic',
    video: '/signs/sorry.mp4',
    gif: '/signs/SORRY.gif',
    image: '/signs/SORRY.png',
    description: 'Closed fist rubbed in a circular motion on the chest over the heart with an apologetic expression.',
    tip: 'Make an "A" fist and rub circular motions over your chest.',
    available: true,
  },
  HELP: {
    word: 'HELP',
    label: 'Help',
    type: 'dynamic',
    video: '/signs/help.mp4',
    gif: '/signs/HELP.gif',
    image: '/signs/HELP.png',
    description: 'Closed fist with thumb upright (thumbs-up) resting on the flat open palm of the other hand, lifted upward together.',
    tip: 'Support the dominant fist (thumbs-up) on your flat open non-dominant palm, lifting both up.',
    available: true,
  },
  OKAY: {
    word: 'OKAY',
    label: 'Okay',
    type: 'static',
    video: null,
    gif: null,
    image: '/signs/OKAY.png',
    description: 'Thumb and index finger touching tips to form an "O" circle, with remaining three fingers upright and slightly spread.',
    tip: 'Touch thumb tip to index tip to form a loop; keep the other three fingers upright.',
    available: true,
  },
  I_LOVE_YOU: {
    word: 'I_LOVE_YOU',
    label: 'I Love You',
    type: 'static',
    video: null,
    gif: null,
    image: '/signs/I_LOVE_YOU.png',
    description: 'Thumb, index finger, and pinky finger extended upright while middle and ring fingers are folded against palm.',
    tip: 'Extend your thumb, index, and pinky upright while keeping middle and ring fingers tucked into the palm.',
    available: true,
  },
};

export const ISL_WORDS = Object.keys(SIGN_ASSETS);

/**
 * Normalizes any spoken, typed, or detected input into a canonical ISL key.
 * Handles variations like "hello", "Hello", "HELLO", "thank you", "THANK YOU", "THANK_YOU", etc.
 */
export function normalizeISLWord(input) {
  if (!input || typeof input !== 'string') return null;

  const raw = input.trim().toUpperCase();
  const underscore = raw.replace(/[\s\-_]+/g, '_');

  // Direct match
  if (SIGN_ASSETS[underscore]) {
    return underscore;
  }

  // Common spoken aliases
  const aliases = {
    'HI': 'HELLO',
    'HEY': 'HELLO',
    'GREETINGS': 'HELLO',
    'THANKS': 'THANK_YOU',
    'THANKYOU': 'THANK_YOU',
    'THANK YOU': 'THANK_YOU',
    'THX': 'THANK_YOU',
    'YEAH': 'YES',
    'YEP': 'YES',
    'OKAY': 'OKAY',
    'OK': 'OKAY',
    'I LOVE YOU': 'I_LOVE_YOU',
    'ILOVEYOU': 'I_LOVE_YOU',
    'ILY': 'I_LOVE_YOU',
    'NAH': 'NO',
    'NOPE': 'NO',
    'DRINK': 'WATER',
    'THIRSTY': 'WATER',
    'PLZ': 'PLEASE',
    'PLS': 'PLEASE',
    'APOLOGIZE': 'SORRY',
    'MY BAD': 'SORRY',
    'ASSIST': 'HELP',
    'ASSISTANCE': 'HELP',
    'EMERGENCY': 'HELP',
  };

  if (aliases[raw] || aliases[underscore]) {
    return aliases[raw] || aliases[underscore];
  }

  // Word extraction inside string
  for (const word of ISL_WORDS) {
    const spaceForm = word.replace(/_/g, ' ');
    const regex = new RegExp(`\\b${spaceForm}\\b|\\b${word}\\b`, 'i');
    if (regex.test(input)) {
      return word;
    }
  }

  return null;
}

/**
 * Extracts a sequence of supported ISL words from a sentence.
 * e.g., "Hello please help me" -> ["HELLO", "PLEASE", "HELP"]
 */
export function extractISLSequence(sentence) {
  if (!sentence || typeof sentence !== 'string') return [];

  const found = [];
  const words = sentence.trim().toUpperCase().split(/\s+/);

  let i = 0;
  while (i < words.length) {
    // Check 2-word phrase first (e.g. "THANK YOU")
    if (i + 1 < words.length) {
      const twoWord = `${words[i]}_${words[i + 1]}`;
      const norm2 = normalizeISLWord(twoWord);
      if (norm2) {
        found.push(norm2);
        i += 2;
        continue;
      }
    }

    // Single word check
    const norm1 = normalizeISLWord(words[i]);
    if (norm1) {
      found.push(norm1);
    }
    i++;
  }

  return found;
}

/**
 * Formats canonical ISL key for human display ("THANK_YOU" -> "Thank You")
 */
export function formatISLWord(word) {
  if (!word) return '';
  if (SIGN_ASSETS[word]?.label) return SIGN_ASSETS[word].label;
  return word
    .split('_')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(' ');
}
