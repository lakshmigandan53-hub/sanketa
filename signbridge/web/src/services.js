import { SIGN_ASSETS, ISL_WORDS, normalizeISLWord, extractISLSequence, formatISLWord } from './config/signAssets';

export const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export { SIGN_ASSETS, ISL_WORDS, normalizeISLWord, extractISLSequence, formatISLWord };
export const normalizeToISLKey = normalizeISLWord;

// ── API calls ────────────────────────────────────────────────────────────────
export async function apiHealth() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 6000);
  try {
    const r = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    clearTimeout(timeoutId);
    return await r.json();
  } catch (e) {
    clearTimeout(timeoutId);
    throw new Error('Backend offline or unreachable at ' + API_BASE);
  }
}

export async function apiTranslateImage(blob) {
  const fd = new FormData();
  fd.append('file', blob, 'frame.jpg');
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 8000);
  try {
    const r = await fetch(`${API_BASE}/translate/image`, {
      method: 'POST',
      body: fd,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return await r.json();
  } catch (e) {
    clearTimeout(timeoutId);
    throw new Error(e.name === 'AbortError' ? 'Translation request timed out' : 'Network error connecting to backend');
  }
}

export async function apiTranslateLandmarks(landmarks) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 6000);
  try {
    const r = await fetch(`${API_BASE}/translate/landmarks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ landmarks }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return await r.json();
  } catch (e) {
    clearTimeout(timeoutId);
    throw new Error('Network error connecting to backend');
  }
}

export async function apiTranslateSequence(sequence) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 8000);
  try {
    const r = await fetch(`${API_BASE}/translate/sequence`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return await r.json();
  } catch (e) {
    clearTimeout(timeoutId);
    throw new Error('Network error connecting to backend');
  }
}

export async function apiTranslateBurst(blobs) {
  const fd = new FormData();
  blobs.forEach((b, i) => {
    fd.append('files', b, `frame_${i}.jpg`);
  });
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 12000);
  try {
    const r = await fetch(`${API_BASE}/translate/burst`, {
      method: 'POST',
      body: fd,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return await r.json();
  } catch (e) {
    clearTimeout(timeoutId);
    throw new Error('Failed to process dynamic gesture burst');
  }
}

export async function apiResetBuffer() {
  try {
    const r = await fetch(`${API_BASE}/translate/reset-buffer`, { method: 'POST' });
    return await r.json();
  } catch (e) {
    return { success: false };
  }
}


export async function apiModelInfo() {
  const r = await fetch(`${API_BASE}/model/info`);
  return r.json();
}

// ── Text-to-Speech ───────────────────────────────────────────────────────────
export function speak(text, lang = 'en-US', rate = 1.0) {
  if (!window.speechSynthesis || !text) return;
  const settings = getSettings();
  if (!settings.speechEnabled) return;
  const effectiveRate = rate !== 1.0 ? rate : (settings.speechRate || 1.0);
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang;
  u.rate = effectiveRate;
  u.pitch = 1.0;
  window.speechSynthesis.speak(u);
}

// ── Speech Recognition ───────────────────────────────────────────────────────
export function createRecognition(lang = 'en-US', onResult, onEnd, onError) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return null;
  const r = new SR();
  r.continuous = false;
  r.interimResults = false;
  r.lang = lang;
  r.onresult = e => {
    const t = Array.from(e.results).map(res => res[0].transcript).join(' ');
    onResult(t);
  };
  r.onend = onEnd || (() => {});
  r.onerror = onError || (() => {});
  return r;
}

// ── Language helpers ─────────────────────────────────────────────────────────
export const LANGUAGES = [
  { code: 'en-US', name: 'English', label: 'EN', flag: '🇬🇧' },
  { code: 'ta-IN', name: 'Tamil',   label: 'TA', flag: '🇮🇳' },
  { code: 'hi-IN', name: 'Hindi',   label: 'HI', flag: '🇮🇳' },
];

export function humanWord(word) {
  if (!word || word === 'uncertain') return word;
  const map = {
    HELLO: 'Hello',
    YES: 'Yes',
    NO: 'No',
    WATER: 'Water',
    FOOD: 'Food',
    MILK: 'Milk',
    TEA: 'Tea',
    BOOK: 'Book',
    PEN: 'Pen',
    PHONE: 'Phone',
    HELP: 'Help',
    THANK_YOU: 'Thank you',
    PLEASE: 'Please',
    SORRY: 'Sorry',
  };
  return map[word] || word.replace(/_/g, ' ');
}

// ── Translation (local static fallback) ─────────────────────────────────────
const TRANSLATIONS = {
  'ta-IN': {
    HELLO: 'வணக்கம்',
    YES: 'ஆம்',
    NO: 'இல்லை',
    WATER: 'தண்ணீர்',
    FOOD: 'உணவு',
    MILK: 'பால்',
    TEA: 'தேநீர்',
    BOOK: 'புத்தகம்',
    PEN: 'பேனா',
    PHONE: 'தொலைபேசி',
    HELP: 'உதவி',
    THANK_YOU: 'நன்றி',
    PLEASE: 'தயவுசெய்து',
    SORRY: 'மன்னிக்கவும்',
  },
  'hi-IN': {
    HELLO: 'नमस्ते',
    YES: 'हाँ',
    NO: 'नहीं',
    WATER: 'पानी',
    FOOD: 'खाना',
    MILK: 'दूध',
    TEA: 'चाय',
    BOOK: 'किताब',
    PEN: 'कलम',
    PHONE: 'फ़ोन',
    HELP: 'मदद',
    THANK_YOU: 'धन्यवाद',
    PLEASE: 'कृपया',
    SORRY: 'माफ़ करें',
  },
};

export function translateWord(word, targetLang) {
  if (!word || word === 'uncertain') return word;
  if (targetLang === 'en-US') return humanWord(word);
  return TRANSLATIONS[targetLang]?.[word] || humanWord(word);
}

// ── localStorage history ─────────────────────────────────────────────────────
const HIST_KEY = 'sb_history';

export function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HIST_KEY) || '[]');
  } catch {
    return [];
  }
}

export function addHistory(entry) {
  const h = getHistory();
  h.unshift({
    ...entry,
    id: Date.now() + Math.random(),
    ts: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
  });
  if (h.length > 100) h.pop();
  localStorage.setItem(HIST_KEY, JSON.stringify(h));
}

export function deleteHistory(id) {
  const h = getHistory().filter(item => item.id !== id);
  localStorage.setItem(HIST_KEY, JSON.stringify(h));
}

export function clearHistory() {
  localStorage.removeItem(HIST_KEY);
}

// ── localStorage settings ────────────────────────────────────────────────────
const SETTINGS_KEY = 'sb_settings';
const DEFAULT_SETTINGS = {
  speechEnabled: true,
  speechRate: 1.0,
  confidenceThreshold: 0.65,
  theme: 'dark',
  language: 'en-US',
  autoSpeak: true,
};

export function getSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
    return { ...DEFAULT_SETTINGS, ...saved };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(updates) {
  const current = getSettings();
  const merged = { ...current, ...updates };
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(merged));
  return merged;
}

// ── localStorage profile ─────────────────────────────────────────────────────
const PROFILE_KEY = 'sb_profile';
const DEFAULT_PROFILE = {
  name: '',
  preferredLanguage: 'en-US',
  learningGoal: '',
  level: 'beginner',
};

export function getProfile() {
  try {
    const saved = JSON.parse(localStorage.getItem(PROFILE_KEY) || '{}');
    return { ...DEFAULT_PROFILE, ...saved };
  } catch {
    return { ...DEFAULT_PROFILE };
  }
}

export function saveProfile(updates) {
  const current = getProfile();
  const merged = { ...current, ...updates };
  localStorage.setItem(PROFILE_KEY, JSON.stringify(merged));
  return merged;
}

// ── Confidence color ─────────────────────────────────────────────────────────
export function confColor(c) {
  if (c >= 0.8) return 'var(--green)';
  if (c >= 0.5) return 'var(--orange)';
  return 'var(--red)';
}

// ── AI Grammar assist & word suggestions (local rule-based) ──────────────────
export function grammarAssist(text) {
  if (!text) return null;
  const t = text.trim();
  let result = t.charAt(0).toUpperCase() + t.slice(1).toLowerCase();
  // Add period if no ending punctuation
  if (!/[.?!]$/.test(result)) result += '.';
  // Fix common patterns
  result = result.replace(/\bi\b/g, 'I');
  result = result.replace(/\bim\b/gi, "I'm");
  result = result.replace(/\bdont\b/gi, "don't");
  result = result.replace(/\bcant\b/gi, "can't");
  result = result.replace(/\bwont\b/gi, "won't");
  return { corrected: result, original: t, changed: result.toLowerCase() !== t.toLowerCase() };
}

export function wordSuggestions(word) {
  const map = {
    HELLO:     ['Hi', 'Greetings', 'Hey'],
    SORRY:     ['Excuse me', 'Pardon', 'Forgive me'],
    HELP:      ['Assist', 'Support', 'Aid'],
    WATER:     ['Drink', 'Hydrate'],
    THANK_YOU: ['Thanks', 'Appreciate it', 'Gratitude'],
    YES:       ['Sure', 'Okay', 'Agreed'],
    NO:        ['Nope', 'Negative', 'Disagree'],
    PLEASE:    ['Kindly', 'Request', 'If you may'],
  };
  return map[word?.toUpperCase()] || [];
}

export function sentenceImprove(text) {
  if (!text) return null;
  const improved = grammarAssist(text);
  const suggestions = [];

  if (text.split(' ').length <= 2) {
    suggestions.push('Add more context to make your message clearer.');
  }
  if (!text.includes('please') && !text.includes('thank')) {
    suggestions.push('Consider adding polite words like "please" or "thank you".');
  }
  suggestions.push('Your sentence has been capitalized and punctuated correctly.');

  return {
    original: text,
    improved: improved?.corrected || text,
    suggestions,
  };
}
