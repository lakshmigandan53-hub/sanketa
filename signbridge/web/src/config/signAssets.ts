// SANKETA — Centralized Indian Sign Language (ISL) Asset Registry (TypeScript)
// web/src/config/signAssets.ts

export interface SignAsset {
  word: string;
  label: string;
  type: 'static' | 'dynamic';
  verifiedIllustrated?: boolean;
  video?: string | null;
  animation?: string | null;
  description: string;
  tip?: string;
  available: boolean;
}

export { SIGN_ASSETS, ISL_WORDS, normalizeISLWord, extractISLSequence, formatISLWord } from './signAssets.js';
