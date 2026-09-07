import { useState, useEffect } from 'react';
import { LANGUAGES, ISL_WORDS, getSettings, saveSettings, API_BASE } from '../services';

export default function Settings({ lang, setLang }) {
  const [settings, setSettings] = useState(getSettings);
  const [saved, setSaved] = useState(false);

  // Sync language from parent
  useEffect(() => {
    updateSetting('language', lang);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lang]);

  const updateSetting = (key, value) => {
    const updated = saveSettings({ [key]: value });
    setSettings(updated);
    setSaved(false);
    if (key === 'language') setLang(value);
  };

  const handleSave = () => {
    saveSettings(settings);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div>
      <div className="page-title">⚙️ Settings</div>
      <div className="page-sub">Configure SANKETA preferences — all settings saved to localStorage</div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
        <div>
          {/* Language */}
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="section-title">🌐 Language</div>
            <p style={{ color: 'var(--text2)', fontSize: '0.85rem', marginBottom: 16 }}>
              Affects Speech Recognition, Text-to-Speech output, and word translations
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {LANGUAGES.map(l => (
                <button
                  key={l.code}
                  className={`btn ${settings.language === l.code ? 'btn-primary' : 'btn-outline'}`}
                  style={{ justifyContent: 'flex-start', gap: 16 }}
                  onClick={() => updateSetting('language', l.code)}
                >
                  <span>{l.flag}</span>
                  <span className="badge badge-purple" style={{ minWidth: 32 }}>{l.label}</span>
                  {l.name}
                  {settings.language === l.code && <span style={{ marginLeft: 'auto' }}>✓</span>}
                </button>
              ))}
            </div>
          </div>

          {/* Speech settings */}
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="section-title">🔊 Speech Settings</div>

            <div className="toggle-row">
              <div>
                <div style={{ fontWeight: 600, marginBottom: 2 }}>Text-to-Speech</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text3)' }}>Speak recognized words aloud</div>
              </div>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.speechEnabled}
                  onChange={e => updateSetting('speechEnabled', e.target.checked)}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>

            <div className="toggle-row">
              <div>
                <div style={{ fontWeight: 600, marginBottom: 2 }}>Auto-Speak</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text3)' }}>Automatically speak on recognition</div>
              </div>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.autoSpeak}
                  onChange={e => updateSetting('autoSpeak', e.target.checked)}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>

            <div style={{ marginTop: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: '0.88rem' }}>
                <span style={{ fontWeight: 600 }}>Speech Rate</span>
                <span style={{ color: 'var(--primary)', fontWeight: 700 }}>{settings.speechRate.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="0.5" max="2.0" step="0.1"
                value={settings.speechRate}
                onChange={e => updateSetting('speechRate', parseFloat(e.target.value))}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text3)', marginTop: 4 }}>
                <span>0.5× (slow)</span>
                <span>1.0× (normal)</span>
                <span>2.0× (fast)</span>
              </div>
            </div>
          </div>
        </div>

        <div>
          {/* Recognition settings */}
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="section-title">🎯 Recognition Settings</div>

            <div style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: '0.88rem' }}>
                <span style={{ fontWeight: 600 }}>Confidence Threshold</span>
                <span style={{ color: 'var(--primary)', fontWeight: 700 }}>{Math.round(settings.confidenceThreshold * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.3" max="0.95" step="0.05"
                value={settings.confidenceThreshold}
                onChange={e => updateSetting('confidenceThreshold', parseFloat(e.target.value))}
              />
              <div style={{ fontSize: '0.78rem', color: 'var(--text3)', marginTop: 8, lineHeight: 1.6 }}>
                Signs below this confidence level will be marked as "Uncertain".
                Lower = more results but less accurate. Higher = fewer but more reliable results.
              </div>
            </div>
          </div>

          {/* Backend info */}
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="section-title">🔧 Backend</div>
            <div style={{ color: 'var(--text2)', fontSize: '0.85rem', lineHeight: 2 }}>
              <div>API: <code style={{ color: 'var(--accent)' }}>{API_BASE}</code></div>
              <div>Static model: <code style={{ color: 'var(--accent)' }}>POST /translate/image</code></div>
              <div>Dynamic model: <code style={{ color: 'var(--accent)' }}>POST /translate/burst</code></div>
              <div>Health: <code style={{ color: 'var(--accent)' }}>GET /health</code></div>
            </div>
          </div>

          {/* Configured words */}
          <div className="card">
            <div className="section-title">📝 Configured Signs ({ISL_WORDS.length})</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
              {ISL_WORDS.map(w => <span key={w} className="badge badge-purple">{w}</span>)}
            </div>
            <p style={{ color: 'var(--text3)', fontSize: '0.78rem' }}>
              4 static (HELLO, YES, NO, WATER) + 4 dynamic (THANK_YOU, PLEASE, SORRY, HELP)
            </p>
          </div>
        </div>
      </div>

      {/* Save button */}
      <div style={{ marginTop: 24 }}>
        <button className="btn btn-primary btn-lg" onClick={handleSave}>
          {saved ? '✅ Settings Saved!' : '💾 Save All Settings'}
        </button>
      </div>
    </div>
  );
}
