import { useState, useEffect } from 'react';
import { getProfile, saveProfile, LANGUAGES } from '../services';

export default function Profile({ lang, setLang }) {
  const [profile, setProfile] = useState(getProfile);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setProfile(getProfile());
  }, []);

  const update = (key, value) => {
    setProfile(p => ({ ...p, [key]: value }));
    setSaved(false);
  };

  const handleSave = () => {
    saveProfile(profile);
    if (profile.preferredLanguage) setLang(profile.preferredLanguage);
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  const levels = ['beginner', 'intermediate', 'advanced'];
  const goals = [
    'Learn ISL for daily communication',
    'Help a family member with ISL',
    'Academic/Research purposes',
    'Professional interpreter training',
    'Personal interest',
  ];

  const stats = [
    { label: 'Words Learned', value: '8', icon: '📚' },
    { label: 'Quiz Attempts', value: '—', icon: '🎯' },
    { label: 'Signs Practiced', value: '—', icon: '🤟' },
  ];

  return (
    <div>
      <div className="page-title">👤 Profile</div>
      <div className="page-sub">Your SANKETA learning profile — saved locally</div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
        {/* Profile form */}
        <div>
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="section-title">Personal Info</div>

            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'block', fontSize: '0.82rem', color: 'var(--text3)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Display Name
              </label>
              <input
                className="input-field"
                type="text"
                placeholder="Enter your name…"
                value={profile.name}
                onChange={e => update('name', e.target.value)}
              />
            </div>

            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'block', fontSize: '0.82rem', color: 'var(--text3)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Preferred Language
              </label>
              <div style={{ display: 'flex', gap: 8 }}>
                {LANGUAGES.map(l => (
                  <button
                    key={l.code}
                    className={`btn btn-sm ${profile.preferredLanguage === l.code ? 'btn-primary' : 'btn-outline'}`}
                    style={{ flex: 1, justifyContent: 'center' }}
                    onClick={() => update('preferredLanguage', l.code)}
                  >
                    {l.flag} {l.label}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'block', fontSize: '0.82rem', color: 'var(--text3)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Experience Level
              </label>
              <div style={{ display: 'flex', gap: 8 }}>
                {levels.map(l => (
                  <button
                    key={l}
                    className={`btn btn-sm ${profile.level === l ? 'btn-primary' : 'btn-outline'}`}
                    style={{ flex: 1, justifyContent: 'center', textTransform: 'capitalize' }}
                    onClick={() => update('level', l)}
                  >
                    {l}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ marginBottom: 20 }}>
              <label style={{ display: 'block', fontSize: '0.82rem', color: 'var(--text3)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Learning Goal
              </label>
              <select
                className="input-field"
                value={profile.learningGoal}
                onChange={e => update('learningGoal', e.target.value)}
                style={{ cursor: 'pointer' }}
              >
                <option value="">Select a goal…</option>
                {goals.map(g => <option key={g} value={g}>{g}</option>)}
              </select>
            </div>

            <button className="btn btn-primary" style={{ width: '100%', justifyContent: 'center' }} onClick={handleSave}>
              {saved ? '✅ Saved!' : '💾 Save Profile'}
            </button>
          </div>
        </div>

        {/* Right column: avatar + stats */}
        <div>
          {/* Avatar display */}
          <div className="card" style={{ textAlign: 'center', marginBottom: 20 }}>
            <div style={{
              width: 80, height: 80, borderRadius: '50%',
              background: 'linear-gradient(135deg, var(--primary), var(--secondary))',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '2.2rem', margin: '0 auto 12px',
            }}>
              {profile.name ? profile.name[0].toUpperCase() : '👤'}
            </div>
            <div style={{ fontWeight: 700, fontSize: '1.2rem', marginBottom: 4 }}>
              {profile.name || 'Anonymous User'}
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', gap: 8 }}>
              <span className="badge badge-purple" style={{ textTransform: 'capitalize' }}>
                {profile.level}
              </span>
              <span className="badge badge-blue">
                {LANGUAGES.find(l => l.code === profile.preferredLanguage)?.name || 'English'}
              </span>
            </div>
            {profile.learningGoal && (
              <div style={{ marginTop: 12, fontSize: '0.82rem', color: 'var(--text2)', fontStyle: 'italic' }}>
                "{profile.learningGoal}"
              </div>
            )}
          </div>

          {/* Stats */}
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="section-title">Your Stats</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              {stats.map(s => (
                <div key={s.label} style={{ textAlign: 'center', background: 'var(--bg-glass)', borderRadius: 12, padding: '14px 8px' }}>
                  <div style={{ fontSize: '1.6rem', marginBottom: 4 }}>{s.icon}</div>
                  <div style={{ fontWeight: 700, fontSize: '1.2rem', color: 'var(--primary)' }}>{s.value}</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text3)', marginTop: 2 }}>{s.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* ISL vocabulary */}
          <div className="card">
            <div className="section-title">Available ISL Signs</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {['HELLO', 'YES', 'NO', 'WATER', 'THANK_YOU', 'PLEASE', 'SORRY', 'HELP'].map(w => (
                <span key={w} className="badge badge-green">{w.replace('_', ' ')}</span>
              ))}
            </div>
            <div style={{ marginTop: 12, fontSize: '0.8rem', color: 'var(--text3)' }}>
              4 static + 4 dynamic signs available for recognition
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
