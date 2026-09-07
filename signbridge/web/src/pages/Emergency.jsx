import { speak, SIGN_ASSETS, humanWord } from '../services';
import { SignDisplay } from './Translate';

const PHRASES = [
  { key: 'HELP',  text: 'HELP!',           icon: '🆘', speaking: 'Help! Help! Help!' },
  { key: 'HELP',  text: 'CALL POLICE',     icon: '🚔', speaking: 'Call the police! Call police now!' },
  { key: 'HELP',  text: 'CALL AMBULANCE',  icon: '🚑', speaking: 'Call an ambulance! Medical emergency!' },
  { key: 'HELP',  text: 'I NEED HELP',     icon: '🙋', speaking: 'I need help! Please help me!' },
  { key: 'HELP',  text: 'I AM IN DANGER',  icon: '⚠️', speaking: 'I am in danger! Please help me!' },
  { key: 'HELP',  text: 'PLEASE HELP ME',  icon: '🤲', speaking: 'Please help me! Please!' },
];

export default function Emergency({ lang }) {
  const handlePhrase = (phrase) => {
    speak(phrase.speaking || phrase.text, lang);
  };

  return (
    <div>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #dc2626, #991b1b)', borderRadius: 16, padding: '24px 28px', marginBottom: 24 }}>
        <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'white', marginBottom: 6 }}>
          🚨 EMERGENCY MODE
        </div>
        <div style={{ color: 'rgba(255,255,255,0.85)', fontSize: '1rem' }}>
          Tap any phrase to speak it immediately · Works even without internet
        </div>
      </div>

      {/* Emergency phrases */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginBottom: 28 }}>
        {PHRASES.map((p, i) => (
          <div key={i} style={{ display: 'flex', gap: 12, alignItems: 'stretch' }}>
            <button
              className="emergency-btn"
              style={{ flex: 1 }}
              onClick={() => handlePhrase(p)}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                <span style={{ fontSize: '1.8rem', flexShrink: 0 }}>{p.icon}</span>
                <div>
                  <div style={{ fontSize: '1.15rem', fontWeight: 800, letterSpacing: '0.02em' }}>{p.text}</div>
                  <div style={{ fontSize: '0.75rem', opacity: 0.65, marginTop: 2 }}>🔊 Tap to speak aloud</div>
                </div>
              </div>
            </button>

            {/* Sign display */}
            {SIGN_ASSETS[p.key] && (
              <div style={{
                background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)',
                borderRadius: 12, padding: '10px 14px',
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                minWidth: 90, textAlign: 'center', gap: 4,
              }}>
                <div style={{ fontSize: '2.2rem' }}>{SIGN_ASSETS[p.key].emoji}</div>
                <div style={{ fontSize: '0.65rem', color: 'var(--text2)', fontWeight: 600 }}>ISL: {humanWord(p.key)}</div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* HELP sign display */}
      <div className="grid-2" style={{ marginBottom: 24, alignItems: 'start' }}>
        <div className="card" style={{ border: '1px solid rgba(239,68,68,0.3)' }}>
          <div className="section-title">ISL Sign: HELP</div>
          <SignDisplay word="HELP" />
        </div>

        {/* Emergency contacts */}
        <div className="card" style={{ border: '1px solid rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.04)' }}>
          <div style={{ fontWeight: 700, color: 'var(--red)', marginBottom: 16, fontSize: '1rem' }}>
            ⚠️ Emergency Contacts (India)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, textAlign: 'center' }}>
            {[['Police', '100', '🚔'], ['Ambulance', '108', '🚑'], ['Emergency', '112', '🆘']].map(([n, num, icon]) => (
              <div key={n} style={{ background: 'rgba(239,68,68,0.1)', borderRadius: 10, padding: '16px 8px', border: '1px solid rgba(239,68,68,0.2)' }}>
                <div style={{ fontSize: '1.6rem', marginBottom: 4 }}>{icon}</div>
                <div style={{ fontWeight: 800, fontSize: '1.8rem', color: 'var(--red)' }}>{num}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text2)', marginTop: 4 }}>{n}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 16, padding: '10px 14px', background: 'rgba(239,68,68,0.08)', borderRadius: 8, fontSize: '0.82rem', color: 'var(--text2)' }}>
            💡 <strong style={{ color: '#fff' }}>Tip:</strong> Dial 112 from any mobile phone in India for all emergencies.
          </div>
        </div>
      </div>
    </div>
  );
}
