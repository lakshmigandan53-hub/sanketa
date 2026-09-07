import { useState } from 'react';
import './index.css';
import Home from './pages/Home';
import Camera from './pages/Camera';
import Translate from './pages/Translate';
import Conversation from './pages/Conversation';
import LearnISL from './pages/LearnISL';
import Quiz from './pages/Quiz';
import History from './pages/History';
import Emergency from './pages/Emergency';
import Settings from './pages/Settings';
import Profile from './pages/Profile';
import Assistant from './pages/Assistant';

const NAV = [
  { id: 'home',         icon: '🏠', label: 'Home' },
  { id: 'translate',    icon: '🔄', label: 'Translate' },
  { id: 'conversation', icon: '💬', label: 'Conversation' },
  { id: 'camera',       icon: '📷', label: 'Camera' },
  { id: 'learn',        icon: '📚', label: 'Learn ISL' },
  { id: 'quiz',         icon: '🎯', label: 'Quiz' },
  { id: 'assistant',    icon: '🤖', label: 'AI Assistant' },
  { id: 'history',      icon: '📋', label: 'History' },
  { id: 'emergency',    icon: '🚨', label: 'Emergency' },
  { id: 'settings',     icon: '⚙️', label: 'Settings' },
  { id: 'profile',      icon: '👤', label: 'Profile' },
];

export default function App() {
  const [page, setPage] = useState('home');
  const [lang, setLang] = useState('en-US');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const navigate = (id) => { setPage(id); setSidebarOpen(false); };

  const renderPage = () => {
    const props = { lang, setLang, setPage: navigate };
    switch (page) {
      case 'home':         return <Home {...props} />;
      case 'translate':    return <Translate {...props} />;
      case 'conversation': return <Conversation {...props} />;
      case 'camera':       return <Camera {...props} />;
      case 'learn':        return <LearnISL {...props} />;
      case 'quiz':         return <Quiz {...props} />;
      case 'assistant':    return <Assistant {...props} />;
      case 'history':      return <History {...props} />;
      case 'emergency':    return <Emergency {...props} />;
      case 'settings':     return <Settings {...props} />;
      case 'profile':      return <Profile {...props} />;
      default:             return <Home {...props} />;
    }
  };

  return (
    <div className="app-layout">
      {/* Mobile menu button */}
      <button
        className="mobile-menu-btn btn btn-ghost"
        onClick={() => setSidebarOpen(p => !p)}
        aria-label="Menu"
      >☰</button>

      {/* Sidebar overlay on mobile */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          style={{ position:'fixed', inset:0, background:'rgba(0,0,0,0.5)', zIndex:99 }}
        />
      )}

      {/* Sidebar */}
      <nav className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-logo">SAN<span>KETA</span></div>
        {NAV.map(n => (
          <button
            key={n.id}
            className={`nav-item ${page === n.id ? 'active' : ''}`}
            onClick={() => navigate(n.id)}
          >
            <span className="icon">{n.icon}</span>
            {n.label}
            {n.id === 'emergency' && (
              <span className="badge badge-red" style={{ marginLeft:'auto', padding:'2px 6px' }}>!</span>
            )}
          </button>
        ))}

        {/* Language quick-select */}
        <div style={{ marginTop:'auto', padding:'12px 8px', borderTop:'1px solid var(--border)', flexShrink: 0 }}>
          <div style={{ fontSize:'0.72rem', color:'var(--text3)', marginBottom:8, textTransform:'uppercase', letterSpacing:'0.05em' }}>Language</div>
          <div style={{ display:'flex', gap:6 }}>
            {[['en-US','EN'],['ta-IN','TA'],['hi-IN','HI']].map(([code, label]) => (
              <button
                key={code}
                className={`btn btn-sm ${lang === code ? 'btn-primary' : 'btn-ghost'}`}
                style={{ flex:1, justifyContent:'center', padding:'5px 0' }}
                onClick={() => setLang(code)}
              >{label}</button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main */}
      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  );
}
