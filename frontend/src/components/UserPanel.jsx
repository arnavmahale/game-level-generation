import { useEffect, useState } from 'react';
import { apiFetch } from '../utils/api';

export default function UserPanel({ user, stats, onLogout }) {
  const isGuest = !!user.isGuest;
  return (
    <div className="user-menu" tabIndex={0}>
      <button className="user-menu-trigger" aria-label="User menu">
        <span className="user-menu-icon">👤</span>
        <span className="user-menu-name">{isGuest ? 'Guest' : user.username}</span>
      </button>
      <div className="user-menu-dropdown">
        <div className="dropdown-header">
          {isGuest ? 'Playing as guest' : <>Signed in as <strong>{user.username}</strong></>}
        </div>
        <div className="dropdown-stats">
          <div className="stat-line">
            <span>Endless best{isGuest ? ' (session)' : ''}</span>
            <strong>{stats?.endless_best ?? 0}</strong>
          </div>
          <div className="stat-line">
            <span>Levels completed{isGuest ? ' (session)' : ''}</span>
            <strong>{stats?.completions?.vae ?? 0}</strong>
          </div>
        </div>
        <button className="dropdown-logout" onClick={onLogout}>
          {isGuest ? 'Log in / Sign up' : 'Log out'}
        </button>
      </div>
    </div>
  );
}

export function Leaderboard({ onClose, isGuest = false }) {
  const [endless, setEndless] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    apiFetch('/api/leaderboard/endless')
      .then((r) => r.json())
      .then((b) => setEndless((b.rows || []).slice(0, 10)))
      .catch((e) => setErr(e.message));
  }, []);

  return (
    <div className="leaderboard-overlay" onClick={onClose}>
      <div className="leaderboard-card" onClick={(e) => e.stopPropagation()}>
        <div className="leaderboard-head">
          <h2>🏆 Endless leaderboard</h2>
          <button className="link-btn" onClick={onClose}>Close</button>
        </div>
        {isGuest && (
          <div className="leaderboard-guest-note">
            Log in to appear on the leaderboard.
          </div>
        )}
        {err && <div className="auth-error">{err}</div>}
        <div className="leaderboard-single">
          {endless === null ? <p>Loading…</p> :
            endless.length === 0 ? <p className="empty">No scores yet. Be the first!</p> : (
              <ol>
                {endless.map((r, i) => (
                  <li key={i}>
                    <span className="lb-rank">#{i + 1}</span>
                    <span className="lb-name">{r.username}</span>
                    <strong className="lb-score">{r.score}</strong>
                  </li>
                ))}
              </ol>
            )}
        </div>
      </div>
    </div>
  );
}
