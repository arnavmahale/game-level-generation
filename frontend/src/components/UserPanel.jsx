import { useEffect, useState } from 'react';

export default function UserPanel({ user, stats, onLogout }) {
  return (
    <div className="user-panel">
      <div className="user-row">
        <span className="user-name">👤 {user.username}</span>
        <button className="link-btn" onClick={onLogout}>Log out</button>
      </div>
      <div className="stats-block">
        <div className="stat-line">
          <span>Endless best</span>
          <strong>{stats?.endless_best ?? 0}</strong>
        </div>
        <div className="stat-line">
          <span>VAE completed</span>
          <strong>{stats?.completions?.vae ?? 0}</strong>
        </div>
        <div className="stat-line">
          <span>Bigram completed</span>
          <strong>{stats?.completions?.bigram ?? 0}</strong>
        </div>
        <div className="stat-line">
          <span>Naive completed</span>
          <strong>{stats?.completions?.naive ?? 0}</strong>
        </div>
      </div>
    </div>
  );
}

export function Leaderboard({ onClose }) {
  const [endless, setEndless] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch('/api/leaderboard/endless', { credentials: 'include' })
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
