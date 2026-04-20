import { useEffect, useState } from 'react';

export default function UserPanel({ user, stats, onLogout, onShowLeaderboard }) {
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
      <button className="link-btn full" onClick={onShowLeaderboard}>
        View leaderboards →
      </button>
    </div>
  );
}

export function Leaderboard({ onClose }) {
  const [vae, setVae] = useState(null);
  const [endless, setEndless] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch('/api/leaderboard/vae', { credentials: 'include' }).then((r) => r.json()),
      fetch('/api/leaderboard/endless', { credentials: 'include' }).then((r) => r.json()),
    ])
      .then(([a, b]) => { setVae(a.rows || []); setEndless(b.rows || []); })
      .catch((e) => setErr(e.message));
  }, []);

  return (
    <div className="leaderboard-overlay" onClick={onClose}>
      <div className="leaderboard-card" onClick={(e) => e.stopPropagation()}>
        <div className="leaderboard-head">
          <h2>Leaderboards</h2>
          <button className="link-btn" onClick={onClose}>Close</button>
        </div>
        {err && <div className="auth-error">{err}</div>}
        <div className="leaderboard-cols">
          <div>
            <h3>Endless — high score</h3>
            {endless === null ? <p>Loading…</p> :
              endless.length === 0 ? <p className="empty">No scores yet.</p> : (
                <ol>
                  {endless.map((r, i) => (
                    <li key={i}><span>{r.username}</span><strong>{r.score}</strong></li>
                  ))}
                </ol>
              )}
          </div>
          <div>
            <h3>VAE — levels completed</h3>
            {vae === null ? <p>Loading…</p> :
              vae.length === 0 ? <p className="empty">No runs yet.</p> : (
                <ol>
                  {vae.map((r, i) => (
                    <li key={i}><span>{r.username}</span><strong>{r.count}</strong></li>
                  ))}
                </ol>
              )}
          </div>
        </div>
      </div>
    </div>
  );
}
