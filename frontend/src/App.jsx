import { useState, useRef, useCallback, useEffect } from 'react';
import ControlPanel from './components/ControlPanel';
import LevelCanvas from './components/LevelCanvas';
import GameCanvas from './components/GameCanvas';
import AuthPage from './components/AuthPage';
import UserPanel, { Leaderboard } from './components/UserPanel';
import { apiFetch, clearToken } from './utils/api';
import './App.css';

async function fetchChunk({ model = 'vae', difficulty = 50, seed = null, repair = true } = {}) {
  const res = await apiFetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, difficulty, seed, repair }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || 'Generation failed');
  }
  return res.json();
}

async function postScore(path, body) {
  try {
    const res = await apiFetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      console.error(`[score] ${path} ${res.status}`, data);
      return { __error: data.error || `HTTP ${res.status}` };
    }
    console.log(`[score] ${path} OK`, data);
    return data;
  } catch (e) {
    console.error(`[score] ${path} threw`, e);
    return { __error: e.message };
  }
}

export default function App() {
  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [stats, setStats] = useState(null);
  const [showLeaderboard, setShowLeaderboard] = useState(false);

  const [level, setLevel] = useState(null);
  const [chunks, setChunks] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [paramsUsed, setParamsUsed] = useState(null);
  const [currentModel, setCurrentModel] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('play');
  const [endlessScore, setEndlessScore] = useState(0);
  const pendingFetchRef = useRef(false);
  const winRecordedRef = useRef(false);

  // Check session on mount.
  useEffect(() => {
    apiFetch('/api/auth/me')
      .then((r) => r.json())
      .then((d) => setUser(d.user))
      .catch(() => setUser(null))
      .finally(() => setAuthChecked(true));
  }, []);

  // Refresh stats whenever user changes.
  useEffect(() => {
    if (!user) { setStats(null); return; }
    apiFetch('/api/stats/me')
      .then((r) => r.json())
      .then(setStats)
      .catch(() => {});
  }, [user]);

  const handleGenerate = async ({ model, difficulty, seed, repair }) => {
    setIsLoading(true);
    setError(null);
    winRecordedRef.current = false;
    try {
      if (model === 'infinite') {
        const data = await fetchChunk({ model: 'vae', difficulty: 50, seed: null, repair: true });
        setChunks([data.level]);
        setLevel(null);
        setMetrics(data.metrics);
        setParamsUsed({ ...data.params_used, mode: 'infinite' });
        setCurrentModel('infinite');
        setViewMode('play');
      } else {
        const data = await fetchChunk({ model, difficulty, seed, repair });
        setLevel(data.level);
        setChunks(null);
        setMetrics(data.metrics);
        setParamsUsed(data.params_used);
        setCurrentModel(model);
      }
    } catch (err) {
      setError(err.message);
      setLevel(null);
      setChunks(null);
      setMetrics(null);
      setParamsUsed(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChunkNeeded = useCallback(async () => {
    if (pendingFetchRef.current) return;
    pendingFetchRef.current = true;
    try {
      const data = await fetchChunk({ model: 'vae', difficulty: 50, seed: null, repair: true });
      setChunks((prev) => (prev ? [...prev, data.level] : [data.level]));
    } catch (err) {
      setError(err.message);
    } finally {
      pendingFetchRef.current = false;
    }
  }, []);

  const handleInfiniteRestart = useCallback(async () => {
    if (pendingFetchRef.current) return;
    pendingFetchRef.current = true;
    try {
      const data = await fetchChunk({ model: 'vae', difficulty: 50, seed: null, repair: true });
      setChunks([data.level]);
      setMetrics(data.metrics);
      setParamsUsed({ ...data.params_used, mode: 'infinite' });
    } catch (err) {
      setError(err.message);
    } finally {
      pendingFetchRef.current = false;
    }
  }, []);

  // GameCanvas calls this on win in finite mode. Record once per generated level.
  const handleWin = useCallback(async () => {
    if (winRecordedRef.current || !currentModel || currentModel === 'infinite') return;
    winRecordedRef.current = true;
    const updated = await postScore('/api/scores/completion', { model: currentModel });
    if (updated && !updated.__error) setStats(updated);
    else if (updated?.__error) setError(`Score not recorded: ${updated.__error}`);
  }, [currentModel]);

  // GameCanvas calls this with distance (in cols) when player dies in infinite mode.
  const handleDeath = useCallback(async (distanceCols) => {
    if (currentModel !== 'infinite') return;
    const updated = await postScore('/api/scores/endless', { score: distanceCols });
    if (updated && !updated.__error) setStats(updated);
    else if (updated?.__error) setError(`Score not recorded: ${updated.__error}`);
  }, [currentModel]);

  const handleProgress = useCallback((distanceCols) => {
    setEndlessScore(distanceCols);
  }, []);

  const handleLogout = async () => {
    await apiFetch('/api/auth/logout', { method: 'POST' });
    clearToken();
    setUser(null);
    setLevel(null);
    setChunks(null);
    setMetrics(null);
    setParamsUsed(null);
    setCurrentModel(null);
  };

  if (!authChecked) {
    return <div className="auth-page"><div className="auth-card"><p>Loading…</p></div></div>;
  }
  if (!user) {
    return <AuthPage onAuth={setUser} />;
  }

  const inInfiniteMode = chunks !== null;

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-title">
          <h1>TerrainGen</h1>
          <p className="app-tagline">AI-generated platformer levels.</p>
        </div>
        <UserPanel user={user} stats={stats} onLogout={handleLogout} />
      </header>

      <main className="app-main">
        <aside className="sidebar">
          <ControlPanel onGenerate={handleGenerate} isLoading={isLoading} />
        </aside>

        <section className="content">
          {error && <div className="error-banner">{error}</div>}
          <div className="topbar-row">
            <div className="topbar-left">
              {!inInfiniteMode && (
                <div className="view-tabs">
                  <button
                    className={`view-tab ${viewMode === 'play' ? 'active' : ''}`}
                    onClick={() => setViewMode('play')}
                  >
                    Play
                  </button>
                  <button
                    className={`view-tab ${viewMode === 'preview' ? 'active' : ''}`}
                    onClick={() => setViewMode('preview')}
                  >
                    Preview
                  </button>
                </div>
              )}
              {inInfiniteMode && (
                <div className="endless-score">
                  <span className="score-label">Score</span>
                  <strong className="score-value">{endlessScore}</strong>
                  <span className="score-hint">best: {stats?.endless_best ?? 0}</span>
                </div>
              )}
            </div>
            <button className="leaderboard-btn" onClick={() => setShowLeaderboard(true)}>
              🏆 Leaderboard
            </button>
          </div>
          {inInfiniteMode ? (
            <GameCanvas
              chunks={chunks}
              onChunkNeeded={handleChunkNeeded}
              onRestart={handleInfiniteRestart}
              onDeath={handleDeath}
              onProgress={handleProgress}
            />
          ) : viewMode === 'play' ? (
            <GameCanvas level={level} onWin={handleWin} playable={metrics?.playable} />
          ) : (
            <LevelCanvas level={level} />
          )}
        </section>
      </main>

      {showLeaderboard && <Leaderboard onClose={() => setShowLeaderboard(false)} />}
    </div>
  );
}
