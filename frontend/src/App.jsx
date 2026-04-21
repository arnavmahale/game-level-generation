import { useState, useRef, useCallback, useEffect } from 'react';
import ControlPanel from './components/ControlPanel';
import LevelCanvas from './components/LevelCanvas';
import GameCanvas from './components/GameCanvas';
import AuthPage from './components/AuthPage';
import UserPanel, { Leaderboard } from './components/UserPanel';
import { apiFetch, clearToken } from './utils/api';
import './App.css';

async function fetchChunk({ model = 'vae', difficulty = 50, seed = null, repair = true, leftContext = null } = {}) {
  const body = { model, difficulty, seed, repair };
  if (leftContext) body.left_context = leftContext;
  const res = await apiFetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || 'Generation failed');
  }
  return res.json();
}

// Extract the rightmost column of a 2D level grid (for endless-mode seam
// stitching — the backend copies this into col 0 of the next chunk).
function rightmostCol(level) {
  return level.map((row) => row[row.length - 1]);
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
      return { __error: data.error || `HTTP ${res.status}` };
    }
    return data;
  } catch (e) {
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
  const [currentModel, setCurrentModel] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('play');
  const [endlessScore, setEndlessScore] = useState(0);
  const [endlessLast, setEndlessLast] = useState(null);
  const pendingFetchRef = useRef(false);
  const winRecordedRef = useRef(false);
  const lastFiniteParamsRef = useRef(null);
  const regenTimerRef = useRef(null);
  // Rightmost column of the most recently added endless chunk; passed to
  // the backend as left_context on the next fetch so seams don't desync.
  const endlessSeamRef = useRef(null);

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
    if (user.isGuest) {
      setStats({
        endless_best: 0,
        completions: { vae: 0, by_difficulty: { easy: 0, medium: 0, hard: 0 } },
      });
      return;
    }
    apiFetch('/api/stats/me')
      .then((r) => r.json())
      .then(setStats)
      .catch(() => {});
  }, [user]);

  const handleGenerate = async ({ model, difficulty, difficultyLabel, seed, repair }) => {
    setIsLoading(true);
    setError(null);
    winRecordedRef.current = false;
    if (regenTimerRef.current) {
      clearTimeout(regenTimerRef.current);
      regenTimerRef.current = null;
    }
    try {
      if (model === 'infinite') {
        const data = await fetchChunk({ model: 'vae', difficulty: 50, seed: null, repair: true });
        setChunks([data.level]);
        endlessSeamRef.current = rightmostCol(data.level);
        setLevel(null);
        setMetrics(data.metrics);
        setCurrentModel('infinite');
        setViewMode('play');
        lastFiniteParamsRef.current = null;
      } else {
        const data = await fetchChunk({ model, difficulty, seed, repair });
        setLevel(data.level);
        setChunks(null);
        setMetrics(data.metrics);
        setCurrentModel(model);
        lastFiniteParamsRef.current = { model, difficulty, difficultyLabel, seed, repair };
      }
    } catch (err) {
      setError(err.message);
      setLevel(null);
      setChunks(null);
      setMetrics(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChunkNeeded = useCallback(async () => {
    if (pendingFetchRef.current) return;
    pendingFetchRef.current = true;
    try {
      const data = await fetchChunk({
        model: 'vae', difficulty: 50, seed: null, repair: true,
        leftContext: endlessSeamRef.current,
      });
      setChunks((prev) => (prev ? [...prev, data.level] : [data.level]));
      endlessSeamRef.current = rightmostCol(data.level);
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
      // Restart = brand-new world, no seam continuity from the previous run.
      endlessSeamRef.current = null;
      const data = await fetchChunk({ model: 'vae', difficulty: 50, seed: null, repair: true });
      setChunks([data.level]);
      endlessSeamRef.current = rightmostCol(data.level);
      setMetrics(data.metrics);
    } catch (err) {
      setError(err.message);
    } finally {
      pendingFetchRef.current = false;
    }
  }, []);

  // GameCanvas calls this on win in finite mode. Record once per generated level,
  // then auto-regenerate a fresh level (same settings) after a short pause.
  const handleWin = useCallback(async () => {
    if (winRecordedRef.current || !currentModel || currentModel === 'infinite') return;
    winRecordedRef.current = true;
    const diffLabel = lastFiniteParamsRef.current?.difficultyLabel ?? null;
    if (user?.isGuest) {
      setStats((s) => {
        const byDiff = { easy: 0, medium: 0, hard: 0, ...(s?.completions?.by_difficulty ?? {}) };
        if (diffLabel && diffLabel in byDiff) byDiff[diffLabel] = (byDiff[diffLabel] ?? 0) + 1;
        return {
          endless_best: s?.endless_best ?? 0,
          completions: {
            vae: (s?.completions?.vae ?? 0) + 1,
            by_difficulty: byDiff,
          },
        };
      });
    } else {
      const updated = await postScore('/api/scores/completion', {
        model: currentModel,
        difficulty: diffLabel,
      });
      if (updated && !updated.__error) setStats(updated);
      else if (updated?.__error) setError(`Score not recorded: ${updated.__error}`);
    }
    const params = lastFiniteParamsRef.current;
    if (params) {
      regenTimerRef.current = setTimeout(() => {
        regenTimerRef.current = null;
        // Fresh seed on auto-regen so the next level isn't a duplicate.
        handleGenerate({ ...params, seed: null });
      }, 2000);
    }
  }, [currentModel, user]);

  // GameCanvas calls this with distance (in cols) when player dies in infinite mode.
  const handleDeath = useCallback(async (distanceCols) => {
    if (currentModel !== 'infinite') return;
    setEndlessLast(distanceCols);
    if (user?.isGuest) {
      setStats((s) => ({
        endless_best: Math.max(s?.endless_best ?? 0, distanceCols),
        completions: { vae: s?.completions?.vae ?? 0 },
      }));
      return;
    }
    const updated = await postScore('/api/scores/endless', { score: distanceCols });
    if (updated && !updated.__error) setStats(updated);
    else if (updated?.__error) setError(`Score not recorded: ${updated.__error}`);
  }, [currentModel, user]);

  const handleProgress = useCallback((distanceCols) => {
    setEndlessScore(distanceCols);
  }, []);

  const handleLogout = async () => {
    if (regenTimerRef.current) {
      clearTimeout(regenTimerRef.current);
      regenTimerRef.current = null;
    }
    lastFiniteParamsRef.current = null;
    if (!user?.isGuest) {
      await apiFetch('/api/auth/logout', { method: 'POST' });
      clearToken();
    }
    setUser(null);
    setLevel(null);
    setChunks(null);
    setMetrics(null);
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
                  <span className="score-hint">
                    best: {stats?.endless_best ?? 0}
                    {endlessLast !== null && <> · last: {endlessLast}</>}
                  </span>
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

      {showLeaderboard && (
        <Leaderboard
          isGuest={!!user?.isGuest}
          currentUsername={user?.isGuest ? null : user?.username}
          onClose={() => setShowLeaderboard(false)}
        />
      )}
    </div>
  );
}
