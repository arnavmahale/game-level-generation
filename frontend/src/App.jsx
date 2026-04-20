import { useState, useRef, useCallback } from 'react';
import ControlPanel from './components/ControlPanel';
import LevelCanvas from './components/LevelCanvas';
import GameCanvas from './components/GameCanvas';
import MetricsPanel from './components/MetricsPanel';
import TileLegend from './components/TileLegend';
import './App.css';

async function fetchChunk({ model = 'vae', difficulty = 50, seed = null, repair = true } = {}) {
  const res = await fetch('/api/generate', {
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

export default function App() {
  const [level, setLevel] = useState(null);
  const [chunks, setChunks] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [paramsUsed, setParamsUsed] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('play');
  const pendingFetchRef = useRef(false);

  const handleGenerate = async ({ model, difficulty, seed, repair }) => {
    setIsLoading(true);
    setError(null);
    try {
      if (model === 'infinite') {
        const data = await fetchChunk({ model: 'vae', difficulty: 50, seed: null, repair: true });
        setChunks([data.level]);
        setLevel(null);
        setMetrics(data.metrics);
        setParamsUsed({ ...data.params_used, mode: 'infinite' });
        setViewMode('play');
      } else {
        const data = await fetchChunk({ model, difficulty, seed, repair });
        setLevel(data.level);
        setChunks(null);
        setMetrics(data.metrics);
        setParamsUsed(data.params_used);
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

  // Called by GameCanvas when the player nears the right edge of the last chunk.
  // Idempotent via pendingFetchRef so a slow network doesn't trigger duplicates.
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

  // Called by GameCanvas when the player dies in infinite mode.
  // Replaces the buffered chunks with a freshly generated one so every death
  // leads to a never-before-seen level.
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

  const inInfiniteMode = chunks !== null;

  return (
    <div className="app">
      <header className="app-header">
        <h1>GenTerrain</h1>
        <p>Generate Mario-style platformer levels using ML models trained on SuperTux data</p>
      </header>

      <main className="app-main">
        <aside className="sidebar">
          <ControlPanel onGenerate={handleGenerate} isLoading={isLoading} />
          <TileLegend />
        </aside>

        <section className="content">
          {error && <div className="error-banner">{error}</div>}
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
          {inInfiniteMode ? (
            <GameCanvas
              chunks={chunks}
              onChunkNeeded={handleChunkNeeded}
              onRestart={handleInfiniteRestart}
            />
          ) : viewMode === 'play' ? (
            <GameCanvas level={level} />
          ) : (
            <LevelCanvas level={level} />
          )}
          <MetricsPanel metrics={metrics} paramsUsed={paramsUsed} />
        </section>
      </main>
    </div>
  );
}
