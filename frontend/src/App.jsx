import { useState } from 'react';
import ControlPanel from './components/ControlPanel';
import LevelCanvas from './components/LevelCanvas';
import GameCanvas from './components/GameCanvas';
import MetricsPanel from './components/MetricsPanel';
import TileLegend from './components/TileLegend';
import './App.css';

export default function App() {
  const [level, setLevel] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [paramsUsed, setParamsUsed] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('play');

  const handleGenerate = async ({ model, difficulty, seed, repair }) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, difficulty, seed, repair }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Generation failed');
      }
      const data = await res.json();
      setLevel(data.level);
      setMetrics(data.metrics);
      setParamsUsed(data.params_used);
    } catch (err) {
      setError(err.message);
      setLevel(null);
      setMetrics(null);
      setParamsUsed(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Game Level Generator</h1>
        <p>Generate Mario-style platformer levels using ML models trained on SuperTux data</p>
      </header>

      <main className="app-main">
        <aside className="sidebar">
          <ControlPanel onGenerate={handleGenerate} isLoading={isLoading} />
          <TileLegend />
        </aside>

        <section className="content">
          {error && <div className="error-banner">{error}</div>}
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
          {viewMode === 'play' ? (
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
