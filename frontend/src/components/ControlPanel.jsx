import { useState } from 'react';

const MODELS = [
  {
    id: 'naive',
    name: 'Naive Baseline',
    desc: 'Random ground fill with learned height',
    paramLabel: 'height_std',
  },
  {
    id: 'bigram',
    name: 'Bigram Transitions',
    desc: 'Learned transition probabilities',
    paramLabel: 'prior_weight / trans_weight',
  },
  {
    id: 'vae',
    name: 'Convolutional VAE',
    desc: 'Deep generative model (64-dim latent)',
    paramLabel: 'temperature',
  },
  {
    id: 'infinite',
    name: 'Endless Mode',
    desc: 'VAE medium, streamed chunk by chunk',
    paramLabel: 'bucket=medium',
  },
];

// Perceived difficulty didn't match the bucket labels the VAE was trained on
// (bucket 0 plays hardest, 1 easiest, 2 medium). Remap button → difficulty
// value so the UI reads correctly. Endless mode still hard-codes 50.
const DIFFICULTY_LEVELS = [
  { id: 'easy', label: 'Easy', value: 50 },
  { id: 'medium', label: 'Medium', value: 100 },
  { id: 'hard', label: 'Hard', value: 0 },
];

export default function ControlPanel({ onGenerate, isLoading }) {
  const [model, setModel] = useState('vae');
  const [difficulty, setDifficulty] = useState(100);
  const [seed, setSeed] = useState('');
  const [repair, setRepair] = useState(true);

  const handleGenerate = () => {
    onGenerate({
      model,
      difficulty,
      seed: seed === '' ? null : parseInt(seed, 10),
      repair,
    });
  };

  return (
    <div className="control-panel">
      <div className="control-section">
        <h3>Model</h3>
        <div className="model-options">
          {MODELS.map((m) => (
            <label
              key={m.id}
              className={`model-option ${model === m.id ? 'selected' : ''}`}
            >
              <input
                type="radio"
                name="model"
                value={m.id}
                checked={model === m.id}
                onChange={(e) => setModel(e.target.value)}
              />
              <div className="model-option-content">
                <span className="model-name">{m.name}</span>
              </div>
            </label>
          ))}
        </div>
      </div>

      {model !== 'infinite' && (
        <>
          <div className="control-section">
            <h3>Difficulty</h3>
            <div className="difficulty-options">
              {DIFFICULTY_LEVELS.map((d) => (
                <button
                  key={d.id}
                  type="button"
                  className={`difficulty-option ${difficulty === d.value ? 'selected' : ''}`}
                  onClick={() => setDifficulty(d.value)}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </div>

          <div className="control-section">
            <h3>Seed <span className="optional">(optional)</span></h3>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="Random"
              className="seed-input"
            />
          </div>

          <div className="control-section">
            <label className="repair-toggle">
              <input
                type="checkbox"
                checked={repair}
                onChange={(e) => setRepair(e.target.checked)}
              />
              <span>Apply BFS playability repair</span>
            </label>
          </div>
        </>
      )}

      <button
        className="generate-btn"
        onClick={handleGenerate}
        disabled={isLoading}
      >
        {isLoading ? (
          <span className="loading-spinner">
            {model === 'infinite' ? 'Starting…' : 'Generating...'}
          </span>
        ) : (
          model === 'infinite' ? 'Start Endless Mode' : 'Generate Level'
        )}
      </button>
    </div>
  );
}
