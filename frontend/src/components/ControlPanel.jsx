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
];

export default function ControlPanel({ onGenerate, isLoading }) {
  const [model, setModel] = useState('vae');
  const [difficulty, setDifficulty] = useState(50);
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

  const getDifficultyLabel = () => {
    if (difficulty <= 20) return 'Very Easy';
    if (difficulty <= 40) return 'Easy';
    if (difficulty <= 60) return 'Medium';
    if (difficulty <= 80) return 'Hard';
    return 'Very Hard';
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

      <div className="control-section">
        <h3>Difficulty</h3>
        <div className="difficulty-slider-container">
          <input
            type="range"
            min="0"
            max="100"
            value={difficulty}
            onChange={(e) => setDifficulty(parseInt(e.target.value, 10))}
            className="difficulty-slider"
          />
          <div className="difficulty-labels">
            <span>Easy</span>
            <span className="difficulty-value">
              {difficulty} — {getDifficultyLabel()}
            </span>
            <span>Hard</span>
          </div>
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

      <button
        className="generate-btn"
        onClick={handleGenerate}
        disabled={isLoading}
      >
        {isLoading ? (
          <span className="loading-spinner">Generating...</span>
        ) : (
          'Generate Level'
        )}
      </button>
    </div>
  );
}
