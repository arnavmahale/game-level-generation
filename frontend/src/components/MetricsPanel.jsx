export default function MetricsPanel({ metrics, paramsUsed }) {
  if (!metrics) return null;

  const metricItems = [
    {
      label: 'Playable',
      value: metrics.playable ? 'Yes' : 'No',
      good: metrics.playable,
      desc: 'Can a player traverse left to right?',
    },
    {
      label: 'JS Divergence',
      value: metrics.js_divergence.toFixed(4),
      good: metrics.js_divergence < 0.05,
      desc: 'Tile distribution similarity (lower = better)',
    },
    {
      label: 'Ground Coverage',
      value: (metrics.ground_coverage * 100).toFixed(1) + '%',
      good: metrics.ground_coverage > 0.15 && metrics.ground_coverage < 0.5,
      desc: 'Fraction of walkable tiles',
    },
    {
      label: 'Ground Contiguity',
      value: (metrics.ground_contiguity * 100).toFixed(1) + '%',
      good: metrics.ground_contiguity > 0.5,
      desc: 'Ground tiles adjacent to other ground',
    },
    {
      label: 'Hazard Density',
      value: (metrics.hazard_density * 100).toFixed(2) + '%',
      good: true,
      desc: 'Fraction of hazard tiles',
    },
  ];

  return (
    <div className="metrics-panel">
      <h3>Evaluation Metrics</h3>
      <div className="metrics-grid">
        {metricItems.map((m) => (
          <div key={m.label} className="metric-item">
            <div className="metric-header">
              <span className="metric-label">{m.label}</span>
              <span className={`metric-value ${m.good ? 'good' : 'warn'}`}>
                {m.value}
              </span>
            </div>
            <span className="metric-desc">{m.desc}</span>
          </div>
        ))}
      </div>
      {paramsUsed && (
        <div className="params-used">
          <h4>Parameters Used</h4>
          <div className="params-list">
            {Object.entries(paramsUsed).map(([key, val]) => (
              <span key={key} className="param-chip">
                {key}: {val}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
