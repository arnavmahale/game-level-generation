export default function MetricsPanel({ metrics }) {
  if (!metrics) return null;
  const playable = metrics.playable;
  return (
    <div className="metrics-panel">
      <div className="metrics-grid">
        <div className="metric-item">
          <div className="metric-header">
            <span className="metric-label">Playable</span>
            <span className={`metric-value ${playable ? 'good' : 'warn'}`}>
              {playable ? 'Yes' : 'No'}
            </span>
          </div>
          <span className="metric-desc">Can a player traverse left to right?</span>
        </div>
      </div>
    </div>
  );
}
