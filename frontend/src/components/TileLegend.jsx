import { TILE_COLORS, CATEGORY_NAMES } from '../utils/colors';

export default function TileLegend() {
  return (
    <div className="tile-legend">
      <h3>Tile Legend</h3>
      <div className="legend-items">
        {Object.entries(CATEGORY_NAMES).map(([id, name]) => (
          <div key={id} className="legend-item">
            <div
              className="legend-swatch"
              style={{ backgroundColor: TILE_COLORS[id] }}
            />
            <span className="legend-label">{name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
