import { useRef, useEffect } from 'react';
import { TILE_COLORS, GRID_HEIGHT, GRID_WIDTH } from '../utils/colors';

export default function LevelCanvas({ level }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!level || !canvasRef.current || !containerRef.current) return;

    const container = containerRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const containerWidth = container.clientWidth;
    const tileSize = Math.max(10, Math.min(20, Math.floor(containerWidth / GRID_WIDTH)));
    const width = tileSize * GRID_WIDTH;
    const height = tileSize * GRID_HEIGHT;

    canvas.width = width;
    canvas.height = height;

    for (let row = 0; row < GRID_HEIGHT; row++) {
      for (let col = 0; col < GRID_WIDTH; col++) {
        const tileId = level[row][col];
        ctx.fillStyle = TILE_COLORS[tileId] || '#000';
        ctx.fillRect(col * tileSize, row * tileSize, tileSize, tileSize);
        ctx.strokeStyle = 'rgba(0,0,0,0.08)';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(col * tileSize, row * tileSize, tileSize, tileSize);
      }
    }
  }, [level]);

  if (!level) {
    return (
      <div className="level-canvas-container" ref={containerRef}>
        <div className="level-canvas-placeholder">
          Generate a level to see it here
        </div>
      </div>
    );
  }

  return (
    <div className="level-canvas-container" ref={containerRef}>
      <canvas ref={canvasRef} className="level-canvas" />
    </div>
  );
}
