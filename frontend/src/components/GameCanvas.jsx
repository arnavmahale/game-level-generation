import { useRef, useEffect, useState, useCallback } from 'react';
import { GRID_HEIGHT, GRID_WIDTH } from '../utils/colors';

const SOLID_TILES = new Set([1, 2, 3]); // solid, slope, platform
const HAZARD_TILE = 6;

const GRAVITY = 0.6;
const JUMP_FORCE = -13;
const MOVE_SPEED = 3;
const MAX_FALL_SPEED = 12;

// In infinite mode, fetch the next chunk when the player is within this many
// columns of the end of the buffered world.
const CHUNK_FETCH_LOOKAHEAD_COLS = GRID_WIDTH;

export default function GameCanvas({ level, chunks, onChunkNeeded, onRestart, onWin, onDeath, onProgress, playable }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const gameStateRef = useRef(null);
  const keysRef = useRef({});
  const animFrameRef = useRef(null);
  const maxDistColRef = useRef(0);
  const imagesRef = useRef({ loaded: false });
  const [gameStatus, setGameStatus] = useState('waiting');
  const [tileSize, setTileSize] = useState(32);

  const isInfinite = Array.isArray(chunks);
  // In single-level mode, build a one-chunk virtual chunks list so the rest
  // of the code can treat both modes uniformly.
  const chunkList = isInfinite ? chunks : (level ? [level] : null);
  const worldCols = chunkList ? chunkList.length * GRID_WIDTH : 0;

  useEffect(() => {
    const assets = new Image();
    assets.src = '/assets/assets.png';
    const bg = new Image();
    bg.src = '/assets/background.png';
    const char = new Image();
    char.src = '/assets/character.jpg';

    let loaded = 0;
    const onLoad = () => {
      loaded++;
      if (loaded === 3) {
        imagesRef.current = { assets, bg, char, loaded: true };
      }
    };
    assets.onload = onLoad;
    bg.onload = onLoad;
    char.onload = onLoad;
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    const updateSize = () => {
      const w = containerRef.current.clientWidth - 2;
      const computed = Math.floor(w / GRID_WIDTH);
      setTileSize(Math.max(12, Math.min(32, computed)));
    };
    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // Initialize spawn position from the first chunk.
  useEffect(() => {
    if (!chunkList || chunkList.length === 0) return;
    const firstChunk = chunkList[0];

    let spawnRow = 0;
    let spawnCol = 0;
    outer: for (let c = 0; c < 5; c++) {
      for (let r = 0; r < GRID_HEIGHT - 1; r++) {
        if (!SOLID_TILES.has(firstChunk[r][c]) && firstChunk[r][c] !== HAZARD_TILE &&
            SOLID_TILES.has(firstChunk[r + 1][c])) {
          spawnRow = r;
          spawnCol = c;
          break outer;
        }
      }
    }
    if (spawnRow === 0 && spawnCol === 0) spawnRow = 2;

    gameStateRef.current = {
      x: spawnCol * tileSize + tileSize * 0.1,
      y: spawnRow * tileSize,
      vx: 0,
      vy: 0,
      onGround: false,
      width: tileSize * 0.8,
      height: tileSize * 0.9,
      spawnX: spawnCol * tileSize + tileSize * 0.1,
      spawnY: spawnRow * tileSize,
    };
    maxDistColRef.current = 0;
    if (onProgress) onProgress(0);
    setGameStatus('playing');
    // Only reset when the underlying world identity changes, not on every
    // chunk append — otherwise infinite mode would reset on each new chunk.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chunkList && chunkList[0], tileSize]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'Space', 'KeyA', 'KeyD', 'KeyW', 'KeyR'].includes(e.code)) {
        e.preventDefault();
        keysRef.current[e.code] = true;
      }
    };
    const handleKeyUp = (e) => {
      keysRef.current[e.code] = false;
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  // World-coordinate tile lookup that spans all buffered chunks.
  const tileAt = useCallback((worldCol, row) => {
    if (!chunkList) return 0;
    if (row < 0 || row >= GRID_HEIGHT) return 0;
    if (worldCol < 0 || worldCol >= worldCols) return 0;
    const chunkIdx = Math.floor(worldCol / GRID_WIDTH);
    const col = worldCol - chunkIdx * GRID_WIDTH;
    return chunkList[chunkIdx][row][col];
  }, [chunkList, worldCols]);

  const isSolid = useCallback((worldCol, row) => {
    if (row >= GRID_HEIGHT) return true; // bottom of world
    if (worldCol < 0) return true;       // left wall only
    if (!isInfinite && worldCol >= GRID_WIDTH) return false;
    return SOLID_TILES.has(tileAt(worldCol, row));
  }, [tileAt, isInfinite]);

  const isHazard = useCallback((worldCol, row) => {
    return tileAt(worldCol, row) === HAZARD_TILE;
  }, [tileAt]);

  useEffect(() => {
    if (!chunkList || !canvasRef.current) return;
    if (gameStatus !== 'playing' && gameStatus !== 'won' && gameStatus !== 'dead') return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const ts = tileSize;

    canvas.width = ts * GRID_WIDTH;
    canvas.height = ts * GRID_HEIGHT;

    const gameLoop = () => {
      const player = gameStateRef.current;
      if (!player) return;

      const keys = keysRef.current;

      if (keys['KeyR']) {
        player.x = player.spawnX;
        player.y = player.spawnY;
        player.vx = 0;
        player.vy = 0;
        setGameStatus('playing');
      }

      // Freeze physics while waiting for a new infinite-mode chunk to arrive,
      // so input doesn't drift the player off-world during the fetch.
      const isFrozen = gameStatus === 'won' || (isInfinite && gameStatus === 'dead');

      player.vx = 0;
      if (!isFrozen) {
        if (keys['ArrowLeft'] || keys['KeyA']) player.vx = -MOVE_SPEED;
        if (keys['ArrowRight'] || keys['KeyD']) player.vx = MOVE_SPEED;
      }

      if (!isFrozen && (keys['ArrowUp'] || keys['Space'] || keys['KeyW']) && player.onGround) {
        player.vy = JUMP_FORCE;
        player.onGround = false;
      }

      if (!isFrozen) {
        player.vy += GRAVITY;
        if (player.vy > MAX_FALL_SPEED) player.vy = MAX_FALL_SPEED;
      } else {
        player.vy = 0;
      }

      player.x += player.vx;
      const pw = player.width;
      const ph = player.height;

      if (player.x < 0) player.x = 0;
      const worldPxWidth = worldCols * ts;
      if (!isInfinite && player.x + pw > worldPxWidth) player.x = worldPxWidth - pw;

      if (player.vx > 0) {
        const rightCol = Math.floor((player.x + pw) / ts);
        const topRow = Math.floor(player.y / ts);
        const botRow = Math.floor((player.y + ph - 1) / ts);
        for (let r = topRow; r <= botRow; r++) {
          if (isSolid(rightCol, r)) {
            player.x = rightCol * ts - pw;
            player.vx = 0;
            break;
          }
        }
      } else if (player.vx < 0) {
        const leftCol = Math.floor(player.x / ts);
        const topRow = Math.floor(player.y / ts);
        const botRow = Math.floor((player.y + ph - 1) / ts);
        for (let r = topRow; r <= botRow; r++) {
          if (isSolid(leftCol, r)) {
            player.x = (leftCol + 1) * ts;
            player.vx = 0;
            break;
          }
        }
      }

      player.y += player.vy;
      player.onGround = false;

      if (player.vy > 0) {
        const leftCol = Math.floor(player.x / ts);
        const rightCol = Math.floor((player.x + pw - 1) / ts);
        const botRow = Math.floor((player.y + ph) / ts);
        for (let c = leftCol; c <= rightCol; c++) {
          if (isSolid(c, botRow)) {
            player.y = botRow * ts - ph;
            player.vy = 0;
            player.onGround = true;
            break;
          }
        }
      } else if (player.vy < 0) {
        const leftCol = Math.floor(player.x / ts);
        const rightCol = Math.floor((player.x + pw - 1) / ts);
        const topRow = Math.floor(player.y / ts);
        for (let c = leftCol; c <= rightCol; c++) {
          if (isSolid(c, topRow)) {
            player.y = (topRow + 1) * ts;
            player.vy = 0;
            break;
          }
        }
      }

      const pLeft = Math.floor(player.x / ts);
      const pRight = Math.floor((player.x + pw - 1) / ts);
      const pTop = Math.floor(player.y / ts);
      const pBot = Math.floor((player.y + ph - 1) / ts);
      let died = false;
      for (let r = pTop; r <= pBot && !died; r++) {
        for (let c = pLeft; c <= pRight; c++) {
          if (isHazard(c, r)) {
            died = true;
            break;
          }
        }
      }
      if (!died && player.y > GRID_HEIGHT * ts + 50) {
        died = true;
      }
      if (died && gameStatus !== 'dead') {
        player.vx = 0;
        player.vy = 0;
        setGameStatus('dead');
        const distanceCols = Math.max(0, Math.floor(player.x / ts));
        if (onDeath) onDeath(distanceCols);
        if (isInfinite && onRestart) {
          // Replaces the buffered chunks; spawn effect re-runs once the new
          // first chunk arrives and flips status back to 'playing'.
          onRestart();
        } else {
          player.x = player.spawnX;
          player.y = player.spawnY;
          setTimeout(() => setGameStatus('playing'), 500);
        }
      }

      // Win only applies to finite mode.
      if (!isInfinite && !isFrozen && player.x + pw >= (GRID_WIDTH - 1) * ts) {
        setGameStatus('won');
        if (onWin) onWin();
      }

      // Infinite mode: trigger a fetch when entering the final chunk's last viewport.
      // Also track max distance reached for the live score HUD.
      if (isInfinite) {
        const playerCol = Math.max(0, Math.floor(player.x / ts));
        if (playerCol > maxDistColRef.current) {
          maxDistColRef.current = playerCol;
          if (onProgress) onProgress(playerCol);
        }
        if (onChunkNeeded) {
          const needFetchAt = worldCols - CHUNK_FETCH_LOOKAHEAD_COLS;
          if (playerCol >= needFetchAt) {
            onChunkNeeded();
          }
        }
      }

      // ===== CAMERA =====
      // Keep the player near the left-third of the viewport in infinite mode.
      let camX = 0;
      if (isInfinite) {
        const viewportW = canvas.width;
        camX = Math.max(0, player.x - viewportW * 0.35);
        const maxCam = Math.max(0, worldCols * ts - viewportW);
        camX = Math.min(camX, maxCam);
      }

      // ===== RENDER =====
      const imgs = imagesRef.current;

      if (imgs.loaded) {
        const bgRatio = imgs.bg.height / (GRID_HEIGHT * ts);
        const bgDrawWidth = imgs.bg.width / bgRatio;
        const parallaxX = -(player.x / (GRID_WIDTH * ts)) * (bgDrawWidth - canvas.width) * 0.5;
        ctx.drawImage(imgs.bg, parallaxX, 0, bgDrawWidth, GRID_HEIGHT * ts);
        if (parallaxX + bgDrawWidth < canvas.width) {
          ctx.drawImage(imgs.bg, parallaxX + bgDrawWidth, 0, bgDrawWidth, GRID_HEIGHT * ts);
        }
      } else {
        ctx.fillStyle = '#87CEEB';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }

      // Draw only tiles inside the viewport.
      const firstCol = Math.max(0, Math.floor(camX / ts));
      const lastCol = Math.min(worldCols - 1, Math.ceil((camX + canvas.width) / ts));
      for (let row = 0; row < GRID_HEIGHT; row++) {
        for (let col = firstCol; col <= lastCol; col++) {
          const tile = tileAt(col, row);
          if (tile === 0) continue;

          const x = col * ts - camX;
          const y = row * ts;

          if (imgs.loaded) {
            if (SOLID_TILES.has(tile)) {
              ctx.drawImage(imgs.assets, 0, 0, 100, 100, x, y, ts, ts);
            } else if (tile === HAZARD_TILE) {
              ctx.drawImage(imgs.assets, 100, 0, 100, 100, x, y, ts, ts);
            } else if (tile === 4) {
              ctx.fillStyle = '#FFD700';
              ctx.fillRect(x, y, ts, ts);
              ctx.fillStyle = '#FFA500';
              ctx.beginPath();
              ctx.arc(x + ts / 2, y + ts / 2, ts * 0.3, 0, Math.PI * 2);
              ctx.fill();
            } else if (tile === 5) {
              ctx.fillStyle = 'rgba(30, 100, 200, 0.7)';
              ctx.fillRect(x, y, ts, ts);
              ctx.fillStyle = 'rgba(100, 180, 255, 0.4)';
              ctx.fillRect(x, y, ts, ts * 0.3);
            } else if (tile === 7) {
              ctx.fillStyle = '#2d8a4e';
              ctx.fillRect(x, y, ts, ts);
              ctx.fillStyle = '#3cb371';
              ctx.fillRect(x + 2, y + 2, ts - 4, ts - 4);
            }
          } else {
            const colors = {
              1: '#8B4513', 2: '#D2B48C', 3: '#DEB887',
              4: '#FFD700', 5: '#4169E1', 6: '#DC143C', 7: '#228B22',
            };
            ctx.fillStyle = colors[tile] || '#888';
            ctx.fillRect(x, y, ts, ts);
          }
        }
      }

      // Finish line only in finite mode.
      if (!isInfinite) {
        const finishX = (GRID_WIDTH - 1) * ts - camX;
        const checkSize = ts / 2;
        for (let r = 0; r < GRID_HEIGHT * 2; r++) {
          for (let c = 0; c < 2; c++) {
            ctx.fillStyle = (r + c) % 2 === 0 ? '#ffffff' : '#000000';
            ctx.fillRect(finishX + c * checkSize, r * checkSize, checkSize, checkSize);
          }
        }
      }

      if (imgs.loaded) {
        ctx.drawImage(imgs.char, player.x - camX, player.y, pw, ph);
      } else {
        ctx.fillStyle = '#ff6600';
        ctx.fillRect(player.x - camX, player.y, pw, ph);
      }

      if (gameStatus === 'won') {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#ffffff';
        ctx.font = `bold ${ts * 3}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('You Win!', canvas.width / 2, canvas.height / 2);
      }

      if (gameStatus === 'dead') {
        ctx.fillStyle = 'rgba(200, 0, 0, 0.3)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }

      animFrameRef.current = requestAnimationFrame(gameLoop);
    };

    animFrameRef.current = requestAnimationFrame(gameLoop);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [chunkList, gameStatus, tileSize, isSolid, isHazard, tileAt, worldCols, isInfinite, onChunkNeeded, onRestart, onWin, onDeath, onProgress]);

  if (!chunkList || chunkList.length === 0) {
    return (
      <div className="game-container" ref={containerRef}>
        <div className="game-placeholder">
          Generate a level to play
        </div>
      </div>
    );
  }

  return (
    <div className="game-container" ref={containerRef}>
      <div className="game-stack">
        <canvas
          ref={canvasRef}
          className="game-canvas"
          tabIndex={0}
        />
        <div className="game-controls-hint">
          {!isInfinite && playable !== undefined && (
            <span className={`playable-badge playable-badge-left ${playable ? 'good' : 'warn'}`}>
              {playable ? 'Playable' : 'Not playable'}
            </span>
          )}
          <span><kbd>A</kbd>/<kbd>D</kbd> Move</span>
          <span><kbd>W</kbd>/<kbd>Space</kbd> Jump</span>
          <span><kbd>R</kbd> Restart</span>
        </div>
      </div>
    </div>
  );
}
