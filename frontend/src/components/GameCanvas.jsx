import { useRef, useEffect, useState, useCallback } from 'react';
import { GRID_HEIGHT, GRID_WIDTH } from '../utils/colors';

// Tile categories that are solid (walkable)
const SOLID_TILES = new Set([1, 2, 3]); // solid, slope, platform
const HAZARD_TILE = 6;

// Player physics constants
const GRAVITY = 0.6;
const JUMP_FORCE = -13;
const MOVE_SPEED = 2;
const MAX_FALL_SPEED = 12;

export default function GameCanvas({ level, onWin, onDeath }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const gameStateRef = useRef(null);
  const keysRef = useRef({});
  const animFrameRef = useRef(null);
  const imagesRef = useRef({ loaded: false });
  const [gameStatus, setGameStatus] = useState('waiting'); // waiting, playing, won, dead
  const [tileSize, setTileSize] = useState(32);

  // Load sprite images
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

  // Calculate tile size based on container
  useEffect(() => {
    if (!containerRef.current) return;
    const updateSize = () => {
      const w = containerRef.current.clientWidth - 2; // account for border
      const computed = Math.floor(w / GRID_WIDTH);
      setTileSize(Math.max(12, Math.min(32, computed)));
    };
    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // Initialize game state when level changes
  useEffect(() => {
    if (!level) return;

    // Find a valid spawn position (leftmost column, find a spot to stand)
    let spawnRow = 0;
    for (let r = 0; r < GRID_HEIGHT - 1; r++) {
      if (!SOLID_TILES.has(level[r][0]) && level[r][0] !== HAZARD_TILE &&
          SOLID_TILES.has(level[r + 1][0])) {
        spawnRow = r;
        break;
      }
    }
    // If no valid spawn on col 0, try col 1, 2, etc.
    let spawnCol = 0;
    if (spawnRow === 0) {
      for (let c = 0; c < 5; c++) {
        for (let r = 0; r < GRID_HEIGHT - 1; r++) {
          if (!SOLID_TILES.has(level[r][c]) && level[r][c] !== HAZARD_TILE &&
              SOLID_TILES.has(level[r + 1][c])) {
            spawnRow = r;
            spawnCol = c;
            break;
          }
        }
        if (spawnRow > 0) break;
      }
    }
    // Fallback: spawn at top-left
    if (spawnRow === 0) spawnRow = 2;

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
    setGameStatus('playing');
  }, [level, tileSize]);

  // Keyboard input
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

  // Check if a point collides with a solid tile
  const isSolid = useCallback((gridCol, gridRow) => {
    if (gridRow < 0 || gridRow >= GRID_HEIGHT || gridCol < 0 || gridCol >= GRID_WIDTH) {
      return gridRow >= GRID_HEIGHT; // bottom of world is solid
    }
    return SOLID_TILES.has(level[gridRow][gridCol]);
  }, [level]);

  const isHazard = useCallback((gridCol, gridRow) => {
    if (gridRow < 0 || gridRow >= GRID_HEIGHT || gridCol < 0 || gridCol >= GRID_WIDTH) return false;
    return level[gridRow][gridCol] === HAZARD_TILE;
  }, [level]);

  // Game loop
  useEffect(() => {
    if (!level || !canvasRef.current) return;
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

      // Handle restart
      if (keys['KeyR']) {
        player.x = player.spawnX;
        player.y = player.spawnY;
        player.vx = 0;
        player.vy = 0;
        setGameStatus('playing');
      }

      const isFrozen = gameStatus === 'won';

      // Horizontal movement
      player.vx = 0;
      if (!isFrozen) {
        if (keys['ArrowLeft'] || keys['KeyA']) player.vx = -MOVE_SPEED;
        if (keys['ArrowRight'] || keys['KeyD']) player.vx = MOVE_SPEED;
      }

      // Jump
      if (!isFrozen && (keys['ArrowUp'] || keys['Space'] || keys['KeyW']) && player.onGround) {
        player.vy = JUMP_FORCE;
        player.onGround = false;
      }

      // Gravity
      if (!isFrozen) {
        player.vy += GRAVITY;
        if (player.vy > MAX_FALL_SPEED) player.vy = MAX_FALL_SPEED;
      } else {
        player.vy = 0;
      }

      // Move horizontally with collision
      player.x += player.vx;
      const pw = player.width;
      const ph = player.height;

      // Clamp to world bounds
      if (player.x < 0) player.x = 0;
      if (player.x + pw > GRID_WIDTH * ts) player.x = GRID_WIDTH * ts - pw;

      // Horizontal collision
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

      // Move vertically with collision
      player.y += player.vy;
      player.onGround = false;

      if (player.vy > 0) {
        // Falling down
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
        // Moving up (jumping)
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

      // Check hazard collision
      const pLeft = Math.floor(player.x / ts);
      const pRight = Math.floor((player.x + pw - 1) / ts);
      const pTop = Math.floor(player.y / ts);
      const pBot = Math.floor((player.y + ph - 1) / ts);
      for (let r = pTop; r <= pBot; r++) {
        for (let c = pLeft; c <= pRight; c++) {
          if (isHazard(c, r)) {
            // Death — respawn
            player.x = player.spawnX;
            player.y = player.spawnY;
            player.vx = 0;
            player.vy = 0;
            setGameStatus('dead');
            if (onDeath) onDeath();
            setTimeout(() => setGameStatus('playing'), 500);
            break;
          }
        }
      }

      // Fell off bottom
      if (player.y > GRID_HEIGHT * ts + 50) {
        player.x = player.spawnX;
        player.y = player.spawnY;
        player.vx = 0;
        player.vy = 0;
        setGameStatus('dead');
        if (onDeath) onDeath();
        setTimeout(() => setGameStatus('playing'), 500);
      }

      // Win condition — reach rightmost area
      if (!isFrozen && player.x + pw >= (GRID_WIDTH - 1) * ts) {
        setGameStatus('won');
        if (onWin) onWin();
      }

      // ===== RENDER =====
      const imgs = imagesRef.current;

      // Background
      if (imgs.loaded) {
        // Draw background scaled to canvas, tiled if needed
        const bgRatio = imgs.bg.height / (GRID_HEIGHT * ts);
        const bgDrawWidth = imgs.bg.width / bgRatio;
        // Parallax: slight shift based on player position
        const parallaxX = -(player.x / (GRID_WIDTH * ts)) * (bgDrawWidth - canvas.width) * 0.5;
        ctx.drawImage(imgs.bg, parallaxX, 0, bgDrawWidth, GRID_HEIGHT * ts);
        // Fill remaining width if needed
        if (parallaxX + bgDrawWidth < canvas.width) {
          ctx.drawImage(imgs.bg, parallaxX + bgDrawWidth, 0, bgDrawWidth, GRID_HEIGHT * ts);
        }
      } else {
        ctx.fillStyle = '#87CEEB';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }

      // Draw tiles
      for (let row = 0; row < GRID_HEIGHT; row++) {
        for (let col = 0; col < GRID_WIDTH; col++) {
          const tile = level[row][col];
          if (tile === 0) continue; // air — show background

          const x = col * ts;
          const y = row * ts;

          if (imgs.loaded) {
            if (SOLID_TILES.has(tile)) {
              // Dirt block — left half of assets.png (0,0 to 100,100)
              ctx.drawImage(imgs.assets, 0, 0, 100, 100, x, y, ts, ts);
            } else if (tile === HAZARD_TILE) {
              // Lava — right half of assets.png (100,0 to 200,100)
              ctx.drawImage(imgs.assets, 100, 0, 100, 100, x, y, ts, ts);
            } else if (tile === 4) {
              // Bonus — gold coin style
              ctx.fillStyle = '#FFD700';
              ctx.fillRect(x, y, ts, ts);
              ctx.fillStyle = '#FFA500';
              ctx.beginPath();
              ctx.arc(x + ts / 2, y + ts / 2, ts * 0.3, 0, Math.PI * 2);
              ctx.fill();
            } else if (tile === 5) {
              // Water
              ctx.fillStyle = 'rgba(30, 100, 200, 0.7)';
              ctx.fillRect(x, y, ts, ts);
              // Wave effect
              ctx.fillStyle = 'rgba(100, 180, 255, 0.4)';
              ctx.fillRect(x, y, ts, ts * 0.3);
            } else if (tile === 7) {
              // Decoration — greenery
              ctx.fillStyle = '#2d8a4e';
              ctx.fillRect(x, y, ts, ts);
              ctx.fillStyle = '#3cb371';
              ctx.fillRect(x + 2, y + 2, ts - 4, ts - 4);
            }
          } else {
            // Fallback colored squares
            const colors = {
              1: '#8B4513', 2: '#D2B48C', 3: '#DEB887',
              4: '#FFD700', 5: '#4169E1', 6: '#DC143C', 7: '#228B22',
            };
            ctx.fillStyle = colors[tile] || '#888';
            ctx.fillRect(x, y, ts, ts);
          }
        }
      }

      // Draw finish line at rightmost column — pure black/white checkerboard
      const finishX = (GRID_WIDTH - 1) * ts;
      const checkSize = ts / 2;
      for (let r = 0; r < GRID_HEIGHT * 2; r++) {
        for (let c = 0; c < 2; c++) {
          ctx.fillStyle = (r + c) % 2 === 0 ? '#ffffff' : '#000000';
          ctx.fillRect(finishX + c * checkSize, r * checkSize, checkSize, checkSize);
        }
      }

      // Draw player
      if (imgs.loaded) {
        ctx.drawImage(imgs.char, player.x, player.y, pw, ph);
      } else {
        ctx.fillStyle = '#ff6600';
        ctx.fillRect(player.x, player.y, pw, ph);
      }

      // Game status overlay
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
  }, [level, gameStatus, tileSize, isSolid, isHazard, onWin, onDeath]);

  // Render initial frame when level loads (for won state too)
  useEffect(() => {
    if (gameStatus === 'won' && canvasRef.current) {
      // Keep the last frame visible
    }
  }, [gameStatus]);

  if (!level) {
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
      <canvas
        ref={canvasRef}
        className="game-canvas"
        tabIndex={0}
      />
      <div className="game-controls-hint">
        <span><kbd>A</kbd>/<kbd>D</kbd> Move</span>
        <span><kbd>W</kbd>/<kbd>Space</kbd> Jump</span>
        <span><kbd>R</kbd> Restart</span>
        <span>Reach the right side to win</span>
      </div>
    </div>
  );
}
