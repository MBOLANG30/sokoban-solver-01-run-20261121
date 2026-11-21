// Sokoban solver: push-optimal per room, stitched through the hallway.
//
// A level is a hallway (the row the player starts on) with one to four rooms
// hanging off it, each holding 3 or 4 boxes and as many goals. Boxes never
// leave their room in an optimal solution, and the level's push-optimal count
// is the sum of the rooms' optima. So: split the level into rooms, solve each
// room push-optimally with A* over pushes (player reachability folded into
// each macro state, minimum-matching heuristic, corner / wall-run / 2x2
// deadlock pruning), then walk the player between rooms and concatenate.
//
// Deterministic by construction: fixed neighbour order, heap ties broken by
// insertion order, work budgets counted in expansions, never the clock.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

type Dir = { dx: number; dy: number; ch: string };
const DIRS: Dir[] = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

// How much search one room may spend before giving up (exact, then weighted).
const EXPANSION_CAP = 400_000;

type Level = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    const row = rows[y]!;
    for (let x = 0; x < w; x++) {
      const ch = x < row.length ? row[x]! : '#';
      const i = y * w + x;
      wall[i] = ch === '#' ? 1 : 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Shortest walk (player steps only, boxes are obstacles) from `from` to `to`.
// Fixed neighbour order, so the path is deterministic. Null when unreachable.
function walkPath(
  wall: Uint8Array,
  w: number,
  h: number,
  blocked: Uint8Array,
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const n = w * h;
  const prev = new Int32Array(n).fill(-1);
  const prevDir = new Int8Array(n).fill(-1);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi]!;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const dir = DIRS[d]!;
      const nx = cx + dir.dx;
      const ny = cy + dir.dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const nc = ny * w + nx;
      if (wall[nc] || blocked[nc] || prev[nc] !== -1) continue;
      prev[nc] = c;
      prevDir[nc] = d;
      if (nc === to) {
        let s = '';
        let cur = nc;
        while (cur !== from) {
          s = DIRS[prevDir[cur]!]!.ch + s;
          cur = prev[cur]!;
        }
        return s;
      }
      queue.push(nc);
    }
  }
  return null;
}

type Room = {
  cells: number[];
  boxes: number[];
  goals: number[];
  door: number;
  entry: number;
};

// Split the level into rooms: the hallway is the row the player starts on;
// every other floor component is a room, entered through the one cell where
// it touches the hallway row. Null when the shape is not hallway-plus-rooms
// (a box in the hallway, a room with no door, ...): the caller then treats
// the whole level as one room.
function decompose(level: Level): Room[] | null {
  const { w, h, wall, goals, boxes, player } = level;
  const hallRow = (player / w) | 0;
  const goalSet = new Set(goals);
  const boxSet = new Set(boxes);
  // A box sitting in the hallway row cannot belong to any room.
  for (const b of boxes) if (((b / w) | 0) === hallRow) return null;
  const seen = new Uint8Array(w * h);
  const rooms: Room[] = [];
  const assigned = new Set<number>();
  for (let i = 0; i < w * h; i++) {
    if (wall[i] || ((i / w) | 0) === hallRow || seen[i]) continue;
    const comp: number[] = [];
    const compSet = new Set<number>();
    const queue = [i];
    seen[i] = 1;
    for (let qi = 0; qi < queue.length; qi++) {
      const c = queue[qi]!;
      comp.push(c);
      compSet.add(c);
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const nc = ny * w + nx;
        if (wall[nc] || ((nc / w) | 0) === hallRow || seen[nc]) continue;
        seen[nc] = 1;
        queue.push(nc);
      }
    }
    const rBoxes = boxes.filter((b) => compSet.has(b));
    if (rBoxes.length === 0) continue;
    const rGoals = goals.filter((g) => compSet.has(g));
    if (rGoals.length !== rBoxes.length) return null;
    // The door: the component cell touching a hallway-row floor cell.
    let door = -1;
    let entry = -1;
    for (const c of comp) {
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        if (ny !== hallRow) continue;
        const nc = ny * w + nx;
        if (wall[nc]) continue;
        door = c;
        entry = nc;
        break;
      }
      if (door >= 0) break;
    }
    if (door < 0) return null;
    for (const b of rBoxes) assigned.add(b);
    rooms.push({ cells: comp, boxes: rBoxes, goals: rGoals, door, entry });
  }
  if (assigned.size !== boxes.length) return null;
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

type RoomSolution = { moves: string; endPlayer: number; endBoxes: number[] };

// Push-optimal A* for one room (or, as a fallback, a whole level).
// walkCells: every cell the player may stand on (room cells plus the hallway
// cell outside its door). boxes/goals: the room's, equal counts.
// Returns null when the expansion cap is hit.
function solveRoom(
  wall: Uint8Array,
  w: number,
  h: number,
  walkCells: number[],
  boxes: number[],
  goals: number[],
  startPlayer: number,
  maxExpansions: number,
  weight: number,
): RoomSolution | null {
  const n = walkCells.length;
  const idxOf = new Int32Array(w * h).fill(-1);
  walkCells.forEach((c, k) => {
    idxOf[c] = k;
  });
  const goalIdx = goals.map((g) => idxOf[g]);
  if (goalIdx.some((g) => g < 0)) return null;
  const isGoal = new Uint8Array(n);
  for (const g of goalIdx) isGoal[g] = 1;

  // All-pairs walking distances over walkCells, ignoring boxes.
  const dist = new Int32Array(n * n).fill(-1);
  for (let s = 0; s < n; s++) {
    const q = [s];
    dist[s * n + s] = 0;
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi]!;
      const cell = walkCells[c]!;
      const cx = cell % w;
      const cy = (cell / w) | 0;
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const ni = idxOf[ny * w + nx];
        if (ni < 0 || dist[s * n + ni] >= 0) continue;
        dist[s * n + ni] = dist[s * n + c]! + 1;
        q.push(ni);
      }
    }
  }

  // Static dead cells: a box there can never reach a goal. Box mobility is
  // computed ignoring other boxes (which only widens the reachable set, so
  // the test stays sound): from a cell the box can step to a neighbouring
  // floor cell when the stance cell behind it is floor too. A box whose
  // mobility set holds no goal is dead. This subsumes corners and wall runs:
  // a box against a wall can slide along it and step away where the wall
  // ends, so only a truly enclosed box is dead.
  const isWallAt = (x: number, y: number): boolean =>
    x < 0 || y < 0 || x >= w || y >= h || wall[y * w + x] === 1;
  const staticDead = new Uint8Array(n);
  for (let k = 0; k < n; k++) {
    const seenMob = new Uint8Array(n);
    const q = [k];
    seenMob[k] = 1;
    let reachesGoal = isGoal[k] === 1;
    for (let qi = 0; qi < q.length && !reachesGoal; qi++) {
      const c = q[qi]!;
      const cell = walkCells[c]!;
      const cx = cell % w;
      const cy = (cell / w) | 0;
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (isWallAt(nx, ny)) continue;
        const ni = idxOf[ny * w + nx];
        if (ni < 0 || seenMob[ni]) continue;
        // Stance: the player must stand on the far side to push this way.
        if (isWallAt(cx - d.dx, cy - d.dy)) continue;
        if (idxOf[(cy - d.dy) * w + (cx - d.dx)] < 0) continue;
        seenMob[ni] = 1;
        if (isGoal[ni]) {
          reachesGoal = true;
          break;
        }
        q.push(ni);
      }
    }
    if (!reachesGoal) staticDead[k] = 1;
  }

  const nb = boxes.length;
  const boxIdx = (bs: number[]): number[] => bs.map((b) => idxOf[b]!);
  const keyOf = (bs: number[], rep: number): string =>
    String.fromCharCode(...bs, rep);
  const solved = (bs: number[]): boolean =>
    bs.every((b) => isGoal[idxOf[b]!] === 1);

  // Minimum-matching heuristic: cheapest assignment of boxes to goals over
  // walking distances. Infinity when some box cannot reach any goal.
  const heuristic = (bIdx: number[]): number => {
    let best = Infinity;
    const perm = goalIdx.slice();
    const rec = (i: number, acc: number): void => {
      if (acc >= best) return;
      if (i === nb) {
        best = acc;
        return;
      }
      for (let j = i; j < nb; j++) {
        [perm[i], perm[j]] = [perm[j]!, perm[i]!];
        const d = dist[bIdx[i]! * n + perm[i]!]!;
        if (d >= 0) rec(i + 1, acc + d);
        [perm[i], perm[j]] = [perm[j]!, perm[i]!];
      }
    };
    rec(0, 0);
    return best;
  };

  // Dynamic deadlock: a box not on a goal that sits in a 2x2 block of
  // walls/boxes, or on a statically dead cell.
  const isDead = (bIdx: number[]): boolean => {
    const occ = new Uint8Array(w * h);
    for (const bi of bIdx) occ[walkCells[bi]!] = 1;
    const isBoxOrWall = (x: number, y: number): boolean => {
      if (x < 0 || y < 0 || x >= w || y >= h) return true;
      if (wall[y * w + x] === 1) return true;
      return occ[y * w + x] === 1;
    };
    for (const bi of bIdx) {
      if (isGoal[bi]) continue;
      if (staticDead[bi]) return true;
      const cell = walkCells[bi]!;
      const cx = cell % w;
      const cy = (cell / w) | 0;
      for (const [ox, oy] of [
        [0, 0],
        [-1, 0],
        [0, -1],
        [-1, -1],
      ]) {
        if (
          isBoxOrWall(cx + ox, cy + oy) &&
          isBoxOrWall(cx + ox + 1, cy + oy) &&
          isBoxOrWall(cx + ox, cy + oy + 1) &&
          isBoxOrWall(cx + ox + 1, cy + oy + 1)
        ) {
          return true;
        }
      }
    }
    return false;
  };

  // Binary heap on (f, h, insertion order): fully deterministic.
  type HeapEntry = [number, number, number, number];
  const heap: HeapEntry[] = [];
  const heapLess = (a: HeapEntry, b: HeapEntry): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: HeapEntry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): HeapEntry => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const startBoxes = boxes.slice().sort((a, b) => a - b);
  const startIdx = boxIdx(startBoxes);
  if (isDead(startIdx)) return null;
  const h0 = heuristic(startIdx);
  if (!Number.isFinite(h0)) return null;

  const stBoxes: number[][] = [startBoxes];
  const stPlayer: number[] = [startPlayer];
  const stParent: number[] = [-1];
  const stPushBox: number[] = [-1];
  const stPushDir: number[] = [-1];
  const stG: number[] = [0];
  const seen = new Map<string, number>();
  let seq = 0;
  // The region representative of the start state: smallest reachable cell.
  const startRegion = regionOf(startPlayer, startBoxes);
  seen.set(keyOf(startBoxes, startRegion.rep), 0);
  heapPush([h0, h0, seq++, 0]);

  let expanded = 0;
  let goalState = -1;

  // Reachable region of the player, given box positions: the cells the
  // player can stand on, and their smallest cell as the representative.
  function regionOf(playerPos: number, bs: number[]): { rep: number; set: Uint8Array } {
    const blocked = new Uint8Array(w * h);
    for (const b of bs) blocked[b] = 1;
    const set = new Uint8Array(w * h);
    const q = [playerPos];
    set[playerPos] = 1;
    let rep = playerPos;
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi]!;
      if (c < rep) rep = c;
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const nc = ny * w + nx;
        if (idxOf[nc] < 0 || blocked[nc] || set[nc]) continue;
        set[nc] = 1;
        q.push(nc);
      }
    }
    return { rep, set };
  }

  while (heap.length > 0) {
    const [, , , s] = heapPop();
    const bs = stBoxes[s]!;
    const g = stG[s]!;
    // Skip stale heap entries superseded by a cheaper route to the same key.
    const region = regionOf(stPlayer[s]!, bs);
    if (seen.get(keyOf(bs, region.rep)) !== s) continue;
    if (solved(bs)) {
      goalState = s;
      break;
    }
    if (++expanded > maxExpansions) return null;
    const bIdx = boxIdx(bs);
    for (let bi = 0; bi < nb; bi++) {
      const b = bs[bi]!;
      const bx = b % w;
      const by = (b / w) | 0;
      for (let d = 0; d < 4; d++) {
        const dir = DIRS[d]!;
        const stance = (by - dir.dy) * w + (bx - dir.dx);
        const dest = (by + dir.dy) * w + (bx + dir.dx);
        if (stance < 0 || stance >= w * h || dest < 0 || dest >= w * h) continue;
        if (!region.set[stance]) continue;
        if (idxOf[dest] < 0 || bs.includes(dest)) continue;
        const next = bs.slice();
        next[bi] = dest;
        next.sort((a, b) => a - b);
        const nextIdx = boxIdx(next);
        if (isDead(nextIdx)) continue;
        const hh = heuristic(nextIdx);
        if (!Number.isFinite(hh)) continue;
        const ng = g + 1;
        const rep = regionOf(b, next).rep;
        const k = keyOf(next, rep);
        const prevState = seen.get(k);
        if (prevState !== undefined && stG[prevState]! <= ng) continue;
        const t = stBoxes.length;
        stBoxes.push(next);
        stPlayer.push(b);
        stParent.push(s);
        stPushBox.push(b);
        stPushDir.push(d);
        stG.push(ng);
        seen.set(k, t);
        heapPush([ng + weight * hh, hh, seq++, t]);
      }
    }
  }
  if (goalState < 0) return null;

  // Reconstruct: for each push, the walk from the parent's player position
  // to the stance cell, then the push step.
  let moves = '';
  const steps: { box: number; dir: number; parent: number }[] = [];
  for (let t = goalState; t > 0; t = stParent[t]!) {
    steps.push({ box: stPushBox[t]!, dir: stPushDir[t]!, parent: stParent[t]! });
  }
  steps.reverse();
  for (const step of steps) {
    const dir = DIRS[step.dir]!;
    const parentBoxes = stBoxes[step.parent]!;
    const blocked = new Uint8Array(w * h);
    for (const b of parentBoxes) blocked[b] = 1;
    const stance = step.box - (dir.dy * w + dir.dx);
    const walk = walkPath(wall, w, h, blocked, stPlayer[step.parent]!, stance);
    if (walk === null) return null;
    moves += walk + dir.ch;
  }
  return { moves, endPlayer: stPlayer[goalState]!, endBoxes: stBoxes[goalState]! };
}

// Replay moves on the level: true only when every step is legal and every
// box ends on a goal.
function replayOk(level: Level, moves: string): boolean {
  const { w, h, wall, goals } = level;
  const boxes = new Set(level.boxes);
  let player = level.player;
  const goalSet = new Set(goals);
  for (const ch of moves) {
    const d = DIRS.find((dd) => dd.ch === ch.toLowerCase());
    if (!d) return false;
    const px = player % w;
    const py = (player / w) | 0;
    const nx = px + d.dx;
    const ny = py + d.dy;
    if (nx < 0 || ny < 0 || nx >= w || ny >= h) return false;
    const nc = ny * w + nx;
    if (wall[nc]) return false;
    if (boxes.has(nc)) {
      const bx = nx + d.dx;
      const by = ny + d.dy;
      if (bx < 0 || by < 0 || bx >= w || by >= h) return false;
      const bc = by * w + bx;
      if (wall[bc] || boxes.has(bc)) return false;
      boxes.delete(nc);
      boxes.add(bc);
    }
    player = nc;
  }
  for (const b of boxes) if (!goalSet.has(b)) return false;
  return true;
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  if (level.player < 0 || level.boxes.length === 0) return '';
  const rooms = decompose(level);
  if (rooms !== null && rooms.length > 0) {
    let player = level.player;
    let curBoxes = level.boxes.slice();
    let moves = '';
    for (const room of rooms) {
      const blocked = new Uint8Array(level.w * level.h);
      for (const b of curBoxes) blocked[b] = 1;
      const path = walkPath(level.wall, level.w, level.h, blocked, player, room.entry);
      if (path === null) return '';
      moves += path;
      player = room.entry;
      const walkCells = room.cells.slice();
      walkCells.push(room.entry);
      let res = solveRoom(
        level.wall, level.w, level.h, walkCells,
        room.boxes, room.goals, player, EXPANSION_CAP, 1,
      );
      if (res === null) {
        res = solveRoom(
          level.wall, level.w, level.h, walkCells,
          room.boxes, room.goals, player, EXPANSION_CAP, 2,
        );
      }
      if (res === null) return '';
      moves += res.moves;
      player = res.endPlayer;
      const roomSet = new Set(room.boxes);
      const next: number[] = [];
      for (const b of curBoxes) if (!roomSet.has(b)) next.push(b);
      next.push(...res.endBoxes);
      next.sort((a, b) => a - b);
      curBoxes = next;
    }
    if (moves.length > 60000) return '';
    return replayOk(level, moves) ? moves : '';
  }
  // Fallback: the whole level as one room.
  const all: number[] = [];
  for (let i = 0; i < level.w * level.h; i++) if (!level.wall[i]) all.push(i);
  let res = solveRoom(
    level.wall, level.w, level.h, all,
    level.boxes, level.goals, level.player, EXPANSION_CAP, 1,
  );
  if (res === null) {
    res = solveRoom(
      level.wall, level.w, level.h, all,
      level.boxes, level.goals, level.player, EXPANSION_CAP, 2,
    );
  }
  if (res === null) return '';
  if (res.moves.length > 60000) return '';
  return replayOk(level, res.moves) ? res.moves : '';
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
