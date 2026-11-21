// Roomwise: a Sokoban solver for make.fun's sokoban-solver-01.
//
// The levels this competition draws are a hallway with one to four small rooms
// off it, each room holding 3 or 4 boxes and as many goals, and a box pushed
// out of a room through its door can never come back -- so the rooms are
// independent problems joined only by the player's walk, and the level's
// push-optimal count is the sum of the rooms' optima. This program solves
// each room on its own, push-optimally, and stitches the walks together.
//
// Each room is searched over pushes, not steps: a state is the boxes' squares
// plus the player's reachable region (named by its lowest square, so every
// player square in one region is the same state). The search is A* with the
// minimum-cost assignment of boxes to goals as its estimate -- each push
// moves one box one square, so that sum can fall by at most one per push,
// which makes the estimate admissible and the first solved state popped
// push-optimal. Two prunings keep the search small: a box standing on a
// square from which no sequence of pushes reaches a goal is dead, and a
// 2 x 2 block of walls and boxes holding a box off its goal is frozen.
// A breadth-first search over the same states is the fallback when A*
// spends its work budget. Everything is budgeted by work, never by the
// clock, and every tie is broken in a fixed order, so two runs of this
// program on one set write the same bytes.
//
// Node 22 runs this file as it is (erasable TypeScript only, node:fs and
// node:path). No network, no files outside /input, /output and /tmp, no clock.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The four directions: [dx, dy, step letter, push letter]. */
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

/** The most states the breadth-first fallback keeps per room, and the most
 *  states the A* expands per room: work, never the clock. */
const BFS_STATE_CAP = 250_000;
const ASTAR_EXPANSION_CAP = 200_000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** A level in XSB text: # wall, space floor, . goal, $ box, * box on a goal,
 *  @ player, + player on a goal. Short rows pad out as walls. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** The squares from which a box can be pushed onto some goal, found by
 *  pulling back from the goals: a box on square y can be pushed to a live
 *  square x when the player can stand on the square behind it, y - (x - y). */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Walking distances from one goal to every square, over floor alone. */
function distancesFrom(level: Level, goal: number): Int32Array {
  const { w, wall } = level;
  const dist = new Int32Array(wall.length).fill(-1);
  const queue = [goal];
  dist[goal] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (!wall[q] && dist[q] < 0) {
        dist[q] = dist[p] + 1;
        queue.push(q);
      }
    }
  }
  return dist;
}

/** Every ordering of 0..n-1, for the assignment estimate (n is at most 4). */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const rec = (): void => {
    if (cur.length === n) {
      out.push([...cur]);
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out;
}

/**
 * One push-search over a level: A* when `astar` is set, breadth-first
 * otherwise. Returns {pushes, moves} (moves in LURD, a lower letter a step,
 * an upper one a push) or null when the work budget runs out first.
 */
function searchPushes(level: Level, astar: boolean, workCap: number): { pushes: number; moves: string } | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const n = level.boxes.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const live = liveSquares(level);
  if (level.boxes.some((b) => !live[b])) return null;
  // The estimate: the cheapest way to match every box to a distinct goal,
  // by walking distance over floor alone -- or, past five boxes, the sum of
  // each box's distance to its nearest goal. Each push moves one box one
  // square, so the estimate can fall by at most one per push: admissible.
  // Cached per box placement.
  const dists = goals.map((g) => distancesFrom(level, g));
  const perms = n <= 5 ? permutations(n) : null;
  const hCache = new Map<string, number>();
  const estimate = (boxes: number[]): number => {
    const key = boxes.join(',');
    const hit = hCache.get(key);
    if (hit !== undefined) return hit;
    let best = Infinity;
    if (perms !== null) {
      for (const perm of perms) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          const d = dists[perm[i]]![boxes[i]!]!;
          if (d < 0) {
            sum = Infinity;
            break;
          }
          sum += d;
        }
        if (sum < best) best = sum;
      }
    } else {
      let sum = 0;
      for (const b of boxes) {
        let d = Infinity;
        for (const dist of dists) if (dist[b]! >= 0 && dist[b]! < d) d = dist[b]!;
        if (d === Infinity) {
          sum = Infinity;
          break;
        }
        sum += d;
      }
      best = sum;
    }
    hCache.set(key, best);
    return best;
  };
  const solved = (boxes: number[]): boolean => boxes.every((b) => goalSet[b] === 1);

  // The flood fill: the squares the player reaches from `from` with the
  // boxes in `occupied` in the way, stamped into `seen`; returns the
  // region's lowest square.
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const regionLow = (occupied: Uint8Array, from: number, marks: Int32Array): number => {
    let low = from;
    let sp = 0;
    stack[sp++] = from;
    marks[from] = ++stamp;
    while (sp > 0) {
      const p = stack[--sp];
      if (p < low) low = p;
      const up = p - w;
      const down = p + w;
      const left = p - 1;
      const right = p + 1;
      if (!wall[up] && !occupied[up] && marks[up] !== stamp) {
        marks[up] = stamp;
        stack[sp++] = up;
      }
      if (!wall[down] && !occupied[down] && marks[down] !== stamp) {
        marks[down] = stamp;
        stack[sp++] = down;
      }
      if (!wall[left] && !occupied[left] && marks[left] !== stamp) {
        marks[left] = stamp;
        stack[sp++] = left;
      }
      if (!wall[right] && !occupied[right] && marks[right] !== stamp) {
        marks[right] = stamp;
        stack[sp++] = right;
      }
    }
    return low;
  };

  /** Whether the box just pushed to `t` sits in a frozen 2 x 2 block of
   *  walls and boxes holding a box off its goal. */
  const frozenBlock = (occupied: Uint8Array, t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
      if ((occupied[a] && !goalSet[a]) || (occupied[b] && !goalSet[b]) || (occupied[c] && !goalSet[c]) || (occupied[d] && !goalSet[d])) return true;
    }
    return false;
  };

  // A state's key: its region's lowest square, then its boxes, as digits in
  // base `size` while that stays exact, else as text.
  const numericKeys = Math.pow(size, n + 1) < Number.MAX_SAFE_INTEGER;
  const keyOf = (boxes: number[], low: number): number | string => {
    if (!numericKeys) return `${low}:${boxes.join(',')}`;
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  // The states, in the order found: their boxes, the player's square, the
  // parent state and the push that made them.
  const sBoxes: number[][] = [level.boxes.slice()];
  const sPlayer: number[] = [level.player];
  const sParent: number[] = [-1];
  const sPushDir: number[] = [-1];
  const gScore: number[] = [0];
  const bestSeen = new Map<number | string, number>();
  const occupied = new Uint8Array(size);
  for (const b of level.boxes) occupied[b] = 1;
  bestSeen.set(keyOf(level.boxes, regionLow(occupied, level.player, seen)), 0);
  for (const b of level.boxes) occupied[b] = 0;
  if (solved(level.boxes)) return { pushes: 0, moves: '' };

  // The open list: A* orders by (estimate + pushes, deeper first, older
  // first); breadth-first orders by pushes alone, older first. A binary
  // heap on a fixed comparison keeps every tie in a fixed order.
  const heap: number[] = [];
  const heapKey: [number, number, number][] = [];
  const less = (a: number, b: number): boolean => {
    const ka = heapKey[a]!;
    const kb = heapKey[b]!;
    return ka[0] < kb[0] || (ka[0] === kb[0] && (ka[1] > kb[1] || (ka[1] === kb[1] && ka[2] < kb[2])));
  };
  const heapPush = (id: number): void => {
    heap.push(id);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  heapKey.push(astar ? [estimate(level.boxes), 0, 0] : [0, 0, 0]);
  heapPush(0);

  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const s = heapPop();
    const boxes = sBoxes[s]!;
    const g = gScore[s]!;
    for (const b of boxes) occupied[b] = 1;
    // This state's region, stamped into `seen`: the push-square test and
    // the stale check below both read it, while the child's region is
    // stamped into `mark`, so the two never clobber each other.
    const low = regionLow(occupied, sPlayer[s]!, seen);
    const regionStamp = stamp;
    // Stale entry: a cheaper way to this state was already found.
    if (bestSeen.get(keyOf(boxes, low)) !== g) {
      for (const b of boxes) occupied[b] = 0;
      continue;
    }
    // A solved state is only claimed when popped, never when generated:
    // popping in (estimate + pushes) order is what makes the first solved
    // state push-optimal.
    if (solved(boxes)) {
      found = s;
      for (const b of boxes) occupied[b] = 0;
      break;
    }
    expanded++;
    if (expanded > workCap) {
      for (const b of boxes) occupied[b] = 0;
      return null;
    }
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di]!;
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== regionStamp || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(occupied, t);
        let key: number | string = -1;
        let next: number[] | null = null;
        let h = 0;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          h = estimate(next);
          if (h < Infinity) key = keyOf(next, regionLow(occupied, b, mark));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null || h === Infinity) continue;
        const g2 = g + 1;
        const prev = bestSeen.get(key);
        if (prev !== undefined && prev <= g2) continue;
        if (!astar && sBoxes.length >= BFS_STATE_CAP) {
          for (const b2 of boxes) occupied[b2] = 0;
          return null;
        }
        bestSeen.set(key, g2);
        const id = sBoxes.length;
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(s);
        sPushDir.push(di);
        gScore.push(g2);
        heapKey.push(astar ? [g2 + h, g2, id] : [g2, g2, id]);
        heapPush(id);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The path of states, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = sBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const di = sPushDir[s]!;
    const [, , , pushLetter] = DIRS[di]!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += pushLetter;
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower
 *  LURD letters (u first on ties, then d, l, r). */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, case aside). */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each part left that
 * holds a box is a room. Null when some part holds more or fewer boxes than
 * goals, or a box starts in the hallway.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** One room's push-optimal solution, A* first, breadth-first as fallback. */
function solveRoom(sub: Level): { pushes: number; moves: string } | null {
  return searchPushes(sub, true, ASTAR_EXPANSION_CAP) ?? searchPushes(sub, false, BFS_STATE_CAP);
}

/**
 * The level solved room by room: each room's push-optimal solution in turn,
 * the other rooms' boxes standing as walls (on their goals once their room
 * is done), the player walking on from where the last room left it. A box
 * never leaves its room, so the rooms' optima add up to the level's.
 * Falls back to a search over the whole level when the level is no set of
 * rooms, or a room's search gives up.
 */
function solveLevel(level: Level): { pushes: number; moves: string } {
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let pushes = 0;
    let moves = '';
    let ok = true;
    for (const room of rooms) {
      const wall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub: Level = {
        w: level.w,
        h: level.h,
        wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      const solved = solveRoom(sub);
      if (solved === null) {
        ok = false;
        break;
      }
      pushes += solved.pushes;
      moves += solved.moves;
      player = endOf(level.w, player, solved.moves);
      room.done = true;
    }
    if (ok) return { pushes, moves };
  }
  const whole = searchPushes(level, true, ASTAR_EXPANSION_CAP) ?? searchPushes(level, false, BFS_STATE_CAP);
  if (whole !== null) return whole;
  // Nothing solved it: an empty answer is still a valid answer (no moves),
  // worth no points but keeping the instance answered in its stated format.
  return { pushes: 0, moves: '' };
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'))).moves;
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
