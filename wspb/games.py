"""Small grid games for the policy families, written from scratch in numpy.

Every game is a 10x10 grid seen as 5 binary channels (500 inputs, index channel*100 + row*10 + col, row 0 at the
top, column 0 on the left) with 5 actions: 0 noop, 1 left, 2 right, 3 up, 4 down. A game ignores the actions it
has no use for. Sticky actions: with probability 0.1 each step the previous executed action is repeated instead.

Episodes are deterministic given a seed. All randomness is drawn in fixed-shape batches every step, whatever the
policy does, so two policies run on the same seed face the same episodes. Each game has an exact rules text, a
scripted expert that reads only the observation, and runs in batches: hundreds of episodes take well under a second.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

G, C, A = 10, 5, 5
NOOP, LEFT, RIGHT, UP, DOWN = range(5)
ACTIONS = ("noop", "left", "right", "up", "down")
STICKY = 0.1
COLS = np.arange(G, dtype=float)

Policy = Callable[[np.ndarray], np.ndarray]   # (n, 500) observations -> (n,) action indices


def _dx(a):
    return (a == RIGHT).astype(int) - (a == LEFT).astype(int)


def _dy(a):
    return (a == DOWN).astype(int) - (a == UP).astype(int)


class Game:
    name = ""
    steps = 200
    channels: tuple = (None,) * C

    def __init__(self, **params):
        self.params = params

    def rules(self) -> str: ...
    def reset(self, g: np.random.Generator, n: int) -> dict: ...
    def step(self, s: dict, a: np.ndarray, g: np.random.Generator) -> tuple[np.ndarray, np.ndarray]: ...
    def obs(self, s: dict) -> np.ndarray: ...
    def expert(self, o: np.ndarray) -> np.ndarray: ...

    def layout(self) -> str:
        return "\n".join(f"  channel {i}: {d or 'unused (always 0)'}" for i, d in enumerate(self.channels))


def _mark(o, ch, rows, cols):
    o[np.arange(len(rows)), ch, rows, cols] = 1.0


def _colsum(o, ch, rows=slice(None)):
    return o[:, ch, rows].sum(1) @ COLS


def _rowsum(o, ch):
    return o[:, ch].sum(2) @ COLS


class Breakout(Game):
    name = "breakout"
    steps = 300
    channels = ("the paddle (2 cells of row 9)", "the ball", "the ball's previous position (see the rules)",
                "bricks", None)

    def rules(self):
        return """Breakout. Rows 1-3 start full of bricks (30). The paddle is 2 cells wide and sits in row 9, covering
columns p and p+1 (p starts uniform in 0..8). The ball starts in row 4 moving diagonally down, having come from row 3:
its start column and left/right direction are random. Actions: left and right move the paddle one column (it stops
at the walls); every other action leaves it where it is.
Each step t = 0, 1, 2, ..., in this order:
1. The paddle moves.
2. Only when t is even: the ball tries to move one row and one column in its current direction. If that column is
   off the grid, its horizontal direction flips first; if that row is above row 0, its vertical direction flips
   first. Then: if the target cell holds a brick, the brick is destroyed (+1 point), the vertical direction flips
   and the ball stays where it is this time. Otherwise, if the target is in row 9 and the paddle covers it, the
   vertical direction flips and the ball stays where it is. Otherwise, if the target is in row 9, the ball is lost
   and the episode ends. Otherwise the ball moves to the target.
3. When every brick is gone, rows 1-3 refill (except a cell the ball is in).
So the paddle moves twice for every ball move. The ball's previous position is where it was before its latest move
attempt (the same cell when it stayed; at the start, the row-3 cell it came from). An episode lasts at most 300
steps; the score is the number of bricks destroyed."""

    def reset(self, g, n):
        c0, dc = g.integers(0, G, n), g.choice([-1, 1], n)
        dc = np.where((c0 + dc < 0) | (c0 + dc >= G), -dc, dc)
        bricks = np.zeros((n, G, G), bool)
        bricks[:, 1:4] = True
        return {"p": g.integers(0, G - 1, n), "r": np.full(n, 4), "c": c0 + dc, "dr": np.ones(n, int), "dc": dc,
                "tr": np.full(n, 3), "tc": c0, "bricks": bricks, "t": 0}

    def step(self, s, a, g):
        n = len(a)
        s["p"] = np.clip(s["p"] + _dx(a), 0, G - 2)
        s["t"] += 1
        if s["t"] % 2 == 0:   # the ball moves on even t, counted before this step
            return np.zeros(n), np.zeros(n, bool)
        r, c = s["r"], s["c"]
        s["tr"], s["tc"] = r.copy(), c.copy()
        dc = np.where((c + s["dc"] < 0) | (c + s["dc"] >= G), -s["dc"], s["dc"])
        dr = np.where(r + s["dr"] < 0, 1, s["dr"])
        nr, nc = r + dr, c + dc
        i = np.arange(n)
        brick = s["bricks"][i, np.minimum(nr, G - 1), nc] & (nr < G)
        s["bricks"][i[brick], nr[brick], nc[brick]] = False
        bottom = ~brick & (nr == G - 1)
        paddle = bottom & ((nc == s["p"]) | (nc == s["p"] + 1))
        lost = bottom & ~paddle
        stay = brick | paddle
        dr = np.where(stay, -dr, dr)
        s["r"], s["c"] = np.where(stay, r, nr), np.where(stay, c, nc)
        s["dr"], s["dc"] = dr, dc
        s["r"] = np.where(lost, 4, s["r"])    # lost episodes are over; keep their state valid
        s["dr"] = np.where(lost, 1, s["dr"])
        empty = ~s["bricks"].any((1, 2))
        if empty.any():
            refill = np.zeros((n, G, G), bool)
            refill[:, 1:4] = True
            refill[i, s["r"], s["c"]] = False
            s["bricks"][empty] = refill[empty]
        return brick.astype(float), lost

    def obs(self, s):
        n = len(s["p"])
        o = np.zeros((n, C, G, G))
        _mark(o, 0, np.full(n, G - 1), s["p"])
        _mark(o, 0, np.full(n, G - 1), s["p"] + 1)
        _mark(o, 1, s["r"], s["c"])
        _mark(o, 2, s["tr"], s["tc"])
        o[:, 3] = s["bricks"]
        return o

    def expert(self, o):
        e = 2 * (2 * _colsum(o, 1) - _colsum(o, 2)) - _colsum(o, 0)
        return np.where(e <= -3, LEFT, np.where(e >= 3, RIGHT, NOOP))


class Pong(Game):
    name = "pong"
    steps = 300
    channels = ("your paddle (2 cells of column 9)", "the ball", "the ball's previous position (see the rules)",
                "the opponent's paddle (2 cells of column 0)", None)

    def rules(self):
        return """Pong. Your paddle is 2 cells tall in column 9, covering rows p and p+1 (p starts uniform in 0..8). The
opponent's paddle is 2 cells tall in column 0, covering rows q and q+1 (q starts uniform in 0..8). The ball is served
from column 4 in a row uniform in 2..7, moving right toward you and diagonally up or down at random, having come
from column 3. Actions: up and down move your paddle one row (it stops at the walls); every other action leaves it
where it is.
Each step t = 0, 1, 2, ..., in this order:
1. Your paddle moves. If t mod 3 = 0, the opponent's paddle moves one row toward the ball's row (up if the ball's
   row is above q, down if it is below q+1, otherwise not at all).
2. Only when t is even: the ball tries to move one row and one column in its current direction. If that row is off
   the grid, its vertical direction flips first. Then: if the target is in column 9 and your paddle covers the
   target's row, you return the ball (+1 point), its horizontal direction flips and it stays where it is this time;
   if the target is in column 9 and your paddle doesn't cover it, the episode ends. If the target is in column 0 and
   the opponent's paddle covers its row, the horizontal direction flips and the ball stays where it is; if the
   opponent's paddle doesn't cover it, the ball is served again as at the start (no points). Otherwise the ball
   moves to the target.
So your paddle moves twice for every ball move. The ball's previous position is where it was before its latest move
attempt (the same cell when it stayed; after a serve, the column-3 cell it came from). An episode lasts at most 300
steps; the score is the number of returns."""

    def _serve(self, s, m, g, rows, drs):
        s["r"] = np.where(m, rows, s["r"])
        s["c"] = np.where(m, 4, s["c"])
        s["dr"] = np.where(m, drs, s["dr"])
        s["dc"] = np.where(m, 1, s["dc"])
        s["tr"] = np.where(m, rows - drs, s["tr"])
        s["tc"] = np.where(m, 3, s["tc"])

    def reset(self, g, n):
        s = {"p": g.integers(0, G - 1, n), "q": g.integers(0, G - 1, n), "t": 0,
             **{k: np.zeros(n, int) for k in ("r", "c", "dr", "dc", "tr", "tc")}}
        self._serve(s, np.ones(n, bool), g, g.integers(2, 8, n), g.choice([-1, 1], n))
        return s

    def step(self, s, a, g):
        n = len(a)
        rows, drs = g.integers(2, 8, n), g.choice([-1, 1], n)
        s["p"] = np.clip(s["p"] + _dy(a), 0, G - 2)
        if s["t"] % 3 == 0:
            s["q"] = np.clip(s["q"] - (s["r"] < s["q"]) + (s["r"] > s["q"] + 1), 0, G - 2)
        s["t"] += 1
        if s["t"] % 2 == 0:
            return np.zeros(n), np.zeros(n, bool)
        r, c = s["r"], s["c"]
        s["tr"], s["tc"] = r.copy(), c.copy()
        dr = np.where((r + s["dr"] < 0) | (r + s["dr"] >= G), -s["dr"], s["dr"])
        nr, nc = r + dr, c + s["dc"]
        mine = (nc == G - 1) & (nr >= s["p"]) & (nr <= s["p"] + 1)
        lost = (nc == G - 1) & ~mine
        theirs = (nc == 0) & (nr >= s["q"]) & (nr <= s["q"] + 1)
        past = (nc == 0) & ~theirs
        stay = mine | theirs
        s["dc"] = np.where(stay, -s["dc"], s["dc"])
        s["dr"] = dr
        s["r"], s["c"] = np.where(stay, r, nr), np.where(stay, c, nc)
        self._serve(s, past | lost, g, rows, drs)
        return mine.astype(float), lost

    def obs(self, s):
        n = len(s["p"])
        o = np.zeros((n, C, G, G))
        for k in (0, 1):
            _mark(o, 0, s["p"] + k, np.full(n, G - 1))
            _mark(o, 3, s["q"] + k, np.zeros(n, int))
        _mark(o, 1, s["r"], s["c"])
        _mark(o, 2, s["tr"], s["tc"])
        return o

    def expert(self, o):
        e = 2 * (2 * _rowsum(o, 1) - _rowsum(o, 2)) - _rowsum(o, 0)
        return np.where(e <= -3, UP, np.where(e >= 3, DOWN, NOOP))


class Freeway(Game):
    name = "freeway"
    steps = 300
    channels = ("the chicken (always in column 4)", "cars", "the cars' positions one step earlier", None, None)

    def __init__(self, dirs=(1, -1, 1, -1, 1, -1, 1, -1), cars=2):
        super().__init__(dirs=tuple(int(d) for d in dirs), cars=int(cars))
        self.dirs, self.cars = np.array(self.params["dirs"]), self.params["cars"]

    def rules(self):
        way = ", ".join(f"row {i + 1} {'right' if d > 0 else 'left'}" for i, d in enumerate(self.dirs))
        cars = "1 car" if self.cars == 1 else f"{self.cars} cars (in different columns)"
        return f"""Freeway. The chicken walks up column 4 from row 9 toward row 0. Rows 1-8 are lanes, each with {cars}
starting in random columns; a car is one cell. All cars in a lane move the same way:
{way}.
A car moving off one side reappears on the other (column 9 to 0 or 0 to 9). Each lane has a speed drawn at the start
of every episode: its cars move one cell every s steps, s uniform in {{1, 2, 3}}, on the steps where (t + o) mod s = 0
for a random offset o in 0..s-1. Rows 0 and 9 have no cars. Actions: up and down move the chicken one row (down
stops at row 9); left, right and noop leave it where it is.
Each step t = 0, 1, 2, ..., in this order:
1. The chicken moves. If it reaches row 0 you score +1 and it goes back to row 9.
2. If a car is in the chicken's cell, the chicken is hit and goes back to row 9.
3. The cars move.
4. If a car is in the chicken's cell, the chicken is hit and goes back to row 9.
Being hit costs nothing but the trip. The cars' previous positions are where they were before step 3. Every
episode lasts exactly 300 steps; the score is the number of crossings."""

    def reset(self, g, n):
        cars = np.zeros((n, G, G), bool)
        for lane in range(1, 9):
            cols = np.argsort(g.random((n, G)), 1)[:, :self.cars]
            for k in range(self.cars):
                cars[np.arange(n), lane, cols[:, k]] = True
        speed = g.integers(1, 4, (n, 8))
        return {"y": np.full(n, 9), "cars": cars, "prev": cars.copy(), "speed": speed,
                "off": (g.random((n, 8)) * speed).astype(int), "t": 0}

    def step(self, s, a, g):
        n = len(a)
        i = np.arange(n)
        s["y"] = np.clip(s["y"] + _dy(a), 0, G - 1)
        cross = s["y"] == 0
        s["y"] = np.where(cross, 9, s["y"])
        s["y"] = np.where(s["cars"][i, s["y"], 4], 9, s["y"])
        s["prev"] = s["cars"].copy()
        move = (s["t"] + s["off"]) % s["speed"] == 0
        for lane in range(8):
            m = move[:, lane]
            s["cars"][m, lane + 1] = np.roll(s["cars"][m, lane + 1], self.dirs[lane], axis=1)
        s["t"] += 1
        s["y"] = np.where(s["cars"][i, s["y"], 4], 9, s["y"])
        return cross.astype(float), np.zeros(n, bool)

    def obs(self, s):
        n = len(s["y"])
        o = np.zeros((n, C, G, G))
        _mark(o, 0, s["y"], np.full(n, 4))
        o[:, 1], o[:, 2] = s["cars"], s["prev"]
        return o

    def danger(self, o, dy):
        """Whether moving by dy (-1 up, 0 stay, +1 down) could put the chicken under a car this step."""
        y = o[:, 0, :, 4].argmax(1)
        to = np.clip(y + dy, 0, G - 1)
        lane = np.clip(to - 1, 0, 7)
        behind = 4 - self.dirs[lane]
        i = np.arange(len(o))
        hit = o[i, 1, to, behind] > 0
        if dy:
            hit |= o[i, 1, to, 4] > 0
        return hit & (to >= 1) & (to <= 8) & ((y + dy) <= G - 1)

    def expert(self, o):
        up, stay, down = self.danger(o, -1), self.danger(o, 0), self.danger(o, 1)
        return np.where(~up, UP, np.where(~stay, NOOP, np.where(~down, DOWN, NOOP)))


class Catch(Game):
    name = "catch"
    steps = 200
    channels = ("you (one cell of row 9)", "fruit", None, "rocks", None)

    def __init__(self, fruit=0.5):
        super().__init__(fruit=float(fruit))
        self.fruit = self.params["fruit"]

    def rules(self):
        return f"""Catch. You are one cell in row 9 (column uniform in 0..9 at the start). Things fall from the top: on
every step t with t mod 3 = 0 (starting at t = 0) one appears in row 0, in a uniform random column, and it is fruit
with probability {self.fruit:g}, otherwise a rock. Actions: left and right move you one column (you stop at the walls);
every other action leaves you where you are.
Each step t = 0, 1, 2, ..., in this order:
1. You move.
2. Everything falling moves down one row. Whatever reaches row 9 lands: a fruit landing in your column scores +1, a
   rock landing in your column ends the episode, and anything that lands disappears either way.
3. If t mod 3 = 0, a new thing appears in row 0.
So falling things are 3 rows apart, rows 0-8 hold at most three of them, and rows 6-8 hold the next one to land.
An episode lasts at most 200 steps; the score is the number of fruit caught."""

    def reset(self, g, n):
        return {"x": g.integers(0, G, n), "fruit": np.zeros((n, G, G), bool), "rock": np.zeros((n, G, G), bool),
                "t": 0}

    def step(self, s, a, g):
        n = len(a)
        i = np.arange(n)
        col, kind = g.integers(0, G, n), g.random(n) < self.fruit
        s["x"] = np.clip(s["x"] + _dx(a), 0, G - 1)
        for k in ("fruit", "rock"):
            s[k] = np.roll(s[k], 1, axis=1)
            s[k][:, 0] = False
        got, dead = s["fruit"][i, 9, s["x"]], s["rock"][i, 9, s["x"]]
        s["fruit"][:, 9] = s["rock"][:, 9] = False
        if s["t"] % 3 == 0:
            s["fruit"][i, 0, col] = kind
            s["rock"][i, 0, col] = ~kind
        s["t"] += 1
        return got.astype(float), dead

    def obs(self, s):
        n = len(s["x"])
        o = np.zeros((n, C, G, G))
        _mark(o, 0, np.full(n, G - 1), s["x"])
        o[:, 1], o[:, 3] = s["fruit"], s["rock"]
        return o

    @staticmethod
    def features(o):
        x = _colsum(o, 0)
        near, far = slice(6, 9), slice(3, 6)
        fn, rn, ff = (o[:, ch, rows].sum((1, 2)) for ch, rows in ((1, near), (3, near), (1, far)))
        return x, fn, _colsum(o, 1, near), rn, _colsum(o, 3, near), ff, _colsum(o, 1, far)

    def expert(self, o):
        """Dodge the next rock if it's over you; else go for the next fruit; else head for the one after, without
        stepping under the next rock."""
        x, fn, fc, rn, rc, ff, gc = self.features(o)
        under = (rn > 0) & (rc == x)
        toward = lambda present, col, d: (present > 0) & (np.sign(col - x) == d)
        a = np.full(len(o), NOOP)
        for d, act in ((1, RIGHT), (-1, LEFT)):
            go = toward(ff, gc, d) & (fn == 0) & ~((rn > 0) & (rc == x + d))
            a = np.where(go, act, a)
            a = np.where((fn > 0) & ~under & toward(fn, fc, d), act, a)
        return np.where(under, np.where(x <= 4, RIGHT, LEFT), a)


GAMES = {"breakout": Breakout, "pong": Pong, "freeway": Freeway, "catch": Catch}


def run(game: Game, policy: Policy, seed, n: int, *, chan=None, acts=None, sticky: float = STICKY,
        steps: int | None = None, explore: float = 0.0, record: bool = False):
    """Play n episodes from one seed and return each episode's score.

    chan[i] is the game channel the policy sees as its channel i and acts[j] the game action its output j means
    (both default to the identity). With explore > 0 that fraction of steps takes a uniform random action instead,
    and with record=True the observations the policy saw and the steps they were live are returned too."""
    g = np.random.default_rng(seed)
    s = game.reset(g, n)
    acts = np.arange(A) if acts is None else np.asarray(acts)
    prev, score, live = np.zeros(n, int), np.zeros(n), np.ones(n, bool)
    seen = []
    for _ in range(steps or game.steps):
        o = game.obs(s)
        x = (o if chan is None else o[:, list(chan)]).reshape(n, -1)
        a = acts[np.asarray(policy(x), int)]
        rand, stick, u = g.integers(0, A, n), g.random(n) < sticky, g.random(n)
        a = np.where(u < explore, rand, a)
        a = np.where(stick, prev, a)
        if record:
            seen.append((x, live.copy()))
        prev = a
        r, done = game.step(s, a, g)
        score += r * live
        live &= ~done
        if not live.any():
            break
    if record:
        return score, np.concatenate([x[m] for x, m in seen]), np.concatenate([m[m] for _, m in seen])
    return score


def expert_policy(game: Game, chan=None, acts=None) -> Policy:
    """The scripted expert, seeing the observation through a layout (chan, acts) the way a network would."""
    inv_c = None if chan is None else np.argsort(chan)
    inv_a = None if acts is None else np.argsort(acts)

    def pol(x):
        o = x.reshape(len(x), C, G, G)
        o = o if inv_c is None else o[:, inv_c]
        a = game.expert(o)
        return a if inv_a is None else inv_a[a]
    return pol


def constant_policy(action: int) -> Policy:
    return lambda x: np.full(len(x), action)


def random_policy(seed) -> Policy:
    g = np.random.default_rng(seed)
    return lambda x: g.integers(0, A, len(x))


def net_policy(p, meta) -> Policy:
    from . import nn
    return lambda x: nn.mlp_forward(p, meta, x).argmax(1)


CACHE = Path(os.environ.get("WSPB_CACHE", Path.home() / ".cache" / "wspb")) / "policies"


def _key(*parts) -> str:
    h = hashlib.sha256()
    for f in (Path(__file__), Path(__file__).with_name("zoo.py")):
        h.update(f.read_bytes())
    h.update(repr(parts).encode())
    return h.hexdigest()[:16]


@lru_cache(maxsize=None)
def trained(name: str, seed: int, hidden: tuple[int, ...] = (32, 32), wd: float = 1e-3, spare: int = 0,
            states: int = 12000, steps: int = 400) -> tuple[dict, dict]:
    """A relu policy behaviour-cloned from the expert: (params, meta). The last `spare` units of every hidden layer
    are held at exactly zero. Trained once per machine and kept on disk (keyed by this file's and zoo.py's source),
    so treat the arrays as read-only."""
    from . import nn, wio, zoo
    sizes = [C * G * G, *hidden, A]
    meta = {"kind": "mlp", "sizes": sizes, "acts": ["relu"] * len(hidden) + ["identity"]}
    path = CACHE / f"{name}-{seed}-{_key(name, seed, hidden, wd, spare, states, steps)}.safetensors"
    if path.exists():
        p, _ = wio.parse(path.read_bytes())
    else:
        game = GAMES[name]()
        _, X, _ = run(game, expert_policy(game), [seed, 1], 256, explore=0.3, record=True)
        X = X[np.random.default_rng([seed, 2]).permutation(len(X))[:states]]
        masks = {}
        if spare:
            for i, h in enumerate(hidden):
                keep = np.r_[np.ones(h - spare), np.zeros(spare)]
                masks[f"l{i}.w"] = masks.get(f"l{i}.w", np.ones((h, sizes[i]))) * keep[:, None]
                masks[f"l{i}.b"] = keep
                masks[f"l{i + 1}.w"] = np.ones((sizes[i + 2], h)) * keep[None, :]
        p = zoo.train(sizes, meta["acts"], X, expert_policy(game)(X), seed=seed, steps=steps, lr=1e-2, wd=wd,
                      masks=masks)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(wio.dump(p, meta))
        tmp.replace(path)
    for v in p.values():
        v.flags.writeable = False
    return p, meta
