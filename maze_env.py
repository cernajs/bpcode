"""
Custom 2D continuous point-maze environments for latent geometry experiments.

Provides pixel observations (RGB), continuous Box(2) actions, and ground-truth
geodesic distance computation via Dijkstra on a discretised free-space graph.

Compatible with the Dreamer training loop in dreamer_dyn_pb.py — exposes the
same interface as DMControlWrapper / PixelObsWrapper:
    observation_space, action_space, reset(**kw), step(action, repeat=1), close()
"""

import numpy as np
import gymnasium as gym
from collections import deque
from typing import List, Tuple, Optional, Dict

# ---------------------------------------------------------------------------
# Predefined maze layouts (1=wall, 0=free, S=start, G=goal)
# ---------------------------------------------------------------------------

MAZE_LAYOUTS: Dict[str, List[str]] = {
    "corridor": [
        "111111111",
        "1S00000G1",
        "111111111",
    ],
    "two_room": [
        "11111111111",
        "1S000100001",
        "10000100001",
        "10000000001",
        "10000100001",
        "10000100G01",
        "11111111111",
    ],
    "loop": [
        "111111111",
        "1S0000001",
        "101111101",
        "100000001",
        "100000001",
        "101111101",
        "10000G001",
        "111111111",
    ],
    "four_room": [
        "11111111111",
        "1S000100001",
        "10000100001",
        "10000000001",
        "10000100001",
        "11100011101",
        "10000100001",
        "10000000001",
        "10000100001",
        "10000100G01",
        "11111111111",
    ],
    "spiral": [
        "11111111111",
        "1S000000001",
        "11111111101",
        "10000000001",
        "10111111111",
        "10000000001",
        "11111111101",
        "10000000001",
        "10111111111",
        "1G000000001",
        "11111111111",
    ],
}

# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def parse_grid(lines: List[str]):
    """Parse maze text lines -> 2D char grid, start (r,c), goal (r,c)."""
    grid, start, goal = [], None, None
    for r, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        row = []
        for c, ch in enumerate(line):
            if ch == "S":
                start = (r, c)
                row.append("0")
            elif ch == "G":
                goal = (r, c)
                row.append("0")
            else:
                row.append(ch)
        grid.append(row)
    assert start is not None, "Maze must contain 'S'"
    assert goal is not None, "Maze must contain 'G'"
    return grid, start, goal


def generate_random_maze(
    h: int = 11, w: int = 11, wall_density: float = 0.20, seed: int = 0
) -> List[str]:
    """Generate a random connected maze with the given wall density."""
    rng = np.random.RandomState(seed)
    grid = [["0"] * w for _ in range(h)]

    for r in range(h):
        grid[r][0] = grid[r][w - 1] = "1"
    for c in range(w):
        grid[0][c] = grid[h - 1][c] = "1"

    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if rng.random() < wall_density:
                grid[r][c] = "1"

    start, goal = (1, 1), (h - 2, w - 2)
    grid[start[0]][start[1]] = "0"
    grid[goal[0]][goal[1]] = "0"

    # BFS connectivity check; carve path if disconnected
    if not _bfs_connected(grid, start, goal, h, w):
        _carve_path(grid, start, goal, h, w)

    lines = ["".join(row) for row in grid]
    s_row = list(lines[start[0]])
    s_row[start[1]] = "S"
    lines[start[0]] = "".join(s_row)
    g_row = list(lines[goal[0]])
    g_row[goal[1]] = "G"
    lines[goal[0]] = "".join(g_row)
    return lines


def _bfs_connected(grid, start, goal, h, w):
    visited = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 < nr < h - 1 and 0 < nc < w - 1 and (nr, nc) not in visited and grid[nr][nc] == "0":
                visited.add((nr, nc))
                q.append((nr, nc))
    return False


def _carve_path(grid, start, goal, h, w):
    """BFS ignoring walls, then remove walls along the found path."""
    visited = {start}
    parent = {}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 < nr < h - 1 and 0 < nc < w - 1 and (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    cell = goal
    while cell != start:
        grid[cell[0]][cell[1]] = "0"
        cell = parent[cell]


# ---------------------------------------------------------------------------
# Geodesic distance computation
# ---------------------------------------------------------------------------


class GeodesicComputer:
    """Pre-computes all-pairs shortest paths on the maze free-space graph
    using 8-connected Dijkstra (cardinal weight 1, diagonal sqrt(2))."""

    def __init__(self, grid: List[List[str]]):
        self.grid = grid
        self.H = len(grid)
        self.W = len(grid[0])

        self.cell_to_idx: Dict[Tuple[int, int], int] = {}
        self.idx_to_cell: List[Tuple[int, int]] = []
        idx = 0
        for r in range(self.H):
            for c in range(self.W):
                if grid[r][c] == "0":
                    self.cell_to_idx[(r, c)] = idx
                    self.idx_to_cell.append((r, c))
                    idx += 1
        self.n_free = idx

        from scipy.sparse import lil_matrix
        from scipy.sparse.csgraph import shortest_path

        adj = lil_matrix((self.n_free, self.n_free))
        _dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        _wts = [np.sqrt(2), 1, np.sqrt(2), 1, 1, np.sqrt(2), 1, np.sqrt(2)]
        for (r, c), i in self.cell_to_idx.items():
            for (dr, dc), w in zip(_dirs, _wts):
                nb = (r + dr, c + dc)
                if nb in self.cell_to_idx:
                    adj[i, self.cell_to_idx[nb]] = w
        self.dist_matrix = shortest_path(adj.tocsr(), method="D", directed=False)

    def pos_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Continuous (x,y) -> nearest free grid cell (row, col)."""
        r, c = int(np.clip(y, 0, self.H - 1)), int(np.clip(x, 0, self.W - 1))
        if (r, c) in self.cell_to_idx:
            return (r, c)
        visited = set()
        q = deque([(r, c)])
        visited.add((r, c))
        while q:
            cr, cc = q.popleft()
            if (cr, cc) in self.cell_to_idx:
                return (cr, cc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.H and 0 <= nc < self.W and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return self.idx_to_cell[0]

    def distance(self, pos_a, pos_b) -> float:
        ca = self.cell_to_idx[self.pos_to_cell(float(pos_a[0]), float(pos_a[1]))]
        cb = self.cell_to_idx[self.pos_to_cell(float(pos_b[0]), float(pos_b[1]))]
        return float(self.dist_matrix[ca, cb])

    def pairwise_distances(self, positions: np.ndarray) -> np.ndarray:
        """positions: [N, 2] (x, y) -> [N, N] geodesic distance matrix."""
        idxs = np.array(
            [self.cell_to_idx[self.pos_to_cell(float(p[0]), float(p[1]))] for p in positions]
        )
        return self.dist_matrix[np.ix_(idxs, idxs)].astype(np.float32)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class PointMazeEnv:
    """Continuous 2-D point maze with pixel observations.

    Agent moves in continuous (x,y) space within free cells.
    Actions: Box(-1, 1, shape=(2,)) — velocity commands.
    Observations: top-down RGB at *img_size*.
    """

    def __init__(
        self,
        layout: str = "corridor",
        img_size: Tuple[int, int] = (64, 64),
        max_speed: float = 0.30,
        goal_radius: float = 0.5,
        max_episode_steps: int = 200,
        step_penalty: float = -0.001,
        goal_reward: float = 1.0,
    ):
        if isinstance(layout, str) and layout in MAZE_LAYOUTS:
            lines = MAZE_LAYOUTS[layout]
        elif isinstance(layout, list):
            lines = layout
        else:
            raise ValueError(f"Unknown layout: {layout}")

        self.grid, self.start_cell, self.goal_cell = parse_grid(lines)
        self.grid_h = len(self.grid)
        self.grid_w = len(self.grid[0])
        self.img_size = img_size
        self.max_speed = max_speed
        self.goal_radius = goal_radius
        self.max_episode_steps = max_episode_steps
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward

        self.start_pos = np.array(
            [self.start_cell[1] + 0.5, self.start_cell[0] + 0.5], dtype=np.float32
        )
        self.goal_pos = np.array(
            [self.goal_cell[1] + 0.5, self.goal_cell[0] + 0.5], dtype=np.float32
        )

        self.x: float = self.start_pos[0]
        self.y: float = self.start_pos[1]
        self.steps: int = 0

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(img_size[0], img_size[1], 3), dtype=np.uint8
        )

        self.geodesic = GeodesicComputer(self.grid)

        # Pre-render static maze image (walls + floor + goal)
        self._bg = self._render_background()

    # ----- public helpers -----

    @property
    def agent_pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    # ----- gym interface -----

    def reset(self, **kwargs):
        seed = kwargs.get("seed", None)
        if seed is not None:
            np.random.seed(seed)
        self.x = self.start_pos[0]
        self.y = self.start_pos[1]
        self.steps = 0
        return self._render_obs(), {}

    def step(self, action, repeat: int = 1):
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(int(repeat)):
            dx = float(action[0]) * self.max_speed
            dy = float(action[1]) * self.max_speed
            nx, ny = self.x + dx, self.y + dy

            if self._is_free(nx, ny):
                self.x, self.y = nx, ny
            elif self._is_free(nx, self.y):
                self.x = nx
            elif self._is_free(self.x, ny):
                self.y = ny

            self.steps += 1
            dist = np.hypot(self.x - self.goal_pos[0], self.y - self.goal_pos[1])
            if dist < self.goal_radius:
                total_reward += self.goal_reward
                terminated = True
            else:
                total_reward += self.step_penalty

            if self.steps >= self.max_episode_steps:
                truncated = True
            if terminated or truncated:
                break

        return self._render_obs(), total_reward, terminated, truncated, {}

    def close(self):
        pass

    # ----- rendering internals -----

    def _is_free(self, x: float, y: float) -> bool:
        c, r = int(x), int(y)
        if r < 0 or r >= self.grid_h or c < 0 or c >= self.grid_w:
            return False
        return self.grid[r][c] == "0"

    def _render_background(self) -> np.ndarray:
        h, w = self.img_size
        img = np.full((h, w, 3), 40, dtype=np.uint8)  # wall colour
        ch = h / self.grid_h
        cw = w / self.grid_w
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                if self.grid[r][c] == "0":
                    y0, y1 = int(r * ch), int((r + 1) * ch)
                    x0, x1 = int(c * cw), int((c + 1) * cw)
                    img[y0:y1, x0:x1] = [220, 220, 220]
        # goal marker
        gx = int(self.goal_pos[0] * cw)
        gy = int(self.goal_pos[1] * ch)
        rad = max(2, int(min(ch, cw) * 0.30))
        _draw_circle(img, gx, gy, rad, (0, 200, 0))
        return img

    def _render_obs(self) -> np.ndarray:
        img = self._bg.copy()
        ch = self.img_size[0] / self.grid_h
        cw = self.img_size[1] / self.grid_w
        ax = int(self.x * cw)
        ay = int(self.y * ch)
        rad = max(2, int(min(ch, cw) * 0.30))
        _draw_circle(img, ax, ay, rad, (220, 30, 30))
        return img


def _draw_circle(img: np.ndarray, cx: int, cy: int, radius: int, colour):
    h, w = img.shape[:2]
    y_lo, y_hi = max(0, cy - radius), min(h, cy + radius + 1)
    x_lo, x_hi = max(0, cx - radius), min(w, cx + radius + 1)
    ys = np.arange(y_lo, y_hi)
    xs = np.arange(x_lo, x_hi)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    img[y_lo:y_hi, x_lo:x_hi][mask] = colour


# ---------------------------------------------------------------------------
# Quick visual sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, name in enumerate(list(MAZE_LAYOUTS.keys())):
        env = PointMazeEnv(layout=name, img_size=(128, 128))
        obs, _ = env.reset()

        # random walk for a few steps so agent moves
        for _ in range(30):
            a = env.action_space.sample()
            obs, r, t, tr, _ = env.step(a)
            if t or tr:
                obs, _ = env.reset()

        axes[idx].imshow(obs)
        axes[idx].set_title(f"{name}  ({env.grid_h}×{env.grid_w})")
        axes[idx].axis("off")
        env.close()

    # procedural maze
    lines = generate_random_maze(11, 11, wall_density=0.18, seed=42)
    env = PointMazeEnv(layout=lines, img_size=(128, 128))
    obs, _ = env.reset()
    for _ in range(30):
        a = env.action_space.sample()
        obs, *_ = env.step(a)
    axes[5].imshow(obs)
    axes[5].set_title("random (11×11)")
    axes[5].axis("off")
    env.close()

    plt.tight_layout()
    plt.show()
