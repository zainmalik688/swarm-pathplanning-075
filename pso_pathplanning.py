"""
Swarm-Based Path Planning with Obstacles (PSO)
Name: Muhammad Zain ul Abidin | Roll No: 075
"""
import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---------------- Problem configuration ----------------
SEED = 75            # roll number 075
GRID_SIZE = 25       # grid is GRID_SIZE x GRID_SIZE
OBSTACLE_DENSITY = 0.25
MIN_START_GOAL_DIST = 0.6 * GRID_SIZE   # keep the problem non-trivial

# ---------------- PSO / path configuration ----------------
NUM_WAYPOINTS = 8          # intermediate waypoints per particle
COLLISION_PENALTY = 100.0  # cost added per sample point inside an obstacle
SAMPLE_STEP = 0.25         # spacing of collision-check samples along a segment


def bfs_reachable(grid, start, goal):
    """Check that goal can be reached from start (4-connected moves).
    Used only to validate the generated instance, never for planning."""
    n = grid.shape[0]
    queue = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and grid[ny, nx] == 0 and (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append((nx, ny))
    return False


def generate_problem(seed=SEED, n=GRID_SIZE, density=OBSTACLE_DENSITY):
    """Generate obstacles, start and goal from the seed.
    grid[y, x] == 1 means blocked. Points are (x, y)."""
    random.seed(seed)
    attempts = 0
    while True:
        attempts += 1
        grid = np.zeros((n, n), dtype=int)
        for y in range(n):
            for x in range(n):
                if random.random() < density:
                    grid[y, x] = 1

        free = [(x, y) for y in range(n) for x in range(n) if grid[y, x] == 0]
        start = random.choice(free)
        goal = random.choice(free)

        dist = np.hypot(start[0] - goal[0], start[1] - goal[1])
        if dist < MIN_START_GOAL_DIST:
            continue
        if bfs_reachable(grid, start, goal):
            return grid, start, goal, attempts


def plot_grid(grid, start, goal, path=None, title="Generated problem (seed 75)",
              save_as=None):
    """Draw obstacles, start, goal and (optionally) a path. Points are (x, y)."""
    n = grid.shape[0]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(grid, cmap="Greys", origin="lower", vmin=0, vmax=1.5)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    if path is not None:
        xs, ys = zip(*path)
        ax.plot(xs, ys, "-o", color="tab:blue", markersize=4, linewidth=2, label="Path")
    ax.scatter(*start, c="green", s=120, marker="s", label="Start", zorder=3)
    ax.scatter(*goal, c="red", s=150, marker="*", label="Goal", zorder=3)

    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="dimgray", label="Obstacle"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1))
    if save_as:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        fig.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.show()


def particle_to_path(position, start, goal):
    """A particle is a flat array [x1, y1, x2, y2, ...] of waypoints.
    The full path is start -> waypoints -> goal."""
    waypoints = position.reshape(-1, 2)
    return [tuple(start)] + [tuple(w) for w in waypoints] + [tuple(goal)]


def path_length(path):
    """Total Euclidean length of the polyline."""
    return sum(np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
               for i in range(len(path) - 1))


def count_collisions(path, grid):
    """Sample points along every segment and count how many fall inside a
    blocked cell. Cell (ix, iy) is centred on integer coordinates."""
    n = grid.shape[0]
    collisions = 0
    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        seg_len = np.hypot(x1 - x0, y1 - y0)
        steps = max(2, int(np.ceil(seg_len / SAMPLE_STEP)) + 1)
        for t in np.linspace(0.0, 1.0, steps):
            ix = int(np.clip(round(x0 + t * (x1 - x0)), 0, n - 1))
            iy = int(np.clip(round(y0 + t * (y1 - y0)), 0, n - 1))
            if grid[iy, ix] == 1:
                collisions += 1
    return collisions


def fitness(position, grid, start, goal):
    """Lower is better: path length + penalty for each colliding sample."""
    path = particle_to_path(position, start, goal)
    length = path_length(path)
    collisions = count_collisions(path, grid)
    return length + COLLISION_PENALTY * collisions, length, collisions


if __name__ == "__main__":
    grid, start, goal, attempts = generate_problem()
    print(f"Seed: {SEED}")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, obstacles: {int(grid.sum())}")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Instance accepted after {attempts} attempt(s)")

    # Test: evenly spaced waypoints on the straight line start -> goal
    t = np.linspace(0, 1, NUM_WAYPOINTS + 2)[1:-1]
    straight = np.column_stack((start[0] + t * (goal[0] - start[0]),
                                start[1] + t * (goal[1] - start[1]))).ravel()
    cost, length, collisions = fitness(straight, grid, start, goal)
    print(f"Straight-line test -> cost: {cost:.2f}, length: {length:.2f}, "
          f"collisions: {collisions}")
    plot_grid(grid, start, goal, path=particle_to_path(straight, start, goal),
              title="Straight-line test path (not optimised)")