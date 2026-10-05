"""
Swarm-Based Path Planning with Obstacles (PSO)
Name: Muhammad Zain ul Abidin | Roll No: 075
"""
import os
import matplotlib.pyplot as plt
import random
from collections import deque

import numpy as np

# ---------------- Problem configuration ----------------
SEED = 75            # roll number 075
GRID_SIZE = 25       # grid is GRID_SIZE x GRID_SIZE
OBSTACLE_DENSITY = 0.25
MIN_START_GOAL_DIST = 0.6 * GRID_SIZE   # keep the problem non-trivial


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
    ax.legend(loc="upper right")
    if save_as:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        fig.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    grid, start, goal, attempts = generate_problem()
    print(f"Seed: {SEED}")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, obstacles: {int(grid.sum())}")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Instance accepted after {attempts} attempt(s)")
    plot_grid(grid, start, goal, save_as="results/instance.png")