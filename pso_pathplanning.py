"""
Swarm-Based Path Planning with Obstacles (PSO)
Name: Muhammad Zain ul Abidin | Roll No: 075
"""
import os
import random
from collections import deque
import matplotlib
matplotlib.use("Agg")  # save plots to files, don't open windows
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---------------- Problem configuration ----------------
SEED = 75            # roll number 075
GRID_SIZE = 25       # grid is GRID_SIZE x GRID_SIZE
OBSTACLE_DENSITY = 0.25
MIN_START_GOAL_DIST = 0.6 * GRID_SIZE   # keep the problem non-trivial

# ---------------- Path / fitness configuration ----------------
NUM_WAYPOINTS = 8          # intermediate waypoints per particle
COLLISION_PENALTY = 100.0  # cost added per sample point inside an obstacle
SAMPLE_STEP = 0.25         # spacing of collision-check samples along a segment

# ---------------- PSO configuration ----------------
SWARM_SIZE = 100
MAX_ITERATIONS = 400
W_START, W_END = 0.9, 0.4  # inertia weight decays linearly
C1, C2 = 1.5, 1.5          # cognitive and social coefficients
STAGNATION_LIMIT = 100     # stop if global best does not improve this long


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
        t = np.linspace(0.0, 1.0, steps)
        ix = np.clip(np.rint(x0 + t * (x1 - x0)).astype(int), 0, n - 1)
        iy = np.clip(np.rint(y0 + t * (y1 - y0)).astype(int), 0, n - 1)
        collisions += int(grid[iy, ix].sum())
    return collisions


def fitness(position, grid, start, goal):
    """Lower is better: path length + penalty for each colliding sample."""
    path = particle_to_path(position, start, goal)
    length = path_length(path)
    collisions = count_collisions(path, grid)
    return length + COLLISION_PENALTY * collisions, length, collisions


def run_pso(grid, start, goal, rng):
    """Particle Swarm Optimisation over waypoint positions.
    Returns (best_position, best_cost, history_of_best_cost)."""
    n = grid.shape[0]
    dim = 2 * NUM_WAYPOINTS
    v_max = 0.2 * (n - 1)

    # Initialise: half the swarm near the straight line, half uniformly random
    t = np.linspace(0, 1, NUM_WAYPOINTS + 2)[1:-1]
    line = np.column_stack((start[0] + t * (goal[0] - start[0]),
                            start[1] + t * (goal[1] - start[1]))).ravel()
    half = SWARM_SIZE // 2
    pos = np.empty((SWARM_SIZE, dim))
    pos[:half] = line + rng.normal(0, 3.0, (half, dim))
    pos[half:] = rng.uniform(0, n - 1, (SWARM_SIZE - half, dim))
    pos = np.clip(pos, 0, n - 1)
    vel = rng.uniform(-v_max, v_max, (SWARM_SIZE, dim))

    # Personal and global bests
    pbest = pos.copy()
    pbest_cost = np.array([fitness(p, grid, start, goal)[0] for p in pos])
    g = int(np.argmin(pbest_cost))
    gbest = pbest[g].copy()
    gbest_cost = float(pbest_cost[g])
    history = [gbest_cost]
    stagnant = 0

    for it in range(MAX_ITERATIONS):
        w = W_START - (W_START - W_END) * it / (MAX_ITERATIONS - 1)
        r1 = rng.random((SWARM_SIZE, dim))
        r2 = rng.random((SWARM_SIZE, dim))

        # Velocity and position update
        vel = w * vel + C1 * r1 * (pbest - pos) + C2 * r2 * (gbest - pos)
        vel = np.clip(vel, -v_max, v_max)
        pos = np.clip(pos + vel, 0, n - 1)

        # Evaluate (length + collision penalty) and update bests
        improved = False
        for i in range(SWARM_SIZE):
            cost = fitness(pos[i], grid, start, goal)[0]
            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest[i] = pos[i]
                if cost < gbest_cost - 1e-9:
                    gbest_cost = float(cost)
                    gbest = pos[i].copy()
                    improved = True

        history.append(gbest_cost)
        stagnant = 0 if improved else stagnant + 1
        if stagnant >= STAGNATION_LIMIT:
            print(f"Stopped early at iteration {it + 1}: no improvement for "
                  f"{STAGNATION_LIMIT} iterations")
            break

    return gbest, gbest_cost, history


if __name__ == "__main__":
    grid, start, goal, attempts = generate_problem()
    print(f"Seed: {SEED}")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, obstacles: {int(grid.sum())}")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Instance accepted after {attempts} attempt(s)")

    rng = np.random.default_rng(SEED)
    best, best_cost, history = run_pso(grid, start, goal, rng)
    cost, length, collisions = fitness(best, grid, start, goal)
    path = particle_to_path(best, start, goal)

    print(f"Best cost: {cost:.2f}")
    print(f"Path length: {length:.2f}")
    print(f"Collisions: {collisions}")
    print("Obstacle-free path found: " + ("YES" if collisions == 0 else "NO"))

    plot_grid(grid, start, goal, path=path,
              title=f"PSO best path (length {length:.2f}, collisions {collisions})",
              save_as="results/final_path.png")

    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best cost")
    plt.title("PSO convergence")
    plt.grid(True)
    plt.savefig("results/convergence.png", dpi=150, bbox_inches="tight")
    plt.show()