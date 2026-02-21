import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class QuadNode:
    # dom(v): axis-aligned rectangle Omega_v = [xmin, xmax] x [ymin, ymax]
    bounds: Tuple[float, float, float, float] # (xmin, ymin, xmax, ymax)
    depth: int
    children: List["QuadNode"] = field(default_factory=list)

    _idx: Optional[np.ndarray] = None
    _visited: bool = False

def _laplace(rng: np.random.Generator, lam: float) -> float:
    # Lap(λ): Laplace noise with scale λ
    return float(rng.laplace(loc=0.0, scale=lam))

def _split_rect(bounds: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
    xmin, ymin, xmax, ymax = bounds
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    return [
        (xmin, ymin, xmid, ymid),  # SW
        (xmid, ymin, xmax, ymid),  # SE
        (xmin, ymid, xmid, ymax),  # NW
        (xmid, ymid, xmax, ymax),  # NE
    ]

def _assign_points_to_children(points: np.ndarray, idx: np.ndarray, bounds: Tuple[float, float, float, float]):
    xmin, ymin, xmax, ymax = bounds
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0

    pts = points[idx]
    x = pts[:, 0]
    y = pts[:, 1]

    left = x < xmid
    bottom = y < ymid

    return [
        idx[left & bottom],
        idx[~left & bottom],
        idx[left & ~bottom],
        idx[~left & ~bottom],
    ]



def privtree_grid(points: np.ndarray,
                             omega_bounds: Tuple[float, float, float, float],
                             lam: float,
                             theta: float,
                             delta: float,
                             seed: Optional[int] = None) -> np.ndarray:
    """
    Run Algorithm 2 (Structure Generation) and retuen Adaptive Grid.

    """
    points = np.asarray(points, dtype=float)
    rng = np.random.default_rng()

    # 1-2: initialize quadtree T with root v1
    root = QuadNode(bounds=omega_bounds, depth=0, _idx=np.arange(points.shape[0]), _visited=False)

    # 3: while exists an unvisited node v
    unvisited: List[QuadNode] = [root]
    while unvisited:
        v = unvisited.pop()  # pick some unvisited node v
        v._visited = True

        # 5: compute biased point count
        c_v = int(v._idx.size)
        b_v = max(c_v - v.depth * delta, theta - delta)

        # 7: compute a noisy version of b(v)
        b_hat = b_v + _laplace(rng, lam)

        # 8-10: if b̂(v) > θ then split v
        if b_hat > theta:
            child_bounds = _split_rect(v.bounds)
            child_idxs = _assign_points_to_children(points, v._idx, v.bounds)

            v.children = []
            for cb, ci in zip(child_bounds, child_idxs):
                child = QuadNode(bounds=cb, depth=v.depth + 1, _idx=ci, _visited=False)
                v.children.append(child)
                unvisited.append(child)

    # 11: Collect leaves (Adaptive Grid) -> cells
    cells = []
    stack = [root]
    while stack:
        node = stack.pop()
        if not node.children:
            cells.append(node.bounds)
        else:
            stack.extend(node.children)

    return np.array(cells, dtype=float)
