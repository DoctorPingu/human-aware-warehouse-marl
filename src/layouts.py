from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

Point = Tuple[float, float]
Poly = Tuple[Point, ...]
Rect = Tuple[float, float, float, float]
Seg = Tuple[float, float, float, float]

RACK_W = 0.7
RACK_H = 1.3


def rect_poly(x: float, y: float, w: float, d: float) -> Poly:
    return ((x, y), (x + w, y), (x + w, y + d), (x, y + d))


def rot_rect(cx: float, cy: float, length: float, width: float,
             deg: float) -> Poly:
    a = math.radians(deg)
    ux, uy = math.cos(a), math.sin(a)
    vx, vy = -math.sin(a), math.cos(a)
    hl, hw = length / 2.0, width / 2.0
    return (
        (cx - hl * ux - hw * vx, cy - hl * uy - hw * vy),
        (cx + hl * ux - hw * vx, cy + hl * uy - hw * vy),
        (cx + hl * ux + hw * vx, cy + hl * uy + hw * vy),
        (cx - hl * ux + hw * vx, cy - hl * uy + hw * vy),
    )


def polygon_edges(poly: Poly) -> Tuple[Seg, ...]:
    n = len(poly)
    return tuple(
        (poly[i][0], poly[i][1], poly[(i + 1) % n][0], poly[(i + 1) % n][1])
        for i in range(n)
    )


def point_in_poly(px: float, py: float, poly: Sequence[Point]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > py) != (yj > py):
            if px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi:
                inside = not inside
        j = i
    return inside


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    def ccw(a: Point, b: Point, c: Point) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)
            and ccw(p1, p2, p3) != ccw(p1, p2, p4))


def segment_hits_shelf(p1: Point, p2: Point,
                        shelves: Sequence[Poly]) -> int:
    for si, poly in enumerate(shelves):
        for (x1, y1, x2, y2) in polygon_edges(poly):
            if _segments_intersect(p1, p2, (x1, y1), (x2, y2)):
                return si
        if point_in_poly(*p1, poly) or point_in_poly(*p2, poly):
            return si
    return -1


def _point_seg_dist(px: float, py: float,
                     x1: float, y1: float, x2: float, y2: float) -> float:
    vx, vy = x2 - x1, y2 - y1
    l2 = vx * vx + vy * vy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - x1) * vx
                                               + (py - y1) * vy) / l2))
    cx, cy = x1 + t * vx, y1 + t * vy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def hop_clearance(p1: Point, p2: Point, shelves: Sequence[Poly],
                   samples: int = 24) -> float:
    best = float("inf")
    for k in range(samples + 1):
        t = k / samples
        px = p1[0] + (p2[0] - p1[0]) * t
        py = p1[1] + (p2[1] - p1[1]) * t
        for poly in shelves:
            if point_in_poly(px, py, poly):
                return 0.0
            for (x1, y1, x2, y2) in polygon_edges(poly):
                d = _point_seg_dist(px, py, x1, y1, x2, y2)
                if d < best:
                    best = d
    return best


@dataclass(frozen=True)
class Layout:
    name: str
    label: str
    scale: str
    width: float
    height: float

    footprint: Poly
    shelves: Tuple[Poly, ...]

    receiving: Rect
    shipping: Rect
    aisles: Tuple[Rect, ...]
    staging: Rect

    charge_pads: Tuple[Point, ...]
    pick_stations: Tuple[Point, ...]
    pd_points: Tuple[Point, ...]

    robot_starts: Tuple[Point, ...]
    robot_tasks: Tuple[Tuple[Point, ...], ...]
    pickup_idx: Tuple[int, ...]
    delivery_idx: Tuple[int, ...]

    human_routes: Tuple[Tuple[Point, ...], ...]
    human_speeds: Tuple[float, ...]
    human_density: Dict[str, int] = field(
        default_factory=lambda: {"standard": 2, "high": 3})

    demand_dwell: Dict[str, float] = field(
        default_factory=lambda: {"standard": 0.50, "peak": 0.25})

    @property
    def n_robots(self) -> int:
        return len(self.robot_starts)

    @property
    def wall_segs(self) -> Tuple[Seg, ...]:
        return polygon_edges(self.footprint)

    def n_humans(self, density: str = "standard") -> int:
        try:
            return self.human_density[density]
        except KeyError:
            raise KeyError(
                f"Unknown density {density!r} for layout {self.name!r}. "
                f"Valid: {sorted(self.human_density)}") from None

    def dwell(self, demand: str = "standard") -> float:
        try:
            return self.demand_dwell[demand]
        except KeyError:
            raise KeyError(
                f"Unknown demand {demand!r} for layout {self.name!r}. "
                f"Valid: {sorted(self.demand_dwell)}") from None

    def human_start(self, h: int) -> Point:
        return self.human_routes[h][0]


_S1_RACKS = tuple(
    rect_poly(5.4 + 3.7 * k, y, RACK_W, 4.0)
    for k in range(5) for y in (2.6, 9.4)
)

SMALL_1 = Layout(
    name="small_1_parallel_aisle",
    label="Small 1 · Parallel-aisle",
    scale="small",
    width=26.0, height=16.0,
    footprint=rect_poly(0.0, 0.0, 26.0, 16.0),
    shelves=_S1_RACKS,
    receiving=(0.8, 1.4, 2.2, 13.2),
    shipping=(23.0, 1.4, 2.2, 13.2),
    aisles=((4.4, 6.6, 17.2, 2.8),),
    staging=(4.4, 13.6, 8.2, 2.2),
    charge_pads=((5.55, 14.55), (7.55, 14.55), (9.55, 14.55), (21.95, 2.75)),
    pick_stations=((14.95, 14.75), (18.25, 14.75), (21.55, 14.75)),
    pd_points=(),
    robot_starts=((5.55, 14.55), (7.55, 14.55), (9.55, 14.55)),
    robot_tasks=(
        ((5.55, 14.55), (4.60, 14.00), (2.00, 6.00), (6.60, 8.60), (8.20, 10.40), (8.20, 13.80), (8.80, 14.40), (14.95, 14.75)),
        ((7.55, 14.55), (5.00, 14.40), (4.40, 13.60), (2.00, 8.00), (4.60, 14.00), (5.20, 14.40), (18.25, 14.75)),
        ((9.55, 14.55), (5.00, 14.40), (2.00, 10.20), (2.00, 10.00), (4.80, 14.20), (6.80, 14.60), (21.55, 14.75)),
    ),
    pickup_idx=(2, 3, 3),
    delivery_idx=(7, 6, 6),
    human_routes=(
        ((11.23, 14.53), (5.00, 14.40), (2.00, 12.00), (5.00, 8.40), (21.20, 7.60), (23.00, 5.80), (24.20, 3.00), (21.60, 7.20), (19.20, 9.00), (18.00, 14.00), (17.40, 14.40), (14.95, 14.75)),
        ((12.13, 14.53), (21.20, 14.40), (24.20, 12.50), (21.60, 8.80), (21.20, 8.40), (5.00, 7.60), (2.00, 4.00), (5.40, 8.00), (13.80, 8.40), (15.40, 10.00), (15.60, 13.80), (16.20, 14.40), (18.25, 14.75)),
        ((2.00, 3.00), (4.80, 7.40), (6.80, 7.80), (21.20, 8.40), (24.20, 10.00), (21.60, 14.00), (21.20, 14.40), (9.55, 14.55), (8.60, 14.20), (7.20, 12.60), (7.00, 9.00), (5.60, 7.60), (4.80, 7.40)),
        ((24.20, 6.00), (20.20, 8.00), (5.00, 8.40), (3.00, 11.20), (2.00, 13.50), (4.80, 8.60), (5.80, 8.40), (12.00, 8.00), (21.20, 7.60)),
        ((2.00, 2.00), (4.80, 7.40), (5.80, 7.60), (21.20, 8.40), (22.00, 9.20), (24.20, 14.00), (12.00, 15.20), (8.60, 14.20), (7.20, 12.60), (7.00, 9.00), (5.60, 7.60), (4.80, 7.40)),
    ),
    human_speeds=(1.0, 0.85, 0.9, 0.95, 1.05),
    human_density={"standard": 2, "high": 3, "stress": 5},
)


_S2_RACKS = tuple(
    rect_poly(2.6 + 3.6 * k, 8.4, RACK_W, 5.0) for k in range(3)
) + tuple(
    rect_poly(15.0, 8.8 + 2.7 * j, 7.4, RACK_W) for j in range(3)
)

SMALL_2 = Layout(
    name="small_2_l_shaped",
    label="Small 2 · L-shaped",
    scale="small",
    width=26.0, height=16.0,
    footprint=((0.0, 0.0), (13.0, 0.0), (13.0, 7.0),
               (26.0, 7.0), (26.0, 16.0), (0.0, 16.0)),
    shelves=_S2_RACKS,
    receiving=(0.8, 0.9, 3.6, 3.0),
    shipping=(23.6, 8.0, 1.8, 7.0),
    aisles=((0.8, 5.2, 11.6, 2.6),),
    staging=(1.0, 13.8, 8.4, 2.2),
    charge_pads=((2.15, 14.75), (4.15, 14.75), (6.15, 14.75), (11.15, 2.15)),
    pick_stations=((16.15, 15.45), (19.15, 15.45), (22.15, 15.45)),
    pd_points=(),
    robot_starts=((2.15, 14.75), (4.15, 14.75), (6.15, 14.75)),
    robot_tasks=(
        ((2.15, 14.75), (1.60, 13.40), (1.60, 4.40), (2.60, 2.40), (11.20, 7.80), (14.00, 13.20), (14.00, 15.20), (14.60, 15.80), (15.60, 15.80), (16.15, 15.45), (15.60, 15.80), (13.00, 15.80)),
        ((4.15, 14.75), (4.20, 13.40), (4.20, 5.80), (2.60, 2.40), (11.20, 7.80), (14.00, 13.20), (14.00, 15.20), (14.60, 15.80), (18.60, 15.80), (19.15, 15.45), (18.60, 15.80), (6.20, 15.80)),
        ((6.15, 14.75), (5.20, 13.80), (3.40, 4.00), (2.60, 2.40), (11.20, 7.80), (14.00, 13.20), (14.00, 15.20), (14.60, 15.80), (21.60, 15.80), (22.15, 15.45), (21.60, 15.80), (8.20, 15.80)),
    ),
    pickup_idx=(3, 3, 3),
    delivery_idx=(9, 9, 9),
    human_routes=(
        ((7.93, 14.73), (7.80, 8.00), (2.60, 2.40), (11.20, 7.80), (23.00, 8.00), (24.50, 9.00), (22.80, 10.40), (14.60, 10.60), (10.80, 14.40), (6.15, 14.75)),
        ((8.83, 14.73), (15.20, 13.20), (22.80, 13.20), (24.50, 14.00), (23.20, 11.00), (22.80, 10.60), (13.80, 10.40), (10.80, 7.40), (2.00, 6.50), (3.80, 7.60), (4.20, 8.00), (4.20, 14.60), (4.15, 14.75)),
        ((1.50, 11.00), (1.60, 8.00), (2.40, 7.40), (11.75, 6.50), (5.60, 7.60), (4.20, 10.20), (4.20, 13.80), (3.60, 14.40), (2.15, 14.75), (1.60, 13.40)),
        ((11.75, 2.00), (11.80, 7.80), (14.20, 10.00), (14.60, 10.40), (22.80, 10.60), (24.50, 12.00), (22.60, 13.20), (12.00, 13.20), (10.80, 14.40), (2.20, 14.40), (1.50, 14.00), (2.60, 14.40), (3.60, 14.40), (5.20, 12.80), (5.20, 8.00)),
        ((6.00, 4.00), (11.20, 7.80), (14.40, 13.00), (15.40, 13.20), (22.80, 13.20), (23.40, 14.00), (23.40, 15.20), (23.20, 15.40), (23.00, 15.50), (22.80, 15.80), (14.60, 15.80), (2.20, 14.40), (1.60, 13.60), (1.50, 9.00), (1.80, 7.80)),
    ),
    human_speeds=(1.0, 0.85, 0.9, 0.95, 1.05),
    human_density={"standard": 2, "high": 3, "stress": 5},
)


_M1_RACKS = tuple(
    rect_poly(5.0 + 3.6 * k, 2.6, RACK_W, 13.0) for k in range(3)
) + tuple(
    rect_poly(19.4 + 3.6 * k, 2.6, RACK_W, 13.0) for k in range(3)
)

MEDIUM_1 = Layout(
    name="medium_1_twin_block",
    label="Medium 1 · Twin-block",
    scale="medium",
    width=32.0, height=20.0,
    footprint=rect_poly(0.0, 0.0, 32.0, 20.0),
    shelves=_M1_RACKS,
    receiving=(0.8, 1.6, 2.2, 13.6),
    shipping=(29.0, 1.6, 2.2, 13.6),
    aisles=((13.8, 1.6, 4.4, 17.0),),
    staging=(4.6, 16.2, 9.2, 2.8),
    charge_pads=((5.75, 17.25), (8.25, 17.25), (10.75, 17.25), (27.35, 3.15)),
    pick_stations=((20.35, 17.15), (23.55, 17.15), (26.75, 17.15)),
    pd_points=((16.00, 5.40), (16.00, 12.40)),
    robot_starts=((5.75, 17.25), (8.25, 17.25), (10.75, 17.25)),
    robot_tasks=(
        ((5.75, 17.25), (4.00, 16.00), (1.90, 6.00), (4.20, 16.20), (4.80, 16.60), (20.35, 17.15)),
        ((8.25, 17.25), (4.40, 16.40), (4.00, 15.80), (1.90, 10.00), (4.20, 16.20), (4.80, 16.60), (23.55, 17.15)),
        ((10.75, 17.25), (4.60, 16.60), (1.90, 13.00), (4.60, 16.60), (26.75, 17.15)),
    ),
    pickup_idx=(2, 3, 2),
    delivery_idx=(5, 6, 4),
    human_routes=(
        ((12.43, 17.43), (4.60, 16.60), (4.00, 15.80), (1.90, 4.00), (4.20, 16.20), (4.80, 16.60), (29.00, 16.60), (30.20, 16.00), (27.00, 16.60), (10.75, 17.25)),
        ((13.33, 17.43), (27.60, 16.60), (28.20, 16.00), (30.20, 4.00), (28.20, 16.00), (27.60, 16.60), (3.00, 16.60), (1.90, 16.00), (8.25, 17.25)),
        ((1.90, 8.00), (4.20, 16.20), (4.80, 16.60), (27.60, 16.60), (28.60, 15.60), (30.20, 12.00), (28.20, 16.00), (27.60, 16.60), (5.75, 17.25), (4.00, 16.00)),
        ((16.00, 3.00), (19.40, 1.60), (27.60, 1.60), (28.20, 2.20), (30.20, 10.00), (28.20, 16.00), (27.60, 16.60), (16.00, 18.50)),
        ((1.90, 2.50), (5.20, 1.60), (13.20, 1.60), (13.80, 2.20), (16.00, 14.00), (19.20, 16.60), (28.50, 18.50), (18.80, 16.40), (14.60, 8.40), (13.80, 2.20), (13.20, 1.60), (3.40, 1.60)),
    ),
    human_speeds=(1.0, 0.85, 0.9, 0.95, 1.05),
    human_density={"standard": 2, "high": 3, "stress": 5},
)


_M2_RACKS = tuple(
    rot_rect(cx, 3.8 + 5.0 * k, 10.0, RACK_W, sign * 28.0)
    for k in range(3)
    for cx, sign in ((8.0, 1.0), (24.0, -1.0))
)

MEDIUM_2 = Layout(
    name="medium_2_fishbone",
    label="Medium 2 · Fishbone",
    scale="medium",
    width=32.0, height=20.0,
    footprint=rect_poly(0.0, 0.0, 32.0, 20.0),
    shelves=_M2_RACKS,
    receiving=(0.8, 1.8, 2.2, 13.4),
    shipping=(29.0, 1.8, 2.2, 13.4),
    aisles=((14.3, 1.8, 3.4, 14.4),),
    staging=(1.0, 16.6, 9.4, 2.8),
    charge_pads=((2.15, 17.65), (4.65, 17.65), (7.15, 17.65), (27.35, 3.35)),
    pick_stations=((21.15, 17.55), (24.35, 17.55), (27.55, 17.55)),
    pd_points=((16.00, 15.20),),
    robot_starts=((2.15, 17.65), (4.65, 17.65), (7.15, 17.65)),
    robot_tasks=(
        ((2.15, 17.65), (1.90, 6.00), (3.00, 7.80), (19.80, 17.40), (21.15, 17.55)),
        ((4.65, 17.65), (1.80, 10.20), (1.90, 10.00), (9.00, 10.80), (19.60, 17.40), (24.35, 17.55)),
        ((7.15, 17.65), (1.90, 14.00), (12.20, 17.40), (27.55, 17.55)),
    ),
    pickup_idx=(1, 2, 1),
    delivery_idx=(4, 5, 3),
    human_routes=(
        ((8.83, 17.83), (2.60, 12.40), (2.40, 11.00), (1.90, 4.00), (13.40, 10.20), (19.20, 17.20), (19.60, 17.40), (30.20, 15.00), (19.60, 17.40), (7.15, 17.65)),
        ((9.73, 17.83), (12.60, 17.40), (18.40, 11.20), (18.60, 10.20), (30.20, 4.00), (18.60, 10.20), (12.80, 17.20), (12.40, 17.40), (1.90, 15.00), (4.65, 17.65)),
        ((1.90, 9.00), (12.20, 12.40), (22.80, 12.40), (30.20, 9.00), (12.20, 17.40), (2.15, 17.65)),
        ((16.00, 2.50), (19.00, 7.00), (19.40, 7.40), (25.00, 6.80), (28.00, 5.20), (28.60, 5.20), (29.20, 5.80), (29.60, 6.80), (30.20, 12.00), (19.60, 17.60), (13.35, 18.50)),
        ((1.90, 2.50), (13.60, 10.40), (18.64, 15.00), (18.40, 16.20), (19.00, 17.00), (19.40, 17.40), (28.00, 18.50), (19.40, 17.40), (13.60, 11.20), (13.40, 10.20)),
    ),
    human_speeds=(1.0, 0.85, 0.9, 0.95, 1.05),
    human_density={"standard": 2, "high": 3, "stress": 5},
)


_L1_RACKS = tuple(
    rect_poly(6.0 + 4.2 * k, y, RACK_W, d)
    for k in range(8)
    for y, d in ((2.6, 4.6), (10.2, 4.4), (17.6, 4.6))
)

LARGE_1 = Layout(
    name="large_1_fulfilment",
    label="Large 1 · Fulfilment centre",
    scale="large",
    width=42.0, height=26.0,
    footprint=rect_poly(0.0, 0.0, 42.0, 26.0),
    shelves=_L1_RACKS,
    receiving=(0.8, 2.0, 2.4, 19.0),
    shipping=(38.8, 2.0, 2.4, 19.0),
    aisles=((4.4, 7.4, 33.2, 2.6), (4.4, 14.8, 33.2, 2.6)),
    staging=(4.4, 22.6, 10.0, 2.8),
    charge_pads=((5.55, 23.65), (7.95, 23.65), (10.35, 23.65),
                 (12.75, 23.65), (36.55, 3.75)),
    pick_stations=((20.35, 23.45), (24.55, 23.45),
                   (28.75, 23.45), (32.95, 23.45)),
    pd_points=((21.00, 8.80), (21.00, 16.20)),
    robot_starts=((5.55, 23.65), (7.95, 23.65),
                  (10.35, 23.65), (12.75, 23.65)),
    robot_tasks=(
        ((5.55, 23.65), (3.40, 17.40), (2.00, 4.00), (6.20, 8.80), (7.20, 9.40), (9.20, 11.60), (9.20, 15.00), (11.40, 16.60), (15.40, 16.60), (16.40, 17.60), (17.80, 22.80), (18.40, 23.20), (20.35, 23.45)),
        ((7.95, 23.65), (7.60, 17.20), (6.00, 15.60), (5.40, 15.40), (2.00, 9.00), (5.40, 15.40), (12.00, 16.60), (19.60, 16.60), (20.40, 17.40), (22.00, 22.80), (22.60, 23.20), (24.55, 23.45)),
        ((10.35, 23.65), (5.60, 23.20), (4.60, 21.80), (2.00, 16.00), (5.20, 22.80), (5.80, 23.20), (28.75, 23.45)),
        ((12.75, 23.65), (5.60, 23.20), (2.00, 21.00), (5.80, 23.20), (32.95, 23.45)),
    ),
    pickup_idx=(2, 4, 3, 2),
    delivery_idx=(12, 11, 6, 4),
    human_routes=(
        ((15.23, 23.83), (13.60, 22.80), (11.80, 17.20), (10.20, 15.60), (9.60, 15.40), (7.60, 11.60), (7.60, 9.80), (6.00, 8.20), (5.40, 8.00), (2.00, 6.00), (5.80, 8.20), (28.00, 9.20), (30.20, 11.40), (30.20, 15.00), (32.20, 16.40), (36.40, 16.60), (40.20, 20.00), (36.40, 23.20), (10.35, 23.65)),
        ((16.18, 23.83), (19.60, 23.20), (21.80, 21.00), (21.80, 17.20), (24.00, 15.60), (32.20, 15.60), (33.00, 14.80), (34.60, 9.60), (40.20, 6.00), (36.20, 8.20), (14.00, 9.20), (11.80, 12.20), (11.80, 15.00), (10.80, 16.00), (5.60, 16.60), (2.00, 20.00), (5.60, 23.20), (7.95, 23.65)),
        ((17.13, 23.83), (13.80, 23.00), (11.80, 19.20), (11.80, 17.20), (10.40, 15.80), (5.60, 15.60), (2.00, 12.00), (5.80, 9.20), (36.40, 9.20), (40.20, 12.00), (36.40, 15.60), (18.20, 16.60), (16.00, 19.60), (16.00, 22.60), (15.40, 23.20), (12.75, 23.65)),
        ((2.00, 16.00), (37.20, 16.60), (40.20, 18.00), (36.80, 22.80), (36.40, 23.20), (5.55, 23.65)),
        ((21.00, 8.70), (36.40, 9.20), (39.60, 12.60), (40.20, 14.00), (36.20, 15.60), (21.00, 16.10)),
        ((2.00, 24.50), (37.00, 24.50), (40.20, 3.00), (36.80, 7.80), (36.20, 8.40), (14.00, 9.20), (11.80, 12.20), (11.80, 15.00), (10.20, 16.60), (9.60, 16.80), (7.60, 20.60), (7.60, 22.60), (7.00, 23.20)),
        ((12.00, 16.10), (15.40, 15.60), (17.60, 13.40), (17.60, 9.80), (18.40, 9.20), (30.00, 8.70), (5.60, 8.20), (2.00, 3.20), (2.00, 3.00), (5.40, 8.00), (7.40, 9.60), (9.20, 12.80), (9.20, 15.00), (10.00, 15.60)),
    ),
    human_speeds=(1.0, 0.85, 0.9, 0.95, 1.05, 0.88, 0.98),
    human_density={"standard": 3, "high": 4, "stress": 7},
)


_L2_ASRS = tuple(
    rect_poly(3.2, 1.6 + 2.4 * r, 34.0, 1.0) for r in range(2)
)
_L2_PODS = tuple(
    rect_poly(4.0 + 9.0 * c + 2.2 * i, 9.4 + 1.9 * j, 1.4, 1.2)
    for c in range(4) for i in range(3) for j in range(4)
)

LARGE_2 = Layout(
    name="large_2_robotic_grid",
    label="Large 2 · Robotic grid (AS/RS)",
    scale="large",
    width=42.0, height=26.0,
    footprint=rect_poly(0.0, 0.0, 42.0, 26.0),
    shelves=_L2_ASRS + _L2_PODS,
    receiving=(0.8, 9.0, 2.4, 11.0),
    shipping=(38.8, 9.0, 2.4, 11.0),
    aisles=((3.4, 6.4, 35.0, 1.9), (3.4, 17.6, 35.0, 1.8)),
    staging=(3.4, 21.6, 10.4, 2.8),
    charge_pads=((4.55, 22.65), (6.95, 22.65), (9.35, 22.65),
                 (11.75, 22.65), (36.55, 22.65)),
    pick_stations=((18.75, 20.55), (22.75, 20.55), (26.75, 20.55),
                   (30.75, 20.55), (34.75, 20.55)),
    pd_points=((21.20, 8.10),),
    robot_starts=((4.55, 22.65), (6.95, 22.65),
                  (9.35, 22.65), (11.75, 22.65)),
    robot_tasks=(
        ((4.55, 22.65), (2.40, 13.40), (2.00, 7.30), (10.20, 8.40), (12.00, 10.80), (12.00, 16.60), (13.80, 18.40), (18.75, 20.55)),
        ((6.95, 22.65), (11.80, 14.80), (12.00, 9.00), (12.80, 8.40), (20.00, 7.30), (21.20, 17.20), (22.75, 20.55)),
        ((9.35, 22.65), (28.40, 17.00), (29.00, 16.20), (30.00, 9.00), (30.80, 8.40), (40.20, 7.30), (37.60, 16.80), (37.20, 17.20), (26.75, 20.55)),
        ((11.75, 22.65), (40.20, 18.50), (30.75, 20.55)),
    ),
    pickup_idx=(2, 4, 5, 1),
    delivery_idx=(7, 6, 8, 2),
    human_routes=(
        ((14.43, 22.83), (3.40, 17.00), (3.00, 16.60), (2.00, 7.30), (40.20, 7.30), (21.60, 8.40), (20.80, 9.40), (19.80, 16.60), (14.20, 21.60), (11.75, 22.65)),
        ((15.38, 22.83), (40.20, 18.50), (2.00, 18.50), (9.35, 22.65)),
        ((16.33, 22.83), (19.80, 16.40), (20.00, 7.30), (28.20, 8.40), (30.00, 10.80), (30.00, 16.60), (30.60, 17.20), (40.20, 20.00), (6.95, 22.65)),
        ((2.00, 18.50), (28.20, 17.20), (30.00, 15.40), (30.00, 9.00), (30.80, 8.40), (40.20, 7.30), (12.60, 8.40), (10.80, 10.80), (10.80, 16.60), (4.55, 22.65)),
        ((20.00, 7.30), (37.20, 8.40), (38.60, 10.40), (40.20, 14.00), (37.20, 17.20), (20.00, 18.50)),
        ((2.00, 20.00), (37.00, 24.50), (40.20, 10.00), (37.60, 16.80), (37.20, 17.20)),
        ((11.40, 18.50), (19.20, 17.20), (21.00, 15.40), (21.00, 9.00), (21.80, 8.40), (29.40, 7.30), (3.60, 8.40), (2.00, 10.00), (3.00, 16.60), (3.60, 17.20)),
    ),
    human_speeds=(1.0, 0.85, 0.9, 0.95, 1.05, 0.88, 0.98),
    human_density={"standard": 3, "high": 4, "stress": 7},
)


LAYOUTS: Dict[str, Layout] = {
    lay.name: lay for lay in (
        SMALL_1, SMALL_2, MEDIUM_1, MEDIUM_2, LARGE_1, LARGE_2)
}

DEMAND_LEVELS = ("standard", "peak")
DENSITY_LEVELS = ("standard", "high")
STRESS_LEVEL = "stress"

ALIASES: Dict[str, str] = {
    "small1": "small_1_parallel_aisle",
    "small2": "small_2_l_shaped",
    "medium1": "medium_1_twin_block",
    "medium2": "medium_2_fishbone",
    "large1": "large_1_fulfilment",
    "large2": "large_2_robotic_grid",
}


def get_layout(name: str) -> Layout:
    key = ALIASES.get(name, name)
    try:
        return LAYOUTS[key]
    except KeyError:
        raise KeyError(
            f"Unknown layout {name!r}. Valid names: "
            f"{sorted(LAYOUTS)}; aliases: {sorted(ALIASES)}") from None


def all_conditions() -> Tuple[Tuple[str, str, str], ...]:
    return tuple(
        (name, demand, density)
        for name in LAYOUTS
        for demand in DEMAND_LEVELS
        for density in DENSITY_LEVELS
    )


def validate(layout: Layout, clearance: float = 0.05) -> list[str]:
    problems: list[str] = []
    fp = layout.footprint

    def check_point(px: float, py: float, what: str) -> None:
        if not point_in_poly(px, py, fp):
            problems.append(f"{what} ({px}, {py}) is outside the footprint")
            return
        for si, shelf in enumerate(layout.shelves):
            if point_in_poly(px, py, shelf):
                problems.append(f"{what} ({px}, {py}) is inside shelf {si}")
                return

    min_clearance = 0.45

    def check_hops(route: Sequence[Point], who: str) -> None:
        n = len(route)
        for w in range(n):
            p1, p2 = route[w], route[(w + 1) % n]
            steps = max(2, int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 0.05))
            for t in range(steps + 1):
                f = t / steps
                qx, qy = p1[0] + f * (p2[0] - p1[0]), p1[1] + f * (p2[1] - p1[1])
                if not point_in_poly(qx, qy, fp):
                    problems.append(f"{who} hop {w}->{(w + 1) % n} ({p1} to {p2}) "
                                    f"passes outside the building near "
                                    f"({qx:.1f}, {qy:.1f})")
                    break
            clr = hop_clearance(p1, p2, layout.shelves)
            if clr < min_clearance:
                problems.append(
                    f"{who} hop {w}->{(w + 1) % n} ({p1} to {p2}) "
                    f"clears racks by only {clr:.2f} m, "
                    f"need >= {min_clearance:.2f} m")

    for i, route in enumerate(layout.robot_tasks):
        if len(route) < 2:
            problems.append(f"robot {i} route has fewer than 2 waypoints")
        for w, (px, py) in enumerate(route):
            check_point(px, py, f"robot {i} waypoint {w}")
        check_hops(route, f"robot {i}")

    for h, route in enumerate(layout.human_routes):
        for w, (px, py) in enumerate(route):
            check_point(px, py, f"human {h} waypoint {w}")
        check_hops(route, f"human {h}")

    for m, (px, py) in enumerate(layout.pick_stations):
        check_point(px, py, f"pick station {m}")
    for m, (px, py) in enumerate(layout.charge_pads):
        check_point(px, py, f"charging pad {m}")
    for m, (px, py) in enumerate(layout.pd_points):
        check_point(px, py, f"P&D point {m}")

    n_r = layout.n_robots
    for tup, nm in ((layout.pickup_idx, "pickup_idx"),
                    (layout.delivery_idx, "delivery_idx")):
        if len(tup) != n_r:
            problems.append(f"{nm} has {len(tup)} entries, expected {n_r}")
    for i in range(min(n_r, len(layout.pickup_idx), len(layout.delivery_idx))):
        route_len = len(layout.robot_tasks[i])
        if not 0 <= layout.pickup_idx[i] < route_len:
            problems.append(f"robot {i} pickup_idx out of range")
        if not 0 <= layout.delivery_idx[i] < route_len:
            problems.append(f"robot {i} delivery_idx out of range")
        if layout.pickup_idx[i] >= layout.delivery_idx[i]:
            problems.append(
                f"robot {i} pickup_idx must come before delivery_idx")

    need = max(layout.human_density.values())
    if len(layout.human_routes) < need:
        problems.append(
            f"only {len(layout.human_routes)} human routes, "
            f"density 'high' needs {need}")
    if len(layout.human_speeds) < need:
        problems.append(
            f"only {len(layout.human_speeds)} human speeds, needs {need}")

    for si, shelf in enumerate(layout.shelves):
        for (px, py) in shelf:
            if not point_in_poly(px, py, fp):
                problems.append(f"shelf {si} corner ({px}, {py}) is outside "
                                f"the footprint")
                break

    return problems


if __name__ == "__main__":
    print("Warehouse Layout Definitions")
    print("=" * 64)
    ok = True
    for nm, lay in LAYOUTS.items():
        issues = validate(lay)
        status = "OK " if not issues else "FAIL"
        if issues:
            ok = False
        print(f"[{status}] {lay.label:<34} {lay.width:>4.0f} x "
              f"{lay.height:<4.0f} m  "
              f"{len(lay.shelves):>3} shelves  "
              f"{lay.n_robots} robots  "
              f"{lay.n_humans('standard')}/{lay.n_humans('high')}/{lay.n_humans('stress')} humans (std/high/stress)")
        for p in issues:
            print(f"         - {p}")
    print("=" * 64)
    print(f"Experimental conditions: {len(all_conditions())}")
    print("All layouts valid." if ok else "Some layouts have problems.")
