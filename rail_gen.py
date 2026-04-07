from dataclasses import dataclass, field
import math
import os
import numpy as np
from pyglm import glm
from xml.etree.ElementTree import Element, SubElement, tostring, indent


@dataclass
class RailSpec:
    """Cross-section specification for a rail type.

    Attributes:
        name: Human-readable rail standard name.
        profile: Cross-section vertices as (x_mm, y_mm) pairs.
        gauge: Track gauge in meters.
    """

    name: str
    profile: list[tuple[float, float]]
    gauge: float


# Data from: https://www.jfe-steel.co.jp/products/katakou/rail/rail_a.html
def _make_jis60kg() -> RailSpec:
    B, C, G, F, E, D, A = 145.0, 65.0, 16.5, 30.1, 94.9, 49.0, 174.0
    return RailSpec(
        name="JIS 60kg N",
        gauge=1.067,
        profile=[
            (-B/2,   0),       (-B/2,   F*0.3),    # left base up
            (-G/2-8, F),       (-G/2,   F),        # left foot taper
            (-G/2,   F+E),     (-C/2,   F+E),      # left web & head
            (-C/2,   A),       ( C/2,   A),        # head top
            ( C/2,   F+E),     ( G/2,   F+E),      # right head & web
            ( G/2,   F),       ( G/2+8, F),        # right foot taper
            ( B/2,   F*0.3),   ( B/2,   0),        # right base down
        ],  # closed: (B/2,0) → (-B/2,0) = base bottom
    )  # fmt: skip


JIS_60KG = _make_jis60kg()


@dataclass
class TerrainSpec:
    """Configuration for terrain heightfield generation."""
    margin: float = 3.0          # extra extent beyond network bounds (m)
    resolution: float = 0.2      # grid cell size (m)
    base_depth: float = 0.15     # depth below rail level (m)
    slope_width: float = 1.2     # lateral distance of slope from rail to base (m)
    flat_radius: float = 0.8     # half-width of flat zone around centerline (m)
    noise_amplitude: float = 0.2 # noise height std-dev (m)
    noise_octaves: int = 20      # Perlin noise octaves (higher = more detail)


# --- Geometry helpers ---


def _obb_overlap_2d(cx1, cy1, a1, cx2, cy2, a2, hx, hy):
    """Check if two 2D oriented boxes overlap using the Separating Axis Theorem.

    Both boxes share the same half-extents (*hx*, *hy*).
    """
    cos1, sin1 = math.cos(a1), math.sin(a1)
    cos2, sin2 = math.cos(a2), math.sin(a2)
    # The 4 separating axes: 2 edge normals per box
    axes = [(cos1, sin1), (-sin1, cos1), (cos2, sin2), (-sin2, cos2)]
    dx, dy = cx2 - cx1, cy2 - cy1
    for ax, ay in axes:
        # Project the center-to-center vector onto the axis
        d = abs(ax * dx + ay * dy)
        # Sum of half-extent projections for both boxes
        r1 = hx * abs(ax * cos1 + ay * sin1) + hy * abs(-ax * sin1 + ay * cos1)
        r2 = hx * abs(ax * cos2 + ay * sin2) + hy * abs(-ax * sin2 + ay * cos2)
        if d > r1 + r2:
            return False  # separating axis found
    return True  # no separating axis → overlap


def _perlin_noise_2d(nrow, ncol, octaves, seed):
    """Generate 2D Perlin noise grid using the perlin-noise package."""
    from perlin_noise import PerlinNoise
    noise = PerlinNoise(octaves=octaves, seed=seed)
    return np.array([[noise([r / nrow, c / ncol]) for c in range(ncol)]
                     for r in range(nrow)])


# --- Network ---


class RailNetwork:
    """A network of railroad tracks stored as polylines with heading.

    Each road is a densely-sampled sequence of ``(x, y, heading_deg)`` tuples
    produced by :class:`RailNetworkBuilder`. Call :meth:`sample_string` to get
    poses compatible with the mesh extrusion pipeline.

    Attributes:
        spec: Rail cross-section specification.
        roads: List of roads, each a list of ``(x, y, heading_deg)`` tuples.
        sleeper_spacing: Distance between sleepers along the track (m).
        sleeper_size: Sleeper dimensions as ``(length, width, height)`` in meters.
    """

    def __init__(
        self,
        spec: RailSpec = JIS_60KG,
        roads: list | None = None,
        sleeper_spacing: float = 0.6,
        sleeper_size: tuple[float, float, float] = (0.2, 1.4, 0.1),
    ):
        self.spec = spec
        self.roads: list[list[tuple[float, float, float]]] = roads or []
        self.sleeper_spacing = sleeper_spacing
        self.sleeper_size = sleeper_size

    def sample_string(self, si: int, offset: float = 0.0, resolution: float = 0.1):
        """Subsample road *si* at *resolution* spacing with lateral *offset*.

        Returns:
            List of ``(glm.vec3, glm.quat)`` poses along the road.
        """
        road = self.roads[si]
        if len(road) < 2:
            return []

        pts = np.array([(x, y) for x, y, _ in road])
        headings = np.array([h for _, _, h in road])
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cumlen = np.concatenate(([0.0], np.cumsum(seg_lens)))
        total = cumlen[-1]
        if total < 1e-9:
            return []

        results = []
        for d in np.linspace(0, total, max(2, int(math.ceil(total / resolution)) + 1)):
            idx = int(
                np.clip(
                    np.searchsorted(cumlen, d, side="right") - 1, 0, len(seg_lens) - 1
                )
            )
            frac = (d - cumlen[idx]) / max(seg_lens[idx], 1e-12)

            x = pts[idx, 0] + frac * diffs[idx, 0]
            y = pts[idx, 1] + frac * diffs[idx, 1]
            j = min(idx + 1, len(headings) - 1)
            h = headings[idx] + frac * (headings[j] - headings[idx])

            h_rad = math.radians(h)
            x -= offset * math.sin(h_rad)
            y += offset * math.cos(h_rad)

            results.append(
                (
                    glm.vec3(x, y, 0.0),
                    glm.angleAxis(h_rad, glm.vec3(0, 0, 1)),
                )
            )
        return results

    def sample_sleepers(self, rng: np.random.Generator | None = None) -> list[tuple[glm.vec3, glm.quat]]:
        """Sample sleeper poses across all roads, eliminating overlaps.

        Sleepers from all roads are pooled. Pairs whose oriented bounding
        boxes overlap are found and one is randomly discarded, repeated
        until no overlaps remain.
        """
        sl, sw, _ = self.sleeper_size
        hx, hy = sl / 2, sw / 2  # half-extents along local axes

        poses = []
        for si in range(len(self.roads)):
            poses.extend(self.sample_string(si, resolution=self.sleeper_spacing))
        if len(poses) < 2:
            return poses

        rng = rng or np.random.default_rng()

        # Extract center (x, y) and heading for OBB tests
        cx = np.array([p.x for p, _ in poses])
        cy = np.array([p.y for p, _ in poses])
        angles = np.array([math.atan2(2 * (q.w * q.z + q.x * q.y),
                                       1 - 2 * (q.y * q.y + q.z * q.z))
                           for _, q in poses])

        def _find_overlap(cx, cy, angles):
            """Return index pair (i, j) of first OBB overlap, or None."""
            n = len(cx)
            # Quick squared-distance pre-filter (diagonal of box as radius)
            r = math.hypot(hx, hy)
            dx = cx[:, None] - cx[None, :]
            dy = cy[:, None] - cy[None, :]
            dist2 = dx * dx + dy * dy
            np.fill_diagonal(dist2, np.inf)
            candidates = np.argwhere(dist2 < (2 * r) ** 2)

            for idx in range(len(candidates)):
                i, j = int(candidates[idx, 0]), int(candidates[idx, 1])
                if i >= j:
                    continue
                # 2D SAT with 4 axes (2 edge normals per box)
                if _obb_overlap_2d(cx[i], cy[i], angles[i],
                                   cx[j], cy[j], angles[j], hx, hy):
                    return i, j
            return None

        while len(poses) >= 2:
            pair = _find_overlap(cx, cy, angles)
            if pair is None:
                break
            drop = rng.choice(pair)
            poses.pop(drop)
            cx = np.delete(cx, drop)
            cy = np.delete(cy, drop)
            angles = np.delete(angles, drop)

        return poses

    def generate_terrain(self, spec: TerrainSpec, rng: np.random.Generator | None = None):
        """Generate terrain elevation grid: flat under rails, sloping to base depth."""
        all_pts = np.vstack([np.array([(x, y) for x, y, _ in r]) for r in self.roads])
        xmin, ymin = all_pts.min(axis=0) - spec.margin
        xmax, ymax = all_pts.max(axis=0) + spec.margin
        nrow = int(math.ceil((ymax - ymin) / spec.resolution)) + 1
        ncol = int(math.ceil((xmax - xmin) / spec.resolution)) + 1
        xs = np.linspace(xmin, xmax, ncol)
        ys = np.linspace(ymin, ymax, nrow)
        gx, gy = np.meshgrid(xs, ys)

        # Min distance to any road centerline (per row for memory efficiency)
        dist = np.empty((nrow, ncol))
        for r in range(nrow):
            row_xy = np.column_stack([gx[r], gy[r]])
            d2 = ((row_xy[:, None, :] - all_pts[None, :, :]) ** 2).sum(axis=2)
            dist[r] = np.sqrt(d2.min(axis=1))

        # Smoothstep profile: flat near rails → slope → base
        t = np.clip((dist - spec.flat_radius) / spec.slope_width, 0, 1)
        t = t * t * (3 - 2 * t)
        elevation = -spec.base_depth * t

        # Add noise (masked by t so it's zero under the rails)
        rng = rng or np.random.default_rng()
        seed = int(rng.integers(0, 2**31))
        elevation += _perlin_noise_2d(nrow, ncol, spec.noise_octaves, seed) * spec.noise_amplitude * t

        return elevation, (xmin, xmax, ymin, ymax)


@dataclass
class RailNetworkBuilder:
    """Builds a :class:`RailNetwork` by growing roads step-by-step.

    Each road is a polyline whose heading evolves via a smoothly-varying
    curvature that randomly changes target. New roads branch off random points
    on existing roads and are discarded if they collide after a diverge zone.

    Attributes:
        step_size: Distance between consecutive road points (m).
        max_turn: Maximum curvature magnitude (deg/step).
        heading_change_speed: Rate at which curvature approaches its target (deg/m).
        change_turn_prob: Probability of picking a new target curvature each step.
        clearance: Minimum distance between non-adjacent roads (m).
        diverge_steps: Steps at a branch start exempt from collision checks.
        branch_margin: Minimum distance from the end of a road for branch points (m).
        min_road_length: Minimum road length (m).
        max_road_length: Maximum road length (m).
        rail_spec: Rail specification for the output network.
        sleeper_spacing: Distance between sleepers along the track (m).
        sleeper_size: Sleeper dimensions as ``(length, width, height)`` in meters.
    """

    step_size: float = 0.1
    max_turn: float = 2.0
    heading_change_speed: float = 3.0
    change_turn_prob: float = 0.01
    clearance: float = 2.0
    diverge_steps: int = 50
    branch_margin: float = 3.0
    min_road_length: float = 10.0
    max_road_length: float = 30.0
    rail_spec: RailSpec = field(default_factory=lambda: JIS_60KG)
    sleeper_spacing: float = 0.6
    sleeper_size: tuple[float, float, float] = (0.2, 1.4, 0.07)

    def _grow_road(self, rng, start, n_steps):
        """Grow a single road from *start* ``(x, y, heading_deg)`` for *n_steps*."""
        x, y, h = start
        curvature = 0.0
        target = 0.0
        road = [(x, y, h)]
        max_change = self.heading_change_speed * self.step_size

        for _ in range(n_steps):
            if rng.random() < self.change_turn_prob:
                target = float(rng.uniform(-self.max_turn, self.max_turn))
            curvature += max(-max_change, min(max_change, target - curvature))
            h += curvature
            x += self.step_size * math.cos(math.radians(h))
            y += self.step_size * math.sin(math.radians(h))
            road.append((x, y, h))
        return road

    def _intersects(self, new_road, roads, skip_start=0):
        """Check whether *new_road* collides with *roads* after *skip_start* points."""
        if not roads:
            return False
        all_pts = np.vstack([np.array([(x, y) for x, y, _ in r]) for r in roads])
        cl2 = self.clearance**2
        for x, y, _ in new_road[skip_start:]:
            if np.min(np.sum((all_pts - [x, y]) ** 2, axis=1)) < cl2:
                return True
        return False

    def build(
        self, rng: np.random.Generator, n_roads: int = 5, max_trials: int = 1000
    ) -> RailNetwork:
        """Build a network of *n_roads* roads.

        The first road starts at the origin. Subsequent roads branch off a
        random point on an existing road. Branches that collide with other
        roads (outside the diverge zone) are discarded and retried.
        """
        roads: list[list[tuple[float, float, float]]] = []

        for ri in range(n_roads):
            for _ in range(max_trials):
                length = float(rng.uniform(self.min_road_length, self.max_road_length))
                n_steps = int(length / self.step_size)

                if not roads:
                    start = (0.0, 0.0, float(rng.uniform(0, 360)))
                    skip = 0
                else:
                    parent = roads[int(rng.integers(len(roads)))]
                    margin_steps = int(self.branch_margin / self.step_size)
                    max_idx = max(0, len(parent) - 1 - margin_steps)
                    start = parent[int(rng.integers(max_idx + 1))]
                    skip = self.diverge_steps

                road = self._grow_road(rng, start, n_steps)
                if not self._intersects(road, roads, skip_start=skip):
                    roads.append(road)
                    break
            else:
                print(f"Warning: could not place road {ri} after {max_trials} trials")

        # Center the network at the origin
        if roads:
            all_pts = np.vstack([np.array([(x, y) for x, y, _ in r]) for r in roads])
            center = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2
            roads = [[(x - center[0], y - center[1], h) for x, y, h in r] for r in roads]

        return RailNetwork(
            spec=self.rail_spec,
            roads=roads,
            sleeper_spacing=self.sleeper_spacing,
            sleeper_size=self.sleeper_size,
        )


# --- Mesh ---


@dataclass
class MeshData:
    """Triangle mesh with per-vertex normals.

    Attributes:
        vertices: Vertex positions, shape ``(V, 3)``.
        faces: Triangle indices, shape ``(F, 3)``.
        normals: Per-vertex normals, shape ``(V, 3)``.
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray


def _compute_vertex_normals(verts, faces):
    normals = np.zeros_like(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(lens < 1e-12, 1.0, lens)


def _extrude_profile(profile_mm, samples, z_offset: float = 0.0):
    """Extrude a 2D cross-section along sampled curve frames."""
    n_prof = len(profile_mm)
    n_samp = len(samples)
    profile = np.array(profile_mm, dtype=float) * 0.001

    verts = np.zeros((n_samp * n_prof, 3))
    for si, (pos, quat) in enumerate(samples):
        left = quat * glm.vec3(0, 1, 0)
        for pi, (px, py) in enumerate(profile):
            v = pos + float(px) * left + float(py + z_offset) * glm.vec3(0, 0, 1)
            verts[si * n_prof + pi] = [v.x, v.y, v.z]

    faces = []
    for si in range(n_samp - 1):
        for pi in range(n_prof):
            pn = (pi + 1) % n_prof
            a, b = si * n_prof + pi, si * n_prof + pn
            c, d = (si + 1) * n_prof + pn, (si + 1) * n_prof + pi
            faces.extend([[a, d, c], [a, c, b]])

    faces = np.array(faces, dtype=int)
    return MeshData(verts, faces, _compute_vertex_normals(verts, faces))


# --- Output ---


def generate_mujoco_xml(net: RailNetwork, resolution: float = 0.2, terrain: TerrainSpec | None = None) -> str:
    """Generate MuJoCo MJCF XML with inline rail meshes and optional terrain hfield."""
    spec = net.spec
    root = Element("mujoco", model="rail_network")
    asset = SubElement(root, "asset")
    SubElement(
        asset,
        "material",
        name="mat_rail",
        rgba="0.55 0.55 0.6 1",
        specular="0.8",
        shininess="0.9",
    )
    SubElement(
        asset,
        "material",
        name="mat_sleeper",
        rgba="0.4 0.28 0.18 1",
        specular="0.2",
        shininess="0.1",
    )
    worldbody = SubElement(root, "worldbody")
    half_g = spec.gauge / 2.0
    _sl, _sw, sh = net.sleeper_size
    rail_z = sh / 2  # rails start at top of (half-buried) sleepers

    # Rail cross-section half-extents for box colliders (mm → m)
    rail_hw = spec.profile[0][0] * 0.001  # half base width (negative, take abs)
    rail_hw = abs(rail_hw)
    rail_hh = max(py for _, py in spec.profile) * 0.001 / 2  # half height

    for si in range(len(net.roads)):
        for tag, off in [("L", half_g), ("R", -half_g)]:
            samples = net.sample_string(si, offset=off, resolution=resolution)
            if len(samples) < 2:
                continue
            mesh = _extrude_profile(spec.profile, samples, z_offset=rail_z)
            name = f"rail_s{si}_{tag}"
            SubElement(
                asset,
                "mesh",
                name=name,
                vertex=" ".join(f"{v:.6f}" for v in mesh.vertices.ravel()),
                face=" ".join(str(i) for i in mesh.faces.ravel()),
            )
            # Visual only — convex-hull collision won't work for long rails
            SubElement(
                worldbody,
                "geom",
                name=name,
                type="mesh",
                mesh=name,
                material="mat_rail",
                contype="0",
                conaffinity="0",
            )
            # Box colliders along the rail for accurate collision
            for ji in range(len(samples) - 1):
                p0, _ = samples[ji]
                p1, _ = samples[ji + 1]
                mx = (p0.x + p1.x) / 2
                my = (p0.y + p1.y) / 2
                mz = rail_z + rail_hh
                dx, dy = p1.x - p0.x, p1.y - p0.y
                half_len = math.sqrt(dx * dx + dy * dy) / 2
                yaw = math.atan2(dy, dx)
                SubElement(
                    worldbody,
                    "geom",
                    name=f"{name}_col{ji}",
                    type="box",
                    size=f"{half_len:.4f} {rail_hw:.4f} {rail_hh:.4f}",
                    pos=f"{mx:.4f} {my:.4f} {mz:.4f}",
                    euler=f"0 0 {yaw:.6f}",
                    contype="1",
                    conaffinity="1",
                    group="3",
                )

    # Sleepers (deduplicated across all roads) — sunk halfway into ground
    sl, sw, _sh = net.sleeper_size
    for ti, (pos, quat) in enumerate(net.sample_sleepers()):
        fwd = quat * glm.vec3(1, 0, 0)
        yaw = math.atan2(float(fwd.y), float(fwd.x))
        SubElement(
            worldbody,
            "geom",
            name=f"sleeper_{ti}",
            type="box",
            size=f"{sl/2:.4f} {sw/2:.4f} {sh/2:.4f}",
            pos=f"{pos.x:.4f} {pos.y:.4f} {0:.4f}",
            euler=f"0 0 {yaw:.6f}",
            material="mat_sleeper",
            contype="1",
            conaffinity="1",
        )

    # Terrain heightfield
    if terrain is not None:
        elevation, (xmin, xmax, ymin, ymax) = net.generate_terrain(terrain)
        nrow, ncol = elevation.shape
        e_min, e_max = float(elevation.min()), float(elevation.max())
        e_range = max(e_max - e_min, 1e-6)
        rx, ry = (xmax - xmin) / 2, (ymax - ymin) / 2
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        SubElement(
            asset, "material", name="mat_terrain",
            rgba="0.45 0.38 0.28 1", specular="0.1", shininess="0.1",
        )
        SubElement(
            asset, "hfield", name="terrain",
            nrow=str(nrow), ncol=str(ncol),
            size=f"{rx:.4f} {ry:.4f} {e_range:.4f} 0.5",
            elevation=" ".join(f"{v:.4f}" for v in elevation[::-1].ravel()),
        )
        SubElement(
            worldbody, "geom", name="terrain",
            type="hfield", hfield="terrain",
            pos=f"{cx:.4f} {cy:.4f} {e_min:.4f}",
            material="mat_terrain", contype="1", conaffinity="1",
        )

    indent(root, space="  ")
    return tostring(root, encoding="unicode")


def log_network(net: RailNetwork, terrain: TerrainSpec | None = None):
    """Log the rail network to Rerun: centerlines, meshes, and optional terrain."""
    import rerun as rr

    spec = net.spec
    half_g = spec.gauge / 2.0
    sl, sw, sh = net.sleeper_size

    for si in range(len(net.roads)):
        samples = net.sample_string(si, resolution=0.2)
        if samples:
            pts = np.array([[p.x, p.y, p.z] for p, _ in samples])
            rr.log(
                f"network/{si}/center",
                rr.LineStrips3D([pts], colors=[80, 80, 80], radii=0.01),
            )

        for tag, off in [("L", half_g), ("R", -half_g)]:
            samples = net.sample_string(si, offset=off, resolution=0.2)
            if len(samples) < 2:
                continue
            mesh = _extrude_profile(spec.profile, samples, z_offset=sh / 2)
            rr.log(
                f"network/{si}/mesh_{tag}",
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    triangle_indices=mesh.faces,
                    vertex_normals=mesh.normals,
                    vertex_colors=[140, 140, 155],
                ),
            )

    # Sleepers (deduplicated across all roads) — sunk halfway into ground
    sleepers = net.sample_sleepers()
    if sleepers:
        centers = []
        half_sizes = []
        rotations = []
        for pos, quat in sleepers:
            fwd = quat * glm.vec3(1, 0, 0)
            yaw = math.atan2(float(fwd.y), float(fwd.x))
            centers.append([pos.x, pos.y, 0.0])
            half_sizes.append([sl / 2, sw / 2, sh / 2])
            q = glm.angleAxis(yaw, glm.vec3(0, 0, 1))
            rotations.append(rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w]))
        rr.log(
            "network/sleepers",
            rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                colors=[[100, 70, 45]],
                quaternions=rotations,
                fill_mode="solid",
            ),
        )

    # Terrain mesh
    if terrain is not None:
        elevation, (xmin, xmax, ymin, ymax) = net.generate_terrain(terrain)
        nrow, ncol = elevation.shape
        xs = np.linspace(xmin, xmax, ncol)
        ys = np.linspace(ymin, ymax, nrow)
        gx, gy = np.meshgrid(xs, ys)
        verts = np.column_stack([gx.ravel(), gy.ravel(), elevation.ravel()])
        i = (np.arange(nrow - 1)[:, None] * ncol + np.arange(ncol - 1)[None, :]).ravel()
        faces = np.vstack([np.column_stack([i, i + ncol, i + 1]),
                           np.column_stack([i + 1, i + ncol, i + ncol + 1])])
        rr.log(
            "network/terrain",
            rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=faces,
                vertex_normals=_compute_vertex_normals(verts, faces),
                vertex_colors=[120, 100, 80],
            ),
        )


class RailwayScene:
    """A complete rail scene: network + terrain, with MuJoCo and Rerun output."""

    def __init__(self, net: RailNetwork, terrain: TerrainSpec | None = None):
        self.net = net
        self.terrain = terrain

    _TERRAIN_DEFAULT = object()  # sentinel: distinguish 'not passed' from None

    @classmethod
    def build(cls, rng: np.random.Generator, n_roads: int = 5,
              terrain: TerrainSpec | None | object = _TERRAIN_DEFAULT,
              **builder_kwargs) -> "RailwayScene":
        if terrain is cls._TERRAIN_DEFAULT:
            terrain = TerrainSpec()
        net = RailNetworkBuilder(**builder_kwargs).build(rng, n_roads=n_roads)
        return cls(net, terrain)

    def log_rerun(self):
        log_network(self.net, terrain=self.terrain)

    def save_mujoco_scene(self, project_root: str,
                          start_pos: tuple[float, float, float] | None = None) -> str:
        """Write a complete MuJoCo scene XML to a temp file and return its path.

        The file includes the Go2 robot, lighting, and all rail/terrain
        geometry.  The caller is responsible for deleting the file when done.

        Args:
            project_root: Absolute path to the project root (for go2.xml include).
            start_pos: Optional (x, y, heading_rad) to place the robot.
        """
        import tempfile
        from xml.etree.ElementTree import parse as ET_parse
        from io import StringIO

        go2_xml = os.path.join(
            project_root, "resources", "go2.xml"
        )

        # Build the base scene (mirroring resources/scene_flat.xml)
        root = Element("mujoco", model="go2 rail scene")
        SubElement(root, "include", file=go2_xml)
        SubElement(root, "statistic", center="0 0 0.1", extent="0.8")

        vis = SubElement(root, "visual")
        SubElement(vis, "headlight", diffuse="0.6 0.6 0.6",
                   ambient="0.3 0.3 0.3", specular="0 0 0")
        SubElement(vis, "rgba", haze="0.15 0.25 0.35 1")
        SubElement(vis, "global", azimuth="-130", elevation="-20",
                   offwidth="1280", offheight="720")
        SubElement(vis, "map", zfar="200")

        asset = SubElement(root, "asset")
        SubElement(asset, "texture", type="skybox", builtin="gradient",
                   rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
        SubElement(asset, "texture", type="2d", name="groundplane",
                   builtin="checker", mark="edge",
                   rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3",
                   markrgb="0.8 0.8 0.8", width="300", height="300")
        SubElement(asset, "material", name="groundplane", texture="groundplane",
                   texuniform="true", texrepeat="5 5", reflectance="0.2")

        worldbody = SubElement(root, "worldbody")
        SubElement(worldbody, "light", pos="0 0 1.5", dir="0 0 -1",
                   directional="true")
        SubElement(worldbody, "camera", name="spectator",
                   pos="0 -3 1.5", xyaxes="1 0 0 0 0.447 0.894")

        # Override the robot start position if requested
        if start_pos is not None:
            sx, sy, syaw = start_pos
            cw, sw_ = math.cos(syaw / 2), math.sin(syaw / 2)
            qpos = (f"{sx} {sy} 0.27 {cw} 0 0 {sw_} "
                    "0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8")
            ctrl = "0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8"
            kf = SubElement(root, "keyframe")
            SubElement(kf, "key", name="rail_start", qpos=qpos, ctrl=ctrl)

        # Human-marker mocap body (visible + collidable for lidar)
        marker = SubElement(worldbody, "body", name="uwb_tag", mocap="true")
        SubElement(marker, "geom", type="cylinder", size="0.2 0.9",
                   rgba="1.0 0.5 0.0 0.5", contype="0", conaffinity="0")

        # Add a flat floor if no terrain heightfield
        if self.terrain is None:
            SubElement(worldbody, "geom", name="floor", type="plane",
                       size="0 0 0.05", material="groundplane")

        # Merge rail/sleeper/terrain from generate_mujoco_xml()
        rail_xml = generate_mujoco_xml(self.net, terrain=self.terrain)
        rail_root = ET_parse(StringIO(rail_xml)).getroot()
        rail_asset = rail_root.find("asset")
        rail_wb = rail_root.find("worldbody")
        if rail_asset is not None:
            for child in rail_asset:
                asset.append(child)
        if rail_wb is not None:
            for child in rail_wb:
                worldbody.append(child)

        indent(root, space="  ")
        xml_str = tostring(root, encoding="unicode")

        fd, path = tempfile.mkstemp(suffix=".xml", prefix="rail_scene_")
        os.write(fd, xml_str.encode())
        os.close(fd)
        return path


if __name__ == "__main__":
    import rerun as rr

    rng = np.random.default_rng()
    rr.init("rail_network", spawn=True)
    builder = RailNetworkBuilder()
    net = builder.build(rng, n_roads=5)
    terrain = TerrainSpec()
    print(f"Roads: {len(net.roads)}, points: {sum(len(r) for r in net.roads)}")
    log_network(net, terrain=terrain)
