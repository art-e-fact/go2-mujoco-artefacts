from dataclasses import dataclass, field
import math
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
            (-B/2,   0),       ( B/2,   0),        # base bottom
            ( B/2,   F*0.3),   ( G/2+8, F),        # foot right taper
            ( G/2,   F),       ( G/2,   F+E),      # web right
            ( C/2,   F+E),     ( C/2,   A),        # head right
            (-C/2,   A),       (-C/2,   F+E),      # head left
            (-G/2,   F+E),     (-G/2,   F),        # web left
            (-G/2-8, F),       (-B/2,   F*0.3),    # foot left taper
        ],
    )  # fmt: skip


JIS_60KG = _make_jis60kg()


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
    sleeper_size: tuple[float, float, float] = (0.2, 1.4, 0.1)

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


def _extrude_profile(profile_mm, samples):
    """Extrude a 2D cross-section along sampled curve frames."""
    n_prof = len(profile_mm)
    n_samp = len(samples)
    profile = np.array(profile_mm, dtype=float) * 0.001

    verts = np.zeros((n_samp * n_prof, 3))
    for si, (pos, quat) in enumerate(samples):
        left = quat * glm.vec3(0, 1, 0)
        for pi, (px, py) in enumerate(profile):
            v = pos + float(px) * left + float(py) * glm.vec3(0, 0, 1)
            verts[si * n_prof + pi] = [v.x, v.y, v.z]

    faces = []
    for si in range(n_samp - 1):
        for pi in range(n_prof):
            pn = (pi + 1) % n_prof
            a, b = si * n_prof + pi, si * n_prof + pn
            c, d = (si + 1) * n_prof + pn, (si + 1) * n_prof + pi
            faces.extend([[a, d, c], [a, c, b]])

    for ring_start, winding in [(0, -1), ((n_samp - 1) * n_prof, 1)]:
        ci = len(verts)
        verts = np.vstack(
            [
                verts,
                np.mean(verts[ring_start : ring_start + n_prof], axis=0, keepdims=True),
            ]
        )
        for pi in range(n_prof):
            pn = (pi + 1) % n_prof
            if winding < 0:
                faces.append([ci, ring_start + pn, ring_start + pi])
            else:
                faces.append([ci, ring_start + pi, ring_start + pn])

    faces = np.array(faces, dtype=int)
    return MeshData(verts, faces, _compute_vertex_normals(verts, faces))


# --- Output ---


def generate_mujoco_xml(net: RailNetwork, resolution: float = 0.2) -> str:
    """Generate MuJoCo MJCF XML with inline rail meshes."""
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

    for si in range(len(net.roads)):
        for tag, off in [("L", half_g), ("R", -half_g)]:
            samples = net.sample_string(si, offset=off, resolution=resolution)
            if len(samples) < 2:
                continue
            mesh = _extrude_profile(spec.profile, samples)
            name = f"rail_s{si}_{tag}"
            SubElement(
                asset,
                "mesh",
                name=name,
                vertex=" ".join(f"{v:.6f}" for v in mesh.vertices.ravel()),
                face=" ".join(str(i) for i in mesh.faces.ravel()),
            )
            SubElement(
                worldbody,
                "geom",
                name=name,
                type="mesh",
                mesh=name,
                material="mat_rail",
                contype="1",
                conaffinity="1",
            )

    # Sleepers (deduplicated across all roads)
    sl, sw, sh = net.sleeper_size
    for ti, (pos, quat) in enumerate(net.sample_sleepers()):
        fwd = quat * glm.vec3(1, 0, 0)
        yaw = math.atan2(float(fwd.y), float(fwd.x))
        SubElement(
            worldbody,
            "geom",
            name=f"sleeper_{ti}",
            type="box",
            size=f"{sl/2:.4f} {sw/2:.4f} {sh/2:.4f}",
            pos=f"{pos.x:.4f} {pos.y:.4f} {sh/2:.4f}",
            euler=f"0 0 {math.degrees(yaw):.2f}",
            material="mat_sleeper",
            contype="1",
            conaffinity="1",
        )

    indent(root, space="  ")
    return tostring(root, encoding="unicode")


def log_network(net: RailNetwork):
    """Log the rail network to Rerun: centerlines and extruded meshes."""
    import rerun as rr

    spec = net.spec
    half_g = spec.gauge / 2.0

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
            mesh = _extrude_profile(spec.profile, samples)
            rr.log(
                f"network/{si}/mesh_{tag}",
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    triangle_indices=mesh.faces,
                    vertex_normals=mesh.normals,
                    vertex_colors=[140, 140, 155],
                ),
            )

    # Sleepers (deduplicated across all roads)
    sl, sw, sh = net.sleeper_size
    sleepers = net.sample_sleepers()
    if sleepers:
        centers = []
        half_sizes = []
        rotations = []
        for pos, quat in sleepers:
            fwd = quat * glm.vec3(1, 0, 0)
            yaw = math.atan2(float(fwd.y), float(fwd.x))
            centers.append([pos.x, pos.y, sh / 2])
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
            ),
        )


if __name__ == "__main__":
    import rerun as rr

    rng = np.random.default_rng()
    rr.init("rail_network", spawn=True)
    builder = RailNetworkBuilder()
    net = builder.build(rng, n_roads=5)
    print(f"Roads: {len(net.roads)}, points: {sum(len(r) for r in net.roads)}")
    log_network(net)
