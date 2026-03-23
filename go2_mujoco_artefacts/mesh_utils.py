from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import rerun as rr


def compute_vertex_normals(
    vertex_positions: np.ndarray,
    triangle_indices: np.ndarray,
) -> np.ndarray:
    """Smooth per-vertex normals from indexed triangles."""
    normals = np.zeros_like(vertex_positions, dtype=np.float32)

    tris = vertex_positions[triangle_indices]  # (F, 3, 3)
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])

    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 0
    face_normals[valid] /= lengths[valid]

    for i in range(3):
        np.add.at(normals, triangle_indices[:, i], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 0
    normals[valid] /= lengths[valid]
    return normals


def _rgba_float_to_u8(rgba: np.ndarray) -> np.ndarray:
    rgba = np.asarray(rgba, dtype=np.float32)
    rgba = np.clip(rgba, 0.0, 1.0)
    return np.round(rgba * 255.0).astype(np.uint8)


def _resolve_geom_id_for_mesh(
    model: mujoco.MjModel,
    mesh_id: int,
    geom_id: int | None,
) -> int | None:
    if geom_id is not None:
        return int(geom_id)

    geom_ids = np.flatnonzero(
        (np.asarray(model.geom_type) == int(mujoco.mjtGeom.mjGEOM_MESH))
        & (np.asarray(model.geom_dataid) == mesh_id)
    )

    if len(geom_ids) >= 1: # defaults to first one
        return int(geom_ids[0])

    # Ambiguous or unused mesh asset.
    return None


def _mat_texid(
    model: mujoco.MjModel,
    mat_id: int,
    role: int,
) -> int:
    mat_texid = np.asarray(model.mat_texid)
    if mat_texid.ndim == 2:
        return int(mat_texid[mat_id, role])
    # Fallback for flattened layouts.
    nroles = int(mujoco.mjtTextureRole.mjNTEXROLE)
    return int(mat_texid[mat_id * nroles + role])


def _extract_texture_image(
    model: mujoco.MjModel,
    tex_id: int,
) -> np.ndarray | None:
    if tex_id < 0:
        return None

    width = int(np.asarray(model.tex_width)[tex_id])
    height = int(np.asarray(model.tex_height)[tex_id])
    nchannel = int(np.asarray(model.tex_nchannel)[tex_id])
    adr = int(np.asarray(model.tex_adr)[tex_id])

    if width <= 0 or height <= 0 or nchannel <= 0:
        return None

    size = width * height * nchannel
    flat = np.asarray(model.tex_data)[adr : adr + size]
    image = np.asarray(flat, dtype=np.uint8).reshape(height, width, nchannel)

    # Rerun's albedo texture path is most useful with 3 or 4 channels.
    if nchannel == 3 or nchannel == 4:
        return image
    if nchannel == 1:
        return np.repeat(image, 3, axis=2)
    if nchannel == 2:
        rgb = np.repeat(image[:, :, :1], 3, axis=2)
        return rgb

    return None


def mujoco_mesh_view_to_rerun(
    model: mujoco.MjModel,
    mesh: Any,  # e.g. model.mesh("foot")
    *,
    geom_id: int | None = None,
    with_normals: bool = True,
) -> rr.Mesh3D:
    """Convert a MuJoCo mesh view (e.g. model.mesh("foot")) into a rerun.Mesh3D.

    Notes:
    - Colors/textures are geom/material-dependent, not mesh-asset-dependent.
    - If `geom_id` is omitted, color/texture is only applied when exactly one mesh geom uses this mesh.
    - Uses triangle soup so separate MuJoCo face indices for positions/normals/UVs are handled robustly.
    """

    mesh_id = int(mesh.id)

    vert_start = int(mesh.vertadr[0])
    vert_count = int(mesh.vertnum[0])
    face_start = int(mesh.faceadr[0])
    face_count = int(mesh.facenum[0])

    vertices = np.asarray(
        model.mesh_vert[vert_start : vert_start + vert_count],
        dtype=np.float32,
    ).reshape(-1, 3)

    faces = np.asarray(
        model.mesh_face[face_start : face_start + face_count],
        dtype=np.int32,
    ).reshape(-1, 3)

    # Expand to triangle soup so we can attach per-corner normals/UVs if needed.
    vertex_positions = vertices[faces.reshape(-1)]
    triangle_indices = np.arange(len(vertex_positions), dtype=np.uint32).reshape(-1, 3)

    kwargs: dict[str, object] = {
        "vertex_positions": vertex_positions,
        "triangle_indices": triangle_indices,
    }

    # --- normals ---
    tri_normals: np.ndarray | None = None
    if with_normals:
        normal_start = int(np.asarray(model.mesh_normaladr)[mesh_id])
        normal_count = int(np.asarray(model.mesh_normalnum)[mesh_id])

        if normal_start >= 0 and normal_count > 0:
            normals = np.asarray(
                model.mesh_normal[normal_start : normal_start + normal_count],
                dtype=np.float32,
            ).reshape(-1, 3)

            face_normal_idx = np.asarray(
                model.mesh_facenormal[face_start : face_start + face_count]
            ).reshape(-1, 3)

            if face_normal_idx.shape == faces.shape and len(normals) > 0:
                # Separate normal index stream.
                tri_normals = normals[face_normal_idx.reshape(-1)]
            elif len(normals) == len(vertices):
                # Per-vertex normals aligned with vertex indices.
                tri_normals = normals[faces.reshape(-1)]

        if tri_normals is None:
            smooth_normals = compute_vertex_normals(vertices, faces)
            tri_normals = smooth_normals[faces.reshape(-1)]

        kwargs["vertex_normals"] = tri_normals.astype(np.float32, copy=False)

    # --- texcoords ---
    tri_uvs: np.ndarray | None = None
    texcoord_start = int(np.asarray(model.mesh_texcoordadr)[mesh_id])
    texcoord_count = int(np.asarray(model.mesh_texcoordnum)[mesh_id])

    if texcoord_start >= 0 and texcoord_count > 0:
        texcoords = np.asarray(
            model.mesh_texcoord[texcoord_start : texcoord_start + texcoord_count],
            dtype=np.float32,
        ).reshape(-1, 2)

        face_texcoord_idx = np.asarray(
            model.mesh_facetexcoord[face_start : face_start + face_count]
        ).reshape(-1, 3)

        if face_texcoord_idx.shape == faces.shape and len(texcoords) > 0:
            # Separate texcoord index stream.
            tri_uvs = texcoords[face_texcoord_idx.reshape(-1)]
        elif len(texcoords) == len(vertices):
            # UVs aligned with vertex indices.
            tri_uvs = texcoords[faces.reshape(-1)]

    # --- color / material / texture ---
    resolved_geom_id = _resolve_geom_id_for_mesh(model, mesh_id, geom_id)
    if resolved_geom_id is not None:
        mat_id = int(np.asarray(model.geom_matid)[resolved_geom_id])

        if mat_id >= 0:
            rgba = np.asarray(model.mat_rgba[mat_id], dtype=np.float32)
            kwargs["albedo_factor"] = _rgba_float_to_u8(rgba)

            # Prefer RGBA, then RGB.
            tex_id = _mat_texid(model, mat_id, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA))
            if tex_id < 0:
                tex_id = _mat_texid(model, mat_id, int(mujoco.mjtTextureRole.mjTEXROLE_RGB))

            if tex_id >= 0 and tri_uvs is not None:
                image = _extract_texture_image(model, tex_id)
                if image is not None:
                    kwargs["vertex_texcoords"] = tri_uvs.astype(np.float32, copy=False)
                    kwargs["albedo_texture"] = image
        else:
            # No material: use geom rgba.
            rgba = np.asarray(model.geom_rgba[resolved_geom_id], dtype=np.float32)
            kwargs["albedo_factor"] = _rgba_float_to_u8(rgba)

    return rr.Mesh3D(**kwargs)
