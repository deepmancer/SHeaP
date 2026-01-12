from typing import Tuple, Union

import numpy as np
import pyrender
import torch
import trimesh


def render_mesh(
    verts: Union[np.ndarray, torch.Tensor],
    faces: Union[np.ndarray, torch.Tensor],
    c2w: Union[np.ndarray, torch.Tensor],
    img_width: int = 512,
    img_height: int = 512,
    fov_degrees: Union[float, int] = 14.2539,
    render_normals: bool = True,
    render_segmentation: bool = False,
    vertex_colors: Union[np.ndarray, None] = None,
    black_background: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render mesh with pyrender. Returns (color, depth) as (H,W,3) uint8 and (H,W) float32."""
    if isinstance(c2w, torch.Tensor):
        c2w = c2w.detach().cpu().numpy()
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    if not isinstance(fov_degrees, (float, int)):
        fov_degrees = float(fov_degrees)

    # Convert degrees to radians
    yfov = np.deg2rad(fov_degrees)

    # Create trimesh mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    if render_segmentation:
        # Use provided vertex colors for segmentation
        if vertex_colors is not None:
            mesh.visual.vertex_colors = vertex_colors
        else:
            # Fallback: create default segmentation colors
            from .flame_segmentation import create_segmentation_texture
            seg_colors = create_segmentation_texture(verts, faces)
            mesh.visual.vertex_colors = seg_colors
    elif render_normals:
        # Get vertex normals and map to RGB colors
        # Trimesh automatically computes normals when accessed
        normals = mesh.vertex_normals
        # Transform normals to camera space
        w2c = np.linalg.inv(c2w)
        normals_camera = normals @ w2c[:3, :3].T
        # Map from [-1, 1] to [0, 255] for RGB
        vertex_colors_normals = ((normals_camera + 1.0) * 0.5 * 255).astype(np.uint8)
        mesh.visual.vertex_colors = vertex_colors_normals

    # Convert to pyrender mesh
    render_mesh = pyrender.Mesh.from_trimesh(mesh)

    # Create scene with black or white background
    bg_color = [0.0, 0.0, 0.0, 1.0] if black_background else [1.0, 1.0, 1.0, 1.0]
    if render_normals or render_segmentation:
        # For normals and segmentation, use full ambient light (no shading)
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=bg_color)
    else:
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=bg_color)
        # Add directional light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=c2w)
    scene.add(render_mesh)

    # Perspective camera
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=img_width / img_height)

    # pyrender expects camera-to-world
    scene.add(camera, pose=c2w)

    # Offscreen render
    renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height)
    color, depth = renderer.render(scene)

    return color, depth
