import numpy as np
import pyrender
import torch
import trimesh


def render_mesh(verts, faces, c2w, img_width=512, img_height=512, fov_degrees=14.2539):
    """
    Render a mesh using pyrender with a perspective camera defined by FOV.

    Parameters
    ----------
    verts : ndarray (N, 3)
        Mesh vertex positions.
    faces : ndarray (F, 3)
        Triangle vertex indices.
    c2w : ndarray (4, 4)
        World-to-camera transform matrix (extrinsics).
    fov_degrees : float
        Vertical field of view in degrees.
    img_width : int
        Rendered image width in pixels.
    img_height : int
        Rendered image height in pixels.

    Returns
    -------
    color : ndarray (H, W, 3) uint8
        RGB image from the render.
    depth : ndarray (H, W) float32
        Depth map from the render.
    """
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

    # Convert to pyrender mesh
    render_mesh = pyrender.Mesh.from_trimesh(mesh)

    # Create scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(render_mesh)

    # Add directional light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=c2w)

    # Perspective camera
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=img_width / img_height)

    # pyrender expects camera-to-world
    scene.add(camera, pose=c2w)

    # Offscreen render
    renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height)
    color, depth = renderer.render(scene)

    return color, depth
