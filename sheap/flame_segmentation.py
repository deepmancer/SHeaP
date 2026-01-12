"""FLAME face segmentation utilities."""

import pickle
from pathlib import Path
import numpy as np
import torch
from typing import Tuple, Union
import trimesh


def load_flame_masks(flame_masks_path: Union[str, Path] = None) -> dict:
    """Load FLAME vertex masks from pickle file."""
    if flame_masks_path is None:
        flame_masks_path = Path(__file__).parent.parent / "FLAME2020" / "FLAME_masks.pkl"
    
    with open(flame_masks_path, 'rb') as f:
        masks = pickle.load(f, encoding='latin1')
    return masks


def get_face_vertices_excluding_neck_boundary(flame_masks_path: Union[str, Path] = None) -> np.ndarray:
    """Get vertex indices for face excluding neck and boundary regions."""
    masks = load_flame_masks(flame_masks_path)
    
    # Get all vertex indices (assuming 5023 vertices)
    all_vertices = np.arange(5023)
    
    # Get vertices to exclude
    neck_vertices = masks['neck']
    boundary_vertices = masks['boundary']
    exclude_vertices = np.concatenate([neck_vertices, boundary_vertices])
    
    # Get face vertices (all vertices except excluded ones)
    face_vertices = np.setdiff1d(all_vertices, exclude_vertices)
    
    return face_vertices


def get_largest_connected_component(
    verts: np.ndarray,
    faces: np.ndarray,
    keep_vertex_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract largest connected component from mesh after keeping only specified vertices.
    
    Returns:
        new_verts: Vertices of largest component
        new_faces: Faces of largest component  
        vertex_mask: Boolean mask indicating which original vertices are kept
    """
    # Create a submesh with only the kept vertices
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    
    # Create vertex mask
    vertex_mask = np.zeros(len(verts), dtype=bool)
    vertex_mask[keep_vertex_indices] = True
    
    # Keep only specified vertices
    mesh.update_vertices(vertex_mask)
    
    # Split into connected components
    components = mesh.split(only_watertight=False)
    
    if len(components) == 0:
        # Return empty mesh
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), vertex_mask
    
    # Find largest component by number of vertices
    largest = max(components, key=lambda m: len(m.vertices))
    
    # Create a new vertex mask for the largest component
    # Map back to original vertex indices
    final_vertex_mask = np.zeros(len(verts), dtype=bool)
    
    # The vertices in largest component maintain their original positions
    # We need to identify which original vertices they correspond to
    kept_indices = np.where(vertex_mask)[0]
    
    # trimesh.split returns meshes with new vertex indices
    # We need to map back: the kept_indices contains the original indices
    # The largest component has len(largest.vertices) vertices which are a subset
    
    # Simpler approach: iterate through original vertices and check if in largest component
    for orig_idx in kept_indices:
        orig_pos = verts[orig_idx]
        # Check if this vertex is in the largest component
        if np.any(np.all(np.isclose(largest.vertices, orig_pos, atol=1e-6), axis=1)):
            final_vertex_mask[orig_idx] = True
    
    return largest.vertices, largest.faces, final_vertex_mask


def create_binary_mask_texture(
    verts: Union[np.ndarray, torch.Tensor],
    faces: Union[np.ndarray, torch.Tensor],
    flame_masks_path: Union[str, Path] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create binary mask texture for FLAME mesh excluding neck and boundary.
    
    Returns:
        new_verts: Vertices of masked mesh
        new_faces: Faces of masked mesh
        vertex_colors: White (255,255,255) for all kept vertices
    """
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    
    # Get vertices to keep (exclude neck and boundary)
    keep_indices = get_face_vertices_excluding_neck_boundary(flame_masks_path)
    
    # Get largest connected component
    new_verts, new_faces, _ = get_largest_connected_component(verts, faces, keep_indices)
    
    # Create white color for all vertices in the mask
    vertex_colors = np.full((len(new_verts), 3), 255, dtype=np.uint8)
    
    return new_verts, new_faces, vertex_colors
