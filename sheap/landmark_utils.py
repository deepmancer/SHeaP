from typing import Tuple

import torch
from torch import Tensor


def vertices_to_landmarks(
    vertices: Tensor,  # shape: (*batch, num_vertices, 3)
    faces: Tensor,  # shape: (num_faces, 3), indices of vertices
    face_indices_with_landmarks: Tensor,  # shape: (num_landmarks,), indices of faces
    barys: Tensor,  # shape: (num_landmarks, 3), barycentric coordinates
) -> Tensor:
    """
    Calculate the 3D world coordinates of landmarks from mesh vertices.

    Args:
        vertices (Tensor): Mesh vertices of shape (*batch, num_vertices, 3).
        faces (Tensor): Mesh faces of shape (num_faces, 3), containing indices into `vertices`.
        face_indices_with_landmarks (Tensor): Indices of faces containing the landmarks, shape (num_landmarks,).
        barys (Tensor): Barycentric coordinates of the landmarks in their respective faces,
                        shape (num_landmarks, 3). The last dimension should sum to 1.0.

    Returns:
        Tensor: Landmark positions of shape (*batch, num_landmarks, 3).
    """
    did_unsqueeze = False
    if vertices.ndim == 2:  # Support no batch dimension case
        vertices = vertices.unsqueeze(0)
        did_unsqueeze = True

    batch_dims = vertices.shape[:-2]

    # Select the faces that contain the landmarks
    relevant_faces = faces[face_indices_with_landmarks]

    # Select vertices corresponding to relevant faces
    selected_vertices = torch.index_select(vertices, len(batch_dims), relevant_faces.view(-1)).view(
        *batch_dims, *relevant_faces.shape, 3
    )

    # Compute landmark positions using barycentric interpolation
    landmark_positions = torch.einsum("b...lvx,lv->b...lx", selected_vertices, barys)

    if did_unsqueeze:
        landmark_positions = landmark_positions[0]

    return landmark_positions


def vertices_to_7_lmks(
    vertices: Tensor,
    flame_faces: Tensor,
    face_alignment_lmk_faces_idx: Tensor,
    face_alignment_lmk_bary_coords: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Extract the 7 specific 3D landmarks (and all landmarks) from mesh vertices.

    Args:
        vertices (Tensor): Mesh vertices of shape (*batch, num_vertices, 3).
        flame_faces (Tensor): Mesh faces of shape (num_faces, 3).
        face_alignment_lmk_faces_idx (Tensor): Indices of faces that contain facial landmarks.
        face_alignment_lmk_bary_coords (Tensor): Barycentric coordinates of landmarks within faces.

    Returns:
        Tuple[Tensor, Tensor]:
            - lmks7_3d: Landmark positions for 7 specific points, shape (*batch, 7, 3).
            - lmks_3d: Landmark positions for all landmarks, shape (*batch, num_landmarks, 3).
    """
    lmks_3d = vertices_to_landmarks(
        vertices,
        flame_faces,
        face_alignment_lmk_faces_idx,
        face_alignment_lmk_bary_coords,
    )

    # Select landmark subset starting from index 17 (e.g., 51 landmarks)
    landmark_51 = lmks_3d[:, 17:]

    # Extract specific 7 landmark indices
    lmks7_3d = landmark_51[:, [19, 22, 25, 28, 16, 31, 37]]

    return lmks7_3d, lmks_3d


def landmarks_2_face_bounding_box(
    landmarks: Tensor,
    valid: Tensor,
    margin: float = 0.1,
    clamp: bool = True,
    shift_up: float = 0.0,
    too_small_threshold: float = 0.02,
    aspect_ratio: float = 1.0,
) -> Tensor:
    """
    Calculate a square bounding box around face landmarks with a specified margin for batched inputs.

    Parameters:
    - landmarks: torch.Tensor of shape [B1,...,BN,L,2], normalized face landmarks.
    - valid: torch.Tensor of shape [B1,...,BN], boolean indicating validity of each entry.
    - margin: float, margin factor to expand the bounding box around the face.
    - clamp: bool, whether to clamp the bounding box to [0, 1].
    - shift_up: float, factor to shift the bounding box up.
    - too_small_threshold: float, threshold for the bounding box size.
    - aspect_ratio: float, aspect ratio of the image that the landmarks live on (width / height).
        The box size will be divided by this value, under the assumption that you are going to
        multiply these normalised coordinates by the image width later.

    Returns:
    - bbox: torch.Tensor of shape [B1,...,BN,4] representing the square bounding box.
    """
    # Calculate min and max coordinates along the last dimension for x and y
    min_coords, _ = landmarks.min(dim=-2)
    max_coords, _ = landmarks.max(dim=-2)

    # Calculate the center and size of the bounding box
    center_coords = (min_coords + max_coords) / 2
    half_size = ((max_coords - min_coords).max(dim=-1).values) / 2
    not_too_small = half_size > too_small_threshold
    valid = valid & not_too_small

    # Apply margin
    shift_up = shift_up * half_size
    half_size *= 1 + margin

    # Calculate the square bounding box coordinates
    x_min = center_coords[..., 0] - half_size / aspect_ratio
    x_max = center_coords[..., 0] + half_size / aspect_ratio
    y_min = center_coords[..., 1] - half_size - shift_up
    y_max = center_coords[..., 1] + half_size - shift_up

    # Stack to get the final bounding box tensor
    bbox = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    # Create a full image bounding box of [0, 0, 1, 1]
    full_image_bbox = torch.tensor([0.0, 0.0, 1.0, 1.0], device=landmarks.device)

    # Overwrite invalid entries with the full image bounding box
    bbox = torch.where(valid.unsqueeze(-1), bbox, full_image_bbox)

    if clamp:
        return bbox.clamp(0, 1)
    return bbox
