import torch


def vertices_to_landmarks(
    vertices,  # Float[Tensor, "*batch num_vertices 3"],
    faces,  # Int[Tensor, "num_faces 3"],
    face_indices_with_landmarks,  # Int[Tensor, " num_landmarks"],  # type: ignore
    barys,  # Float[Tensor, " num_landmarks 3"],
):
    """
    Calculate the 3D locations of the landmarks from the vertices of the mesh.

    Args:
    - vertices: Mesh vertices.
    - faces: Mesh faces (each face contains the indices of its three vertices).
    - face_indices_with_landmarks: Indices of the faces that contain the landmarks.
    - barys: Barycentric coordinates of the landmarks in the faces (last dimension
      should sum to 1.0).

    Returns: computed 3D coordinates of the landmarks.
    """
    did_unsqueeze = False
    if vertices.ndim == 2:  # Now supporting even the no batch dims case!
        vertices = vertices.unsqueeze(0)
        did_unsqueeze = True

    batch_dims = vertices.shape[:-2]
    # First grab just the faces that contain the landmarks
    relevant_faces = faces[face_indices_with_landmarks]

    # Now select the vertices that are part of the relevant faces
    selected_vertices = torch.index_select(vertices, len(batch_dims), relevant_faces.view(-1)).view(
        *batch_dims, *relevant_faces.shape, 3
    )
    # The next two lines are eqiuvalent:
    # (selected_vertices * barys.view([1] * len(batch_dims) + list(barys.shape) + [1])).sum(dim=-2)
    landmark_positions = torch.einsum("b...lvx,lv->b...lx", selected_vertices, barys)
    if did_unsqueeze:
        landmark_positions = landmark_positions[0]
    return landmark_positions


def vertices_to_7_lmks(
    vertices, flame_faces, face_alignment_lmk_faces_idx, face_alignment_lmk_bary_coords
):
    lmks_3d = vertices_to_landmarks(
        vertices,
        flame_faces,
        face_alignment_lmk_faces_idx,
        face_alignment_lmk_bary_coords,
    )
    landmark_51 = lmks_3d[:, 17:]
    lmks7_3d = landmark_51[:, [19, 22, 25, 28, 16, 31, 37]]
    return lmks7_3d, lmks_3d
