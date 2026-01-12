import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from roma import rotvec_to_rotmat
from scipy import ndimage

from sheap import inference_images_list, load_sheap_model, render_mesh
from sheap.flame_segmentation import create_binary_mask_texture
from sheap.tiny_flame import TinyFlame, pose_components_to_rotmats

os.environ["PYOPENGL_PLATFORM"] = "egl"


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in binary mask (e.g., open mouth region)."""
    # Invert: black (0) -> white (255), white (255) -> black (0)
    inverted = 255 - mask
    
    # Find connected components in inverted image
    labeled, num_features = ndimage.label(inverted)
    
    if num_features > 1:
        # Calculate size of each component
        component_sizes = ndimage.sum(inverted, labeled, range(1, num_features + 1))
        
        # Find largest component (main background)
        largest_component_label = np.argmax(component_sizes) + 1
        
        # Keep only largest black region, fill all other black regions with white
        largest_component_mask = (labeled == largest_component_label)
        mask = np.where(largest_component_mask, 0, 255).astype(np.uint8)
    
    return mask


def rotmat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to XYZ Euler angles (in radians)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def extract_head_orientation(predictions: dict, frame_idx: int) -> dict:
    """Extract head orientation as Euler angles and direction vectors."""
    torso_rotvec = predictions["torso_pose"][frame_idx].cpu()
    neck_rotvec = predictions["neck_pose"][frame_idx].cpu()
    
    torso_rotmat = rotvec_to_rotmat(torso_rotvec.unsqueeze(0))[0]
    neck_rotmat = rotvec_to_rotmat(neck_rotvec.unsqueeze(0))[0]
    
    head_rotmat = (torso_rotmat @ neck_rotmat).numpy()
    euler_xyz = rotmat_to_euler_xyz(head_rotmat)
    
    right_vec = head_rotmat[:, 0]
    up_vec = head_rotmat[:, 1]
    forward_vec = -head_rotmat[:, 2]
    
    return {
        "euler_xyz_radians": euler_xyz.tolist(),
        "forward": forward_vec.tolist(),
        "up": up_vec.tolist(),
        "right": right_vec.tolist(),
    }


def render_flame_mask(
    verts: torch.Tensor,
    faces: torch.Tensor,
    c2w: torch.Tensor,
    output_size: int = 512,
) -> np.ndarray:
    """Render binary FLAME mask with holes filled."""
    mask_verts, mask_faces, mask_colors = create_binary_mask_texture(verts, faces)
    mask_render, _ = render_mesh(
        verts=mask_verts,
        faces=mask_faces,
        c2w=c2w,
        img_width=output_size,
        img_height=output_size,
        render_normals=False,
        render_segmentation=True,
        vertex_colors=mask_colors,
        black_background=True
    )
    
    # Convert to grayscale and fill holes (mouth region)
    mask_gray = mask_render[:, :, 0]  # Take single channel
    mask_filled = fill_mask_holes(mask_gray)
    
    return mask_filled


if __name__ == "__main__":
    # Load SHeaP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sheap_model = load_sheap_model(model_type="expressive").to(device)

    # Inference on example images
    folder_containing_images = Path("example_images/")
    image_paths = list(sorted(folder_containing_images.glob("*.jpg")))
    with torch.no_grad():
        predictions = inference_images_list(
            model=sheap_model,
            device=device,
            image_paths=image_paths,
        )

    # Load and infer FLAME with our predicted parameters
    flame_dir = Path("FLAME2020/")
    flame = TinyFlame(flame_dir / "generic_model.pt", eyelids_ckpt=flame_dir / "eyelids.pt")
    verts = flame(
        shape=predictions["shape_from_facenet"],
        expression=predictions["expr"],
        pose=pose_components_to_rotmats(predictions),
        eyelids=predictions["eyelids"],
        translation=predictions["cam_trans"],
    )

    # Render the FLAME mesh for each input image
    c2w = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.float32
    )
    for i_frame in range(verts.shape[0]):
        # Create output folder named after the image
        output_dir = image_paths[i_frame].parent / image_paths[i_frame].stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Render and save FLAME mask
        mask = render_flame_mask(
            verts=verts[i_frame],
            faces=flame.faces,
            c2w=c2w,
            output_size=512,
        )
        Image.fromarray(mask).save(output_dir / "flame_segmentation.png")

        # Extract and save head orientation
        orientation = extract_head_orientation(predictions, i_frame)
        with open(output_dir / "head_orientation.json", "w") as f:
            json.dump(orientation, f, indent=2)
