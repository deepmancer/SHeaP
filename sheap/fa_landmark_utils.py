from typing import Tuple

import face_alignment
import numpy as np
import torch
from numpy.typing import NDArray

from sheap.landmark_utils import landmarks_2_face_bounding_box


def get_fa_landmarks(
    np_array_im_255_uint8: NDArray[np.uint8],
    fa: face_alignment.FaceAlignment,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract facial landmarks from an image using face_alignment.

    Args:
        np_array_im_255_uint8: Image array of shape (H, W, 3) with values in [0, 255]
        fa: FaceAlignment model instance
        normalize: If True, normalize landmarks to [0, 1] range

    Returns:
        Tuple of (landmarks, success):
            - landmarks: Tensor of shape (68, 2) with normalized coordinates
            - success: Boolean tensor indicating if face was detected
    """
    preds = fa.get_landmarks(np_array_im_255_uint8)
    if preds is not None:
        if normalize:
            h, w = np_array_im_255_uint8.shape[:2]
            lmks = preds[0][:, :2] / np.array([w, h])
        else:
            lmks = preds[0][:, :2]
        success = True
    else:
        lmks = np.zeros((68, 2))
        success = False

    lmks_tensor = torch.from_numpy(lmks).float()
    success_tensor = torch.tensor(success).bool()
    return lmks_tensor, success_tensor


def detect_face_and_crop(
    image: torch.Tensor,
    fa_model: face_alignment.FaceAlignment,
    margin: float = 0.6,
    shift_up: float = 0.2,
) -> Tuple[int, int, int, int]:
    """
    Detect face and compute bounding box coordinates.

    Args:
        image: torch.Tensor of shape (3, H, W) with values in [0, 1]
        fa_model: FaceAlignment model instance for landmark detection

    Returns:
        tuple: (x0, x1, y0, y1) bounding box coordinates in pixels
    """
    _, h, w = image.shape

    # Convert image to numpy format for face_alignment (H, W, 3) with values [0, 255]
    image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Get facial landmarks
    lmks, success = get_fa_landmarks(image_np, fa_model, normalize=True)

    if not success:
        # If face detection fails, return center square from image
        if h > w:
            y0 = (h - w) // 2
            y1 = y0 + w
            x0 = 0
            x1 = w
        else:
            x0 = (w - h) // 2
            x1 = x0 + h
            y0 = 0
            y1 = h
        return x0, x1, y0, y1

    # Add batch dimension for landmarks_2_face_bounding_box
    lmks_batched = lmks.unsqueeze(0)  # Shape: (1, 68, 2)
    valid = torch.ones(1, dtype=torch.bool)

    # Compute bounding box in normalized coordinates
    bbox = landmarks_2_face_bounding_box(
        lmks_batched, valid, margin=margin, clamp=True, shift_up=shift_up, aspect_ratio=w / h
    )

    x0, y0, x1, y1 = bbox[0].tolist()
    x0, y0, x1, y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

    return x0, y0, x1, y1
