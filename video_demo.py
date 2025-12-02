import argparse
import os
import shutil
import subprocess
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from demo import create_rendering_image
from sheap import load_sheap_model
from sheap.tiny_flame import TinyFlame, pose_components_to_rotmats

try:
    import face_alignment
except ImportError:
    raise ImportError(
        "The 'face_alignment' package is required. Please install it via 'pip install face-alignment'."
    )
from sheap.fa_landmark_utils import detect_face_and_crop


class RenderingThread(threading.Thread):
    """Background thread for rendering frames to images."""

    def __init__(
        self,
        render_queue: Queue,
        temp_dir: Path,
        faces: torch.Tensor,
        c2w: torch.Tensor,
        render_size: int,
    ):
        """
        Initialize rendering thread.

        Args:
            render_queue: Queue containing (frame_idx, cropped_frame, verts) tuples
            temp_dir: Directory to save rendered images
            faces: Face indices tensor from FLAME model
            c2w: Camera-to-world transformation matrix
            render_size: Size of each sub-image in the rendered output
        """
        super().__init__(daemon=True)
        self.render_queue = render_queue
        self.temp_dir = temp_dir
        self.faces = faces
        self.c2w = c2w
        self.render_size = render_size
        self.stop_event = threading.Event()
        self.frames_rendered = 0

    def run(self):
        """Process rendering queue until stop signal is received."""
        # Set PyOpenGL platform for this thread
        os.environ["PYOPENGL_PLATFORM"] = "egl"

        while not self.stop_event.is_set():
            try:
                # Get item from queue with timeout to allow checking stop_event
                try:
                    item = self.render_queue.get(timeout=0.1)
                except Empty:  # Haven't finished, but nothing to render yet
                    continue
                if item is None:  # Sentinel value to stop
                    break

                frame_idx, cropped_frame, verts = item
                frame_idx, cropped_frame, verts = item

                # Render the frame
                cropped_pil = Image.fromarray(cropped_frame)
                combined = create_rendering_image(
                    original_image=cropped_pil,
                    verts=verts,
                    faces=self.faces,
                    c2w=self.c2w,
                    output_size=self.render_size,
                )

                # Save to temp directory with zero-padded frame number
                output_path = self.temp_dir / f"frame_{frame_idx:06d}.png"
                combined.save(output_path)

                self.frames_rendered += 1
                self.render_queue.task_done()

            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"Error rendering frame: {e}")
                    import traceback

                    traceback.print_exc()

    def stop(self):
        """Signal the thread to stop."""
        self.stop_event.set()


class VideoFrameDataset(IterableDataset):
    """Iterable dataset for streaming video frames with face detection and cropping.

    Uses a background thread for video frame loading while face detection runs in the main thread.
    """

    def __init__(
        self,
        video_path: str,
        fa_model: face_alignment.FaceAlignment,
        smoothing_alpha: float = 0.3,
        frame_buffer_size: int = 32,
    ):
        """
        Initialize video frame dataset.

        Args:
            video_path: Path to video file
            fa_model: FaceAlignment model instance for face detection
            smoothing_alpha: Smoothing factor for bounding box (0=no smoothing, 1=no change).
                           Lower values = more smoothing
            frame_buffer_size: Size of the frame buffer queue for the background thread
        """
        super().__init__()
        self.video_path = video_path
        self.fa_model = fa_model
        self.smoothing_alpha = smoothing_alpha
        self.frame_buffer_size = frame_buffer_size
        self.prev_bbox: Optional[Tuple[int, int, int, int]] = None

        # Get video metadata (don't keep capture open)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(
            f"Video info: {self.num_frames} frames, {self.fps:.2f} fps, {self.width}x{self.height}"
        )

    def _video_reader_thread(self, frame_queue: Queue, stop_event: threading.Event):
        """Background thread that reads video frames and puts them in a queue.

        Args:
            frame_queue: Queue to put (frame_idx, frame_rgb) tuples
            stop_event: Event to signal thread to stop
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            frame_queue.put(("error", f"Could not open video file: {self.video_path}"))
            return

        frame_idx = 0
        try:
            while not stop_event.is_set():
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Put frame in queue (blocks if queue is full)
                frame_queue.put((frame_idx, frame_rgb))
                frame_idx += 1

        finally:
            cap.release()
            # Signal end of video
            frame_queue.put(None)

    def __iter__(self):
        """
        Iterate through video frames sequentially.

        Video frame loading happens in a background thread, while face detection
        and processing happen in the main thread.

        Yields:
            Dictionary containing frame_idx, processed image, and bounding box
        """
        # Reset smoothing state for new iteration
        self.prev_bbox = None

        # Create queue and start background thread for video reading
        frame_queue = Queue(maxsize=self.frame_buffer_size)
        stop_event = threading.Event()
        reader_thread = threading.Thread(
            target=self._video_reader_thread, args=(frame_queue, stop_event), daemon=True
        )
        reader_thread.start()

        try:
            while True:
                # Get frame from background thread
                item = frame_queue.get()

                # Check for end of video
                if item is None:
                    break

                # Check for error
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "error":
                    raise RuntimeError(item[1])

                frame_idx, frame_rgb = item

                # Convert to torch tensor (C, H, W) with values in [0, 1]
                image = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

                # Detect face and crop (runs in main thread, can use GPU)
                bbox = detect_face_and_crop(image, self.fa_model, margin=0.9, shift_up=0.5)

                # Apply smoothing using exponential moving average
                bbox = self._smooth_bbox(bbox)
                x0, y0, x1, y1 = bbox

                cropped = image[:, y0:y1, x0:x1]

                # Resize to 224x224 for SHEAP model
                cropped_resized = TF.resize(cropped, [224, 224], antialias=True)
                cropped_for_render = TF.resize(cropped, [512, 512], antialias=True)

                yield {
                    "frame_idx": frame_idx,
                    "image": cropped_resized,
                    "bbox": bbox,
                    "original_frame": frame_rgb,  # Keep original for reference (as numpy array)
                    "cropped_frame": cropped_for_render,  # Cropped region resized to 512x512
                }

        finally:
            # Clean up background thread
            stop_event.set()
            reader_thread.join(timeout=1.0)

    def _smooth_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Apply exponential moving average smoothing to bounding box."""
        if self.prev_bbox is None:
            self.prev_bbox = bbox
            return bbox

        x0, y0, x1, y1 = bbox
        prev_x0, prev_y0, prev_x1, prev_y1 = self.prev_bbox

        # Smooth: new_bbox = alpha * detected_bbox + (1 - alpha) * prev_bbox
        smoothed = (
            int(self.smoothing_alpha * x0 + (1 - self.smoothing_alpha) * prev_x0),
            int(self.smoothing_alpha * y0 + (1 - self.smoothing_alpha) * prev_y0),
            int(self.smoothing_alpha * x1 + (1 - self.smoothing_alpha) * prev_x1),
            int(self.smoothing_alpha * y1 + (1 - self.smoothing_alpha) * prev_y1),
        )

        self.prev_bbox = smoothed
        return smoothed

    def __len__(self) -> int:
        return self.num_frames


def process_video(
    video_path: str,
    model_type: str = "expressive",
    batch_size: int = 1,
    num_workers: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_video_path: Optional[str] = None,
    render_size: int = 512,
    num_render_workers: int = 1,
    max_queue_size: int = 128,
) -> List[Dict[str, Any]]:
    """
    Process video frames through SHEAP model and optionally render output video.

    Uses an IterableDataset for efficient sequential video processing without seeking overhead.
    Rendering is done in a background thread, and ffmpeg is used to create the final video.

    Args:
        video_path: Path to video file
        model_type: SHEAP model variant ("paper", "expressive", or "lightweight")
        batch_size: Batch size for processing
        num_workers: Number of workers (0 or 1 only). Will be clamped to max 1.
        device: Device to run model on ("cpu" or "cuda")
        output_video_path: If provided, render and save output video to this path
        render_size: Size of each sub-image in the rendered output
        num_render_workers: Number of background threads for rendering
        max_queue_size: Maximum size of the rendering queue

    Returns:
        List of dictionaries containing frame index, bounding box, and FLAME parameters
    """
    # Enforce num_workers constraint for IterableDataset
    num_workers = min(num_workers, 1)
    if num_workers > 1:
        print(f"Warning: num_workers > 1 not supported with IterableDataset. Using num_workers=1.")

    # Load SHEAP model
    print(f"Loading SHEAP model (type: {model_type})...")
    sheap_model = load_sheap_model(model_type=model_type)
    sheap_model.eval()
    sheap_model = sheap_model.to(device)

    # Load face alignment model
    # Force CPU for FA when using num_workers=1 (subprocess issues with GPU)
    fa_device = "cpu" if num_workers >= 1 else device
    print(f"Loading face alignment model on {fa_device}...")
    fa_model = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.THREE_D, flip_input=False, device=fa_device
    )

    # Create dataset and dataloader
    dataset = VideoFrameDataset(video_path, fa_model)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Processing {len(dataset)} frames from {video_path}")

    # Initialize FLAME model and rendering thread if rendering
    flame = None
    rendering_threads = []
    render_queue = None
    temp_dir = None
    c2w = None

    if output_video_path:
        print("Loading FLAME model for rendering...")
        flame_dir = Path("FLAME2020/")
        flame = TinyFlame(flame_dir / "generic_model.pt", eyelids_ckpt=flame_dir / "eyelids.pt")
        flame = flame.to(device)  # Move FLAME to GPU
        c2w = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.float32
        )

        # Create temporary directory for rendered frames
        temp_dir = Path("./temp_sheap_render/")
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using temporary directory: {temp_dir}")

        # Start multiple background rendering threads
        render_queue = Queue(maxsize=max_queue_size)
        for _ in range(num_render_workers):
            thread = RenderingThread(render_queue, temp_dir, flame.faces, c2w, render_size)
            thread.start()
            rendering_threads.append(thread)
        print(f"Started {num_render_workers} background rendering threads")

    results = []
    frame_count = 0

    with torch.no_grad():
        progbar = tqdm(total=len(dataset), desc="Processing frames")
        for batch in dataloader:
            frame_indices = batch["frame_idx"]
            images = batch["image"].to(device)
            bboxes = batch["bbox"]

            # Process through SHEAP model
            flame_params_dict = sheap_model(images)

            # Generate vertices for this batch if rendering
            if output_video_path and flame is not None:
                verts = flame(
                    shape=flame_params_dict["shape_from_facenet"],
                    expression=flame_params_dict["expr"],
                    pose=pose_components_to_rotmats(flame_params_dict),
                    eyelids=flame_params_dict["eyelids"],
                    translation=flame_params_dict["cam_trans"],
                )

            # Store results and queue for rendering
            for i in range(len(frame_indices)):
                frame_idx = _extract_scalar(frame_indices[i])
                bbox = tuple(_extract_scalar(b[i]) for b in bboxes)

                result = {
                    "frame_idx": frame_idx,
                    "bbox": bbox,
                    "flame_params": {k: v[i].cpu() for k, v in flame_params_dict.items()},
                }
                results.append(result)

                # Queue frame for rendering
                if output_video_path:
                    cropped_frame = _tensor_to_numpy_image(batch["cropped_frame"][i])
                    render_queue.put((frame_idx, cropped_frame, verts[i].cpu()))
                    frame_count += 1

            progbar.update(len(frame_indices))
        progbar.close()

    # Finalize rendering and create output video
    if output_video_path and render_queue is not None:
        _finalize_rendering(
            rendering_threads,
            render_queue,
            num_render_workers,
            temp_dir,
            dataset.fps,
            output_video_path,
        )

    return results


def _extract_scalar(value: Any) -> int:
    """Extract scalar integer from tensor or return as-is."""
    return value.item() if isinstance(value, torch.Tensor) else value


def _tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor [0, 1] to numpy (H, W, C) uint8 [0, 255]."""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _finalize_rendering(
    rendering_threads: List[RenderingThread],
    render_queue: Queue,
    num_render_workers: int,
    temp_dir: Path,
    fps: float,
    output_video_path: str,
) -> None:
    """Finish rendering threads and create final video with ffmpeg."""
    print("\nWaiting for rendering threads to complete...")

    # Add sentinel values to stop workers
    for _ in range(num_render_workers):
        render_queue.put(None)

    # Wait for all threads to finish
    for thread in rendering_threads:
        thread.join()

    total_rendered = sum(thread.frames_rendered for thread in rendering_threads)
    print(f"Rendered {total_rendered} frames")

    # Create video with ffmpeg
    print("Creating video with ffmpeg...")
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-framerate",
        str(fps),
        "-i",
        str(temp_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "23",
        str(output_path),
    ]

    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    print(f"Video saved to: {output_video_path}")

    # Clean up temporary directory
    if temp_dir.exists():
        print(f"Removing temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        print("Cleanup complete")


if __name__ == "__main__":
    # video_path = "skarsgard.mp4"
    # output_video_path = "skarsgard_rendered.mp4"
    parser = argparse.ArgumentParser(description="Process and render video with SHEAP model.")
    parser.add_argument("in_path", type=str, help="Path to input video file.")
    parser.add_argument(
        "--out_path", type=str, help="Path to save rendered output video.", default=None
    )
    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = str(Path(args.in_path).with_name(f"{Path(args.in_path).stem}_rendered.mp4"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = process_video(
        video_path=args.in_path,
        model_type="expressive",
        device=device,
        output_video_path=args.out_path,
    )
