"""
Gradio demo for SHeaP (Self-Supervised Head Geometry Predictor).
Accepts video or image input and renders the SHEAP output overlayed.
"""

import os

# CRITICAL: Set PYOPENGL_PLATFORM before any OpenGL/pyrender imports
# This must happen before importing demo.py or sheap.render
# Note: HF Spaces with GPU will have EGL available, use osmesa for CPU-only environments
if "PYOPENGL_PLATFORM" not in os.environ:
    # Default to egl (works on HF Spaces with GPU)
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import shutil
import subprocess
import tempfile
from pathlib import Path
from queue import Queue
from typing import Optional

import face_alignment
import gradio as gr
import numpy as np
import torch
import torch.hub
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader

try:
    import spaces

    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

    # Define a no-op decorator for local development
    class spaces:
        @staticmethod
        def GPU(func):
            return func


from demo import create_rendering_image
from sheap import load_sheap_model
from sheap.fa_landmark_utils import detect_face_and_crop
from sheap.tiny_flame import TinyFlame, pose_components_to_rotmats
from video_demo import RenderingThread, VideoFrameDataset, _tensor_to_numpy_image

# Global variables for models (load once)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sheap_model = None
flame = None
fa_model = None
c2w = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.float32)


def initialize_models():
    """Initialize all models (called lazily on first use)."""
    global sheap_model, flame, fa_model

    if sheap_model is not None:
        return  # Already initialized

    print("Loading SHeaP model...", flush=True)
    sheap_model = load_sheap_model(model_type="expressive").to(device)
    sheap_model.eval()

    print("Loading FLAME model...", flush=True)
    flame_dir = Path("FLAME2020/")
    flame = TinyFlame(flame_dir / "generic_model.pt", eyelids_ckpt=flame_dir / "eyelids.pt").to(
        device
    )

    print("Loading face alignment model...", flush=True)
    # Set torch hub cache to local directory to use pre-downloaded models
    torch.hub.set_dir(str(Path(__file__).parent / "face_alignment_cache"))
    fa_model = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device=str(device), flip_input=False
    )

    print("Models loaded successfully!", flush=True)


@spaces.GPU
def process_image(image: np.ndarray) -> Image.Image:
    """
    Process a single image and return the rendered output.

    Args:
        image: Input image as numpy array (RGB)

    Returns:
        PIL Image with three views side-by-side (original, mesh, blended)
    """
    # Initialize models on first use (lazy loading for @spaces.GPU)
    initialize_models()

    # Convert to torch tensor for face detection (C, H, W) format with values in [0, 1]
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    # Detect face and get crop coordinates
    x0, y0, x1, y1 = detect_face_and_crop(image_tensor, fa_model, margin=0.9, shift_up=0.5)

    # Crop the image
    cropped_tensor = image_tensor[:, y0:y1, x0:x1]

    # Resize to 224x224 for SHEAP model
    cropped_resized = TF.resize(cropped_tensor, [224, 224], antialias=True)

    # Prepare input tensor for model
    img_tensor = cropped_resized.unsqueeze(0).to(device)

    # Also create a 512x512 version for rendering
    cropped_for_render = TF.resize(cropped_tensor, [512, 512], antialias=True)

    # Run inference
    with torch.no_grad():
        predictions = sheap_model(img_tensor)

        # Get FLAME vertices (predictions are already on device from model)
        verts = flame(
            shape=predictions["shape_from_facenet"],
            expression=predictions["expr"],
            pose=pose_components_to_rotmats(predictions),
            eyelids=predictions["eyelids"],
            translation=predictions["cam_trans"],
        )

        # Move vertices to CPU for rendering
        verts = verts.cpu()

    # Convert cropped_for_render back to PIL Image for rendering
    cropped_pil = TF.to_pil_image(cropped_for_render)

    # Create rendering
    combined = create_rendering_image(
        original_image=cropped_pil,
        verts=verts[0],
        faces=flame.faces,
        c2w=c2w,
        output_size=512,
    )

    return combined


@spaces.GPU
def process_video_frames(video_path: str, temp_dir: Path, progress=gr.Progress()):
    """
    Process video frames with GPU (inference and rendering).
    Returns fps and number of frames processed.
    """
    # Initialize models on first use (lazy loading for @spaces.GPU)
    initialize_models()

    render_size = 512
    # Prepare dataset and dataloader
    dataset = VideoFrameDataset(video_path, fa_model)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    fps = dataset.fps
    num_frames = len(dataset)
    # Prepare rendering thread and queue
    render_queue = Queue(maxsize=32)
    num_render_workers = 1
    rendering_threads = []
    for _ in range(num_render_workers):
        thread = RenderingThread(render_queue, temp_dir, flame.faces, c2w, render_size)
        thread.start()
        rendering_threads.append(thread)
    progress(0, desc="Processing video frames...")
    frame_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            cropped_frames = batch["cropped_frame"]
            # Run inference
            predictions = sheap_model(images)
            verts = flame(
                shape=predictions["shape_from_facenet"],
                expression=predictions["expr"],
                pose=pose_components_to_rotmats(predictions),
                eyelids=predictions["eyelids"],
                translation=predictions["cam_trans"],
            )
            verts = verts.cpu()
            for i in range(images.shape[0]):
                cropped_frame = _tensor_to_numpy_image(cropped_frames[i])
                render_queue.put((frame_idx, cropped_frame, verts[i]))
                frame_idx += 1
                progress(
                    0.95 * frame_idx / num_frames, desc=f"Processing frame {frame_idx}/{num_frames}"
                )
    # Stop rendering threads
    for _ in range(num_render_workers):
        render_queue.put(None)
    for thread in rendering_threads:
        thread.join()
    if frame_idx == 0:
        raise ValueError("No frames were successfully processed!")

    return fps, frame_idx


def process_video(video_path: str, progress=gr.Progress()) -> str:
    """
    Process a video and return path to the rendered output video.
    """
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Process frames with GPU
        fps, num_frames = process_video_frames(video_path, temp_dir, progress)

        # Create output video using ffmpeg (CPU-only, outside GPU context)
        progress(0.95, desc="Encoding video...")
        output_path = temp_dir / "output.mp4"
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_path),
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        progress(1.0, desc="Done!")
        return str(output_path)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


def process_input(image: Optional[np.ndarray], video: Optional[str]):
    """
    Process either image or video input.

    Args:
        image: Input image (if provided)
        video: Input video path (if provided)

    Returns:
        Either an image or video path depending on input
    """
    if image is not None:
        return process_image(image), None
    elif video is not None:
        return None, process_video(video)
    else:
        raise ValueError("Please provide either an image or video!")


# Don't initialize models at startup when using @spaces.GPU
# They will be loaded lazily on first use
# initialize_models()

# Create Gradio interface
with gr.Blocks(title="SHeaP Demo") as demo:
    gr.Markdown(
        """
    # 🐑 SHeaP: Self-Supervised Head Geometry Predictor 🐑

    Upload an image or video to predict head geometry and render a 3D mesh overlay!

    The output shows three views:
    - **Left**: Original cropped face
    - **Center**: Rendered FLAME mesh
    - **Right**: Mesh overlaid on original

    [Project Page](https://nlml.github.io/sheap) | [Paper](https://arxiv.org/abs/2504.12292) | [GitHub](https://github.com/nlml/sheap)
    """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input")
            image_input = gr.Image(label="Upload Image", type="numpy")
            video_input = gr.Video(label="Upload Video")
            process_btn = gr.Button("Process", variant="primary")

        with gr.Column():
            gr.Markdown("### Output")
            image_output = gr.Image(label="Rendered Image", type="pil")
            video_output = gr.Video(label="Rendered Video")

    gr.Markdown(
        """
    ### Tips:
    - For best results, use images/videos with clearly visible faces
    - The model works best with frontal face views
    - Video processing may take a few minutes depending on length
    """
    )

    # Connect the button
    process_btn.click(
        fn=process_input,
        inputs=[image_input, video_input],
        outputs=[image_output, video_output],
    )

    # Add examples
    gr.Examples(
        examples=[
            ["example_images/00000206.jpg", None],
            [None, "example_videos/dafoe.mp4"],
        ],
        inputs=[image_input, video_input],
        outputs=[image_output, video_output],
        fn=process_input,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
