import urllib.request
from pathlib import Path

import torch

MODEL_PAPER_URL = "https://github.com/nlml/sheap/releases/download/v1.0.0/model_paper.pt"
MODEL_EXPRESSIVE_URL = "https://github.com/nlml/sheap/releases/download/v1.0.0/model_expressive.pt"
MODEL_FILENAME_PAPER = "model_paper.pt"
MODEL_FILENAME_EXPRESSIVE = "model_expressive.pt"


def ensure_model_downloaded(paper_model=True, models_dir=Path("./models")):
    model_url = MODEL_PAPER_URL if paper_model else MODEL_EXPRESSIVE_URL
    model_path = models_dir / (MODEL_FILENAME_PAPER if paper_model else MODEL_FILENAME_EXPRESSIVE)
    if not model_path.exists():
        print(f"Downloading model to {model_path}...")
        model_path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)


def load_sheap_model(paper_model=True, models_dir=Path("./models")):
    """
    Loads the SHeaP model as a PyTorch jit trace.
    Downloads the model if not already present.

    Parameters
    ----------
    paper_model : bool
        If True, loads the model used in the original SHeaP paper.
        If False, loads the more expressive model, trained for longer.

    models_dir : Path
        The directory where the model is stored.

    Returns
    -------
    sheap_model : torch.jit.ScriptModule
        The loaded SHeaP model.
    """
    models_dir = Path(models_dir)
    ensure_model_downloaded(paper_model=paper_model, models_dir=models_dir)
    if paper_model:
        sheap_model = torch.load(models_dir / MODEL_FILENAME_PAPER, weights_only=False)
    else:
        sheap_model = torch.load(models_dir / MODEL_FILENAME_EXPRESSIVE, weights_only=False)
    return sheap_model
