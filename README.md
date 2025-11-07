# 🐑 SHeaP 🐑
Code and models for inferencing [SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians](https://nlml.github.io/sheap).

## Example usage

After setting up, run `python demo.py`.

## Setup

### Step 1: Install dependencies

We just require `torch>=2.0.0` and a few other dependencies.

Just install the latest `torch` in a new venv, then `pip install .`

Or, if you use [`uv`](https://docs.astral.sh/uv/), you can just run `uv sync`.

### Step 2: Download and convert FLAME

Download [FLAME2020](https://flame.is.tue.mpg.de/).

Put it in the `FLAME2020/` dir. We only need gerneric_model.pkl. Your `FLAME2020/` directory should look like this:

```bash
FLAME2020/
├── eyelids.pt
├── flame_landmark_idxs_barys.pt
└── generic_model.pkl
```

Now convert FLAME to our format:

```bash
python convert_flame.py
```

### Done!

## Reproduce paper results on NoW dataset

To reproduce the validation results from the paper (median=0.93mm):

First, update submodules:

```bash
git submodule update --init --recursive
```

Then build the NoW Evaluation docker image:

```bash
docker build -t noweval now/now_evaluation
```

Then predict FLAME meshes for all images in NoW using SHeaP:

```
cd now/
python now.py --now-dataset-root /path/to/NoW_Evaluation/dataset
```

Upon finishing, the above command will print a command like the following:

```
chmod 777 -R /home/user/sheap/now/now_eval_outputs/now_preds && docker run --ipc host --gpus all -it --rm -v /data/NoW_Evaluation/dataset:/dataset -v /home/user/sheap/now/now_eval_outputs/now_preds:/preds noweval
```

Run that command. This will run NoW evaluation on the FLAME meshes we just predicted.

Finally, the results will be placed in `/home/user/sheap/now/now_eval_outputs/now_preds` (or equivalent). The mean and median are already calculated:

```bash
➜ cat /home/user/sheap/now/now_eval_outputs/now_preds/results/RECON_computed_distances.npy.meanmedian
0.9327719333872148  # result in the paper
1.1568168246248534
```
