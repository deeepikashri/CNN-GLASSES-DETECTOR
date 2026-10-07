# 🕶️ Glasses Detector

A CNN that classifies whether a person in a photo is wearing glasses, served
through a Gradio web UI. RGB input is combined with a Sobel-Y edge channel
(4 channels total) to give the model an explicit signal for glasses' frame
edges, on top of raw color.

## Project structure

```
glasses-detector/
├── app.py                    # Gradio inference app
├── train_model.py            # Training script (data prep, CNN, training, eval)
├── examples/                 # Optional sample photos for the UI's "Try an example"
├── requirements.txt          # Runtime deps for app.py
├── requirements-train.txt    # Extra deps for train_model.py
└── glasses_classifier.keras  # Trained model (you provide this — see below)
```

## How it works

- **Input:** a face photo, resized to 128×128, RGB, normalized to [0, 1].
- **Extra channel:** a Sobel-Y edge map (grayscale → vertical gradient →
  normalized) is appended as a 4th channel, so the model sees both color and
  edge structure.
- **Model:** a 3-block CNN (Conv2D + MaxPooling ×3) → Flatten → Dense(128) →
  Dropout(0.5) → Dense(1, sigmoid). Binary output: `0 = glasses`,
  `1 = no glasses`.
- **Training augmentation:** each training image is rotated (45°/120°/150°)
  and flipped (horizontal/vertical), plus a synthetic low-light variant of
  every geometric variant (gamma-darkened with a soft glow and sensor noise),
  to make the model robust to lighting and pose.

`app.py` reproduces the exact same preprocessing used in `train_model.py` so
inference matches training.

## Setup

```bash
git clone https://github.com/<your-username>/glasses-detector.git
cd glasses-detector
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Get a trained model

`app.py` needs `glasses_classifier.keras` next to it. Either:

- Train your own with `train_model.py` (see below), or
- Drop an existing `glasses_classifier.keras` file into the project root, or
- Point at a model anywhere else with an environment variable:
  ```bash
  export MODEL_PATH=/path/to/glasses_classifier.keras
  ```

The model file isn't committed to this repo (it's in `.gitignore` since
Keras model files can be large) — use [Git LFS](https://git-lfs.com/) if you
want to version it directly on GitHub, or host it externally (e.g. a GitHub
Release, Hugging Face Hub, or cloud storage) and download it in your deploy
step.

### Run the app

```bash
python app.py
```

Opens at `http://localhost:9001` (override with `PORT=xxxx`).

## Training your own model

```bash
pip install -r requirements.txt -r requirements-train.txt
```

`train_model.py` was converted from a Jupyter notebook and expects a dataset
laid out as:

```
<DATASET_BASE_PATH>/
├── train/
│   ├── glasses/
│   └── noglasses/
├── validate/
└── test/
```

Configure paths via environment variables (defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `DATASET_ZIP_PATH` | `./data/archive.zip` | Optional zip to extract before training |
| `DATASET_EXTRACT_DIR` | `./data/extracted` | Where the zip is extracted |
| `DATASET_BASE_PATH` | `./data/glasses-noglasses` | Root folder containing `train/validate/test` |
| `MODEL_OUTPUT_PATH` | `./glasses_classifier.keras` | Where the trained model is saved |
| `TEST_IMAGE_PATH`, `TEST_IMAGE_PATH_2` | `./data/test_image_1.jpg`, `./data/test_image_2.jpg` | Ad-hoc images used by the manual prediction cells at the end of the script |

Then run:

```bash
python train_model.py
```

(It still contains `# In[n]:` cell markers from its notebook origin — those
are comments only and don't affect execution.)

## Deploying the Gradio app

Any host that runs a long-lived Python process works — e.g. **Hugging Face
Spaces** (Gradio SDK, first-class support, free CPU tier), Render, Railway,
or your own VM. Steps are the same everywhere:

1. Make sure `glasses_classifier.keras` is available at deploy time (see
   above).
2. `pip install -r requirements.txt`
3. `python app.py` (respects the `PORT` env var)

### Hugging Face Spaces (recommended, easiest)

1. Create a new Space → SDK: **Gradio**.
2. Push this repo's contents to the Space's git remote.
3. Upload `glasses_classifier.keras` via the Spaces UI, or add it with Git
   LFS.
4. The Space builds `requirements.txt` and runs `app.py` automatically.

## Notes

- Labels: index `0 = Glasses`, `1 = No Glasses` (matches the `classes` list
  order in `train_model.py`).
- `opencv-python-headless` is used instead of `opencv-python` in
  `requirements.txt` since the app runs headless on a server (no local
  display needed).

## License

MIT — see `LICENSE`.
