#!/usr/bin/env python
"""Gradio UI for the glasses / no-glasses CNN classifier.

Loads the model trained in cv_to_cnn_final.py and reproduces its exact
preprocessing (resize -> RGB -> normalize -> + Sobel-Y channel) so that
predictions made here match what the training script would produce.
"""

import os

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

# --------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------

CANDIDATE_MODEL_PATHS = [
    os.environ.get("MODEL_PATH", ""),
    os.path.join(os.path.dirname(__file__), "glasses_classifier.keras"),
]

MODEL_PATH = next((p for p in CANDIDATE_MODEL_PATHS if p and os.path.exists(p)), None)

if MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find glasses_classifier.keras. Place it next to app.py, "
        "or set the MODEL_PATH environment variable to its location."
    )

model = tf.keras.models.load_model(MODEL_PATH)

# Labels were encoded during training as: 0 = glasses, 1 = noglasses
CLASS_NAMES = ["Glasses", "No Glasses"]

# A couple of images from the training set, used as clickable examples.
# These live inside the app's own directory so Gradio's file-serving
# sandbox allows them (paths outside cwd/tmp are rejected in Gradio 6.x).
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
EXAMPLES = [
    os.path.join(EXAMPLES_DIR, "glasses_example.jpg"),
    os.path.join(EXAMPLES_DIR, "noglasses_example.jpg"),
]
EXAMPLES = [p for p in EXAMPLES if os.path.exists(p)]


# --------------------------------------------------------------------
# Preprocessing (mirrors preprocess_image in cv_to_cnn_final.py)
# --------------------------------------------------------------------

def build_model_input(image_rgb, size=128):
    """image_rgb: HxWx3 uint8 array, as produced by gr.Image(type='numpy')."""
    resized = cv2.resize(image_rgb, (size, size))
    normalized = resized.astype(np.float32) / 255.0

    gray = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y).astype(np.float32) / 255.0
    sobel_display = (sobel_y * 255).astype(np.uint8)
    sobel_y = np.expand_dims(sobel_y, axis=-1)

    combined = np.concatenate((normalized, sobel_y), axis=-1)
    combined = np.expand_dims(combined, axis=0)  # (1, size, size, 4)

    return combined, resized, sobel_display


# --------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------

def predict(image, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Please upload a photo first.")

    progress(0.15, desc="Preprocessing image...")
    model_input, resized_preview, sobel_preview = build_model_input(image)

    progress(0.55, desc="Running the CNN...")
    raw_score = float(model.predict(model_input, verbose=0)[0][0])

    progress(0.9, desc="Formatting results...")
    p_noglasses = raw_score
    p_glasses = 1.0 - raw_score
    confidences = {"Glasses": p_glasses, "No Glasses": p_noglasses}

    progress(1.0, desc="Done")
    return confidences, resized_preview, sobel_preview


# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container {max-width: 960px !important; margin: auto;}
#title_md h1 {margin-bottom: 0.1em;}
#title_md p {color: var(--body-text-color-subdued); margin-top: 0;}
footer {display: none !important;}
"""

with gr.Blocks(title="Glasses Detector") as demo:
    gr.Markdown(
        "# 🕶️ Glasses Detector\n"
        "Upload a face photo and the CNN will predict whether the person is wearing glasses.",
        elem_id="title_md",
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload a photo",
                type="numpy",
                sources=["upload", "clipboard"],
                height=320,
            )
            predict_btn = gr.Button("Predict", variant="primary")
            if EXAMPLES:
                gr.Examples(examples=EXAMPLES, inputs=image_input, label="Try an example")

        with gr.Column(scale=1):
            result_label = gr.Label(label="Prediction", num_top_classes=2)
            with gr.Row():
                resized_output = gr.Image(label="Model input (128x128)", height=150)
                sobel_output = gr.Image(label="Sobel-Y edge map", height=150)

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[result_label, resized_output, sobel_output],
    )
    image_input.change(
        fn=predict,
        inputs=image_input,
        outputs=[result_label, resized_output, sobel_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_port=int(os.environ.get("PORT", 9001)),
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CUSTOM_CSS,
    )
