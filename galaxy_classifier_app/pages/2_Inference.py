
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import preprocess_input


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Galaxy Morphology Classifier",
    page_icon="🔭",
    layout="wide",
)

# ============================================================
# 2. Project Paths
# ============================================================

# This file is inside galaxy_classifier_app/pages/
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "model-2"
    / "galaxy10_efficientnetb0_phase3.keras"
)

SAMPLES_DIR = BASE_DIR / "assets" / "samples"

# ============================================================
# 3. Six-Class Model Configuration
# ============================================================

CLASS_NAMES = [
    "Disturbed Galaxy",
    "Merging Galaxies",
    "Elliptical Galaxy",
    "Barred Spiral",
    "Unbarred Spiral",
    "Edge-on Galaxy",
]

CLASS_DESCRIPTIONS = {
    "Disturbed Galaxy":
        "Shows an asymmetric structure, tidal features, or irregular "
        "morphological disruption.",

    "Merging Galaxies":
        "Two or more gravitationally interacting galaxy cores "
        "in the process of merging.",

    "Elliptical Galaxy":
        "A smooth galaxy with an elliptical or rounded light profile.",

    "Barred Spiral":
        "A spiral galaxy featuring a prominent central bar structure.",

    "Unbarred Spiral":
        "A spiral galaxy with defined spiral arms but no prominent "
        "central stellar bar.",

    "Edge-on Galaxy":
        "A disk galaxy viewed nearly edge-on, producing a thin "
        "elongated profile.",
}

# ============================================================
# 4. Model Loader
# ============================================================

@st.cache_resource
def load_classifier_model(model_path):
    return tf.keras.models.load_model(model_path)


# ============================================================
# 5. Image Preprocessing
# ============================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert image to RGB, pad to a square, resize to 224x224,
    and apply EfficientNet preprocessing.
    """

    img = image.convert("RGB")

    max_dim = max(img.size)

    padded_img = Image.new(
        "RGB",
        (max_dim, max_dim),
        (0, 0, 0),
    )

    padded_img.paste(
        img,
        (
            (max_dim - img.size[0]) // 2,
            (max_dim - img.size[1]) // 2,
        ),
    )

    resized_img = padded_img.resize((224, 224))

    img_array = np.asarray(
        resized_img,
        dtype=np.float32,
    )

    img_array = preprocess_input(img_array)

    return np.expand_dims(img_array, axis=0)


# ============================================================
# 6. Prediction Probability Handling
# ============================================================

def get_probabilities(raw_predictions):

    raw_predictions = np.asarray(
        raw_predictions,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(raw_predictions)):
        raise ValueError("Model returned invalid prediction values.")

    # Convert logits to probabilities if necessary.
    if (
        raw_predictions.min() < 0
        or raw_predictions.max() > 1
        or not np.isclose(raw_predictions.sum(), 1.0, atol=1e-3)
    ):

        shifted = raw_predictions - np.max(raw_predictions)
        exp_values = np.exp(shifted)

        probabilities = exp_values / exp_values.sum()

    else:
        probabilities = raw_predictions

    return probabilities


# ============================================================
# 7. Header
# ============================================================

st.title("🔭 Interactive Galaxy Morphology Classifier")

st.markdown("""
Explore galaxy morphology using a deep learning classifier trained
on the **Galaxy10 DECaLS dataset**.

The project includes an original 10-class classification experiment
and a subsequently developed six-class model for interactive
image classification.
""")

st.divider()


# ============================================================
# 8. Original 10-Class Experiment
# ============================================================

with st.expander(
    "📊 Original Galaxy10 DECaLS Experiment",
    expanded=True,
):

    st.markdown("""
    ### Dataset and classification

    The original experiment used the **Galaxy10 DECaLS dataset**,
    containing 17,736 astronomical images across 10 morphological
    categories.

    The 10-class experiment established the initial classification
    pipeline and evaluation baseline.
    """)

    metric_cols = st.columns(4)

    metric_cols[0].metric(
        "Test Accuracy",
        "71.98%",
    )

    metric_cols[1].metric(
        "Macro Precision",
        "0.70",
    )

    metric_cols[2].metric(
        "Macro Recall",
        "0.70",
    )

    metric_cols[3].metric(
        "Test Images",
        "3,548",
    )

    st.caption(
        "These evaluation metrics belong to the original "
        "10-class experiment, not the six-class model currently "
        "used for predictions."
    )

    st.markdown("""
    **Dataset:** Galaxy10 DECaLS

    **Original categories:** 10 morphological classes

    **Evaluation:** Classification accuracy, macro precision,
    and macro recall on the original test set.
    """)


# ============================================================
# 9. Six-Class Model Details
# ============================================================

with st.expander("🧠 Six-Class Model — Training and Validation"):

    st.markdown("""
    The original Galaxy10 categories were regrouped into six
    broader morphological classes for the final interactive model.

    **Model architecture**

    - ImageNet-pretrained EfficientNetB0 backbone
    - Global Average Pooling
    - Dense layer with 256 ReLU units
    - Dropout (0.4)
    - Six-unit softmax output layer

    **Training strategy**

    - Phase 1: Frozen-backbone training
    - Phase 2: Fine-tuning the upper backbone layers
    - Phase 3: Augmented fine-tuning

    **Final Phase 3 validation results**
    """)

    validation_cols = st.columns(3)

    validation_cols[0].metric(
        "Validation Accuracy",
        "78.69%",
    )

    validation_cols[1].metric(
        "Validation Loss",
        "0.6096",
    )

    validation_cols[2].metric(
        "Training Epochs",
        "47",
    )

    st.caption(
        "These are six-class validation results. They should not "
        "be interpreted as the 10-class test-set metrics above."
    )


# ============================================================
# 10. Classification Notes
# ============================================================

with st.expander("📌 Classification Notes and Limitations"):

    st.markdown("""
    - **Distinct morphologies:** Edge-on galaxies, barred spirals,
      and merging galaxies may exhibit identifiable structural
      features.

    - **Morphological overlap:** Elliptical and disturbed galaxies
      can share visual characteristics. Barred and unbarred spirals
      may also be difficult to distinguish when the bar is weak
      or obscured.

    - **Six-class classification:** The deployed model predicts
      one of six broader morphological categories derived from
      the original Galaxy10 classification scheme.

    - **Prediction confidence:** The displayed confidence is the
      model's output probability for a predicted class. It is not
      the model's measured classification accuracy.
    """)


st.divider()


# ============================================================
# 11. Load Six-Class Model
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        f"Model file not found: {MODEL_PATH}"
    )

    st.stop()

try:

    model = load_classifier_model(str(MODEL_PATH))

except Exception as error:

    st.error("Unable to load the six-class classification model.")
    st.exception(error)
    st.stop()


if model.output_shape[-1] != len(CLASS_NAMES):

    st.error(
        "Model output does not match the six-class configuration."
    )

    st.stop()


st.sidebar.header("⚙️ Model Information")

st.sidebar.success("Six-class Phase 3 model loaded")

st.sidebar.markdown("""
**Architecture:** EfficientNetB0

**Input resolution:** 224 × 224

**Output classes:** 6

**Training stage:** Phase 3
""")


# ============================================================
# 12. Session State
# ============================================================

if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None

if "image_source_name" not in st.session_state:
    st.session_state["image_source_name"] = ""


# ============================================================
# 13. Two-Column Interface
# ============================================================

col_input, col_display = st.columns(
    [1, 1],
    gap="large",
)


# ============================================================
# 14. Image Selection
# ============================================================

with col_input:

    st.subheader("1. Select or Upload Image")

    upload_tab, sample_tab = st.tabs(
        ["📤 Upload Image", "🖼️ Try Sample Image"]
    )

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    with upload_tab:

        uploaded_file = st.file_uploader(
            "Choose a galaxy image",
            type=["jpg", "jpeg", "png"],
            help="Upload an optical astronomical image.",
        )

        if uploaded_file is not None:

            try:

                uploaded_image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.session_state["selected_image"] = uploaded_image

                st.session_state["image_source_name"] = (
                    uploaded_file.name
                )

            except Exception:

                st.error("Unable to read the uploaded image.")

    # --------------------------------------------------------
    # Sample Images
    # --------------------------------------------------------

    with sample_tab:

        st.markdown("Test the classifier using sample images:")

        sample_images = {
            "Barred Spiral": "sample_barred",
            "Merging": "sample_merging",
            "Edge-on": "sample_edgeon",
            "Unbarred Spiral": "sample_unbarred_spiral",
            "Disturbed": "sample_distorted",
            "Elliptical": "sample_elliptical",
        }

        sample_cols = st.columns(3)

        for idx, (label, filename) in enumerate(
            sample_images.items()
        ):

            col = sample_cols[idx % 3]

            valid_path = None

            for ext in [
                ".jpg", ".jpeg", ".png",
                ".JPG", ".JPEG", ".PNG",
            ]:

                candidate = SAMPLES_DIR / f"{filename}{ext}"

                if candidate.exists():

                    valid_path = candidate
                    break

            if valid_path is not None:

                if col.button(
                    f"Load {label}",
                    key=f"sample_{idx}",
                    use_container_width=True,
                ):

                    st.session_state["selected_image"] = (
                        Image.open(valid_path).convert("RGB")
                    )

                    st.session_state["image_source_name"] = (
                        f"Sample ({label})"
                    )

                    st.rerun()

            else:

                col.caption(f"{label} (missing)")


# ============================================================
# 15. Preview and Classification
# ============================================================

with col_display:

    st.subheader("2. Preview and Classification")

    selected_image = st.session_state["selected_image"]

    if selected_image is not None:

        st.image(
            selected_image,
            caption=(
                f"Input Image: "
                f"{st.session_state['image_source_name']}"
            ),
            use_container_width=True,
        )

        if st.button(
            "🚀 Classify Galaxy Morphology",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Processing image through the neural network..."
            ):

                try:

                    input_tensor = preprocess_image(
                        selected_image
                    )

                    raw_predictions = model.predict(
                        input_tensor,
                        verbose=0,
                    )[0]

                    if len(raw_predictions) != 6:

                        st.error(
                            "Expected six model outputs, but received "
                            f"{len(raw_predictions)}."
                        )

                        st.stop()

                    probabilities = get_probabilities(
                        raw_predictions
                    )

                    # ----------------------------------------
                    # Top Prediction
                    # ----------------------------------------

                    top_idx = int(
                        np.argmax(probabilities)
                    )

                    top_class = CLASS_NAMES[top_idx]

                    top_confidence = (
                        probabilities[top_idx] * 100
                    )

                    st.divider()

                    # ----------------------------------------
                    # Prediction Metrics
                    # ----------------------------------------

                    metric_col1, metric_col2 = st.columns(2)

                    metric_col1.metric(
                        "Predicted Class",
                        top_class,
                    )

                    metric_col2.metric(
                        "Confidence Score",
                        f"{top_confidence:.2f}%",
                    )

                    # ----------------------------------------
                    # Description
                    # ----------------------------------------

                    st.info(
                        f"💡 **Description:** "
                        f"{CLASS_DESCRIPTIONS[top_class]}"
                    )

                    # ----------------------------------------
                    # Probability Distribution
                    # ----------------------------------------

                    st.markdown(
                        "#### Probability Distribution Across Six Classes"
                    )

                    df_probs = pd.DataFrame({
                        "Morphological Class": CLASS_NAMES,
                        "Confidence (%)": probabilities * 100,
                    }).sort_values(
                        "Confidence (%)",
                        ascending=True,
                    )

                    fig_probs = px.bar(
                        df_probs,
                        x="Confidence (%)",
                        y="Morphological Class",
                        orientation="h",
                        text_auto=".2f",
                        color="Confidence (%)",
                        color_continuous_scale="Blues",
                    )

                    fig_probs.update_layout(
                        height=380,
                        margin=dict(
                            l=10,
                            r=10,
                            t=10,
                            b=10,
                        ),
                        showlegend=False,
                        xaxis_title="Confidence (%)",
                        yaxis_title="",
                    )

                    st.plotly_chart(
                        fig_probs,
                        use_container_width=True,
                    )

                except Exception as error:

                    st.error("Classification failed.")
                    st.exception(error)

    else:

        st.info(
            "👈 Upload an image or select a sample galaxy "
            "to begin classification."
        )
