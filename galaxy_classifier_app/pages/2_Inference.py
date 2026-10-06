import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Galaxy Classifier - Real-Time Inference",
    page_icon="🔭",
    layout="wide",
)


# ============================================================
# 2. Project Paths
# ============================================================

# This file is assumed to be inside galaxy_classifier_app/pages/
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model-2" / "galaxy10_efficientnetb0_phase3.keras"
st.write("Model path:", MODEL_PATH)
st.write("Model exists:", MODEL_PATH.exists())
SAMPLES_DIR = BASE_DIR / "assets" / "samples"


# ============================================================
# 3. Six-Class Morphological Classification
# ============================================================

CLASS_NAMES = [
    "Distorted Galaxy",
    "Merging Galaxies",
    "Elliptical Galaxy",
    "Barred Spiral",
    "Unbarred Spiral",
    "Edge-on Galaxy",
]


CLASS_DESCRIPTIONS = {
    "Distorted Galaxy":
        "Shows a non-symmetric shape, tidal tails, or irregular structural disruption.",

    "Merging Galaxies":
        "Two or more gravitationally interacting galaxy cores in the process of merging.",

    "Elliptical Galaxy":
        "A smooth, featureless galaxy with an elliptical or rounded light profile.",

    "Barred Spiral":
        "Spiral galaxy featuring a prominent central linear bar structure of stars.",

    "Unbarred Spiral":
        "Spiral galaxy with clearly defined spiral arms but without a central stellar bar.",

    "Edge-on Galaxy":
        "A disk galaxy viewed nearly edge-on, producing a thin elongated profile.",
}


# ============================================================
# 4. Cached Model Loader
# ============================================================

@st.cache_resource
def load_classifier_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Model exists, but could not be loaded: {e}")
        return None


# ============================================================
# 5. Image Preprocessing
# ============================================================

def preprocess_image(
    image: Image.Image,
    target_size=(224, 224)
) -> np.ndarray:

    """Pads image to square aspect ratio, resizes, and preprocesses it."""

    img = image.convert("RGB")

    # Pad to square ratio to avoid distorting galaxy morphology
    max_dim = max(img.size)

    padded_img = Image.new(
        "RGB",
        (max_dim, max_dim),
        (0, 0, 0)
    )

    padded_img.paste(
        img,
        (
            (max_dim - img.size[0]) // 2,
            (max_dim - img.size[1]) // 2
        )
    )

    resized_img = padded_img.resize(target_size)

    from tensorflow.keras.applications.efficientnet import preprocess_input

    img_array = np.array(
        resized_img,
        dtype=np.float32
    )

    img_array = preprocess_input(img_array)

    return np.expand_dims(img_array, axis=0)


# ============================================================
# 6. Header Section
# ============================================================

st.title("🔭 Interactive Galaxy Morphology Classifier")

st.markdown("""
Upload custom optical astronomical images (JPG, JPEG, or PNG) to classify their
morphological type according to the **Hubble Sequence** using your trained
six-class deep learning model.
""")

st.divider()


# ============================================================
# 7. Sidebar Controls
# ============================================================

st.sidebar.header("⚙️ Model Settings")

IMAGE_SIZE = st.sidebar.selectbox(
    "Target Image Resolution",
    options=[
        (224, 224),
        (256, 256),
        (69, 69)
    ],
    format_func=lambda x:
        f"{x[0]} x {x[1]} "
        f"{'(Default)' if x == (224, 224) else ''}",
    index=0,
    help="Select input resolution expected by your neural network",
)


# ============================================================
# 8. Load Model
# ============================================================

model = load_classifier_model(str(MODEL_PATH))

if model is None:

    st.sidebar.warning(
        f"⚠️ Model file not found at `{MODEL_PATH}`. "
        "Using mock preview mode."
    )

else:

    st.sidebar.success(
        "✅ Six-class model loaded successfully!"
    )


# ============================================================
# 9. Classification Notes
# ============================================================

with st.expander("📌 Note: Classification Performance & Edge Cases"):

    st.markdown("""
    **Model Performance Observations:**

    - **Distinct Morphologies:** Edge-on galaxies, barred spirals,
      and merging galaxies contain strong structural features that
      can help the model distinguish them.

    - **Morphological Overlap:** Elliptical and distorted galaxies
      can show overlapping visual characteristics, while barred and
      unbarred spirals may be difficult to distinguish when the bar
      structure is weak or obscured.

    - **Six-Class Classification:** The original Galaxy10 DECaLS
      categories are consolidated into six broader morphological
      classes for the final classifier.
    """)


# ============================================================
# 10. Session State
# ============================================================

if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None

if "image_source_name" not in st.session_state:
    st.session_state["image_source_name"] = ""


# ============================================================
# 11. Dual Column Interface
# ============================================================

col_input, col_display = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# 12. Image Selection
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
            "Choose a galaxy image file",
            type=["jpg", "jpeg", "png"],
            help="Supports standard optical astronomical images (RGB)",
        )

        if uploaded_file is not None:

            st.session_state["selected_image"] = Image.open(
                uploaded_file
            )

            st.session_state["image_source_name"] = uploaded_file.name


    # --------------------------------------------------------
    # Sample Images
    # --------------------------------------------------------

    with sample_tab:

        st.markdown(
            "Test with sample astronomical images:"
        )

        sample_cols = st.columns(3)

        sample_images = {

            "Barred Spiral":
                "sample_barred",

            "Merging":
                "sample_merging",

            "Edge-on":
                "sample_edgeon",

            "Unbarred Spiral":
                "sample_unbarred_spiral",

            "Distorted":
                "sample_distorted",

            "Elliptical":
                "sample_elliptical",
        }


        for idx, (label, filename) in enumerate(
            sample_images.items()
        ):

            col = sample_cols[idx % 3]

            valid_path = None

            # Look for supported image extensions
            for ext in [
                ".jpeg",
                ".jpg",
                ".png",
                ".JPEG",
                ".JPG",
                ".PNG"
            ]:

                full_path = SAMPLES_DIR / f"{filename}{ext}"

                if full_path.exists():

                    valid_path = full_path
                    break


            if valid_path:

                if col.button(
                    f"Load {label}",
                    key=f"sample_{idx}",
                    use_container_width=True,
                ):

                    st.session_state["selected_image"] = Image.open(
                        valid_path
                    )

                    st.session_state["image_source_name"] = (
                        f"Sample ({label})"
                    )

            else:

                col.caption(
                    f"`{label}` (missing)"
                )


# ============================================================
# 13. Preview & Classification
# ============================================================

with col_display:

    st.subheader("2. Preview & Classification")


    if st.session_state["selected_image"] is not None:

        st.image(
            st.session_state["selected_image"],
            caption=(
                f"Input Image: "
                f"{st.session_state['image_source_name']}"
            ),
            use_container_width=True,
        )


        # ----------------------------------------------------
        # Classification Button
        # ----------------------------------------------------

        if st.button(
            "🚀 Classify Galaxy Morphology",
            type="primary"
        ):

            with st.spinner(
                "Processing image through neural network..."
            ):

                # --------------------------------------------
                # Preprocess
                # --------------------------------------------

                input_tensor = preprocess_image(
                    st.session_state["selected_image"],
                    target_size=IMAGE_SIZE
                )


                # --------------------------------------------
                # Model Prediction
                # --------------------------------------------

                if model is not None:

                    raw_preds = model.predict(
                        input_tensor,
                        verbose=0
                    )[0]

                    # The Phase 3 model is a six-class model.
                    if len(raw_preds) != 6:

                        st.error(
                            f"Expected a six-class model, "
                            f"but the loaded model returned "
                            f"{len(raw_preds)} outputs."
                        )

                        st.stop()


                    # If outputs are logits rather than probabilities
                    if (
                        raw_preds.max() > 1.0
                        or raw_preds.min() < 0.0
                    ):

                        exp_preds = np.exp(
                            raw_preds - np.max(raw_preds)
                        )

                        probs = (
                            exp_preds /
                            exp_preds.sum()
                        )

                    else:

                        probs = raw_preds


                # --------------------------------------------
                # Mock Preview
                # --------------------------------------------

                else:

                    np.random.seed(
                        len(
                            st.session_state[
                                "image_source_name"
                            ]
                        )
                    )

                    raw_mock = np.random.dirichlet(
                        np.ones(6)
                    )

                    probs = (
                        raw_mock /
                        raw_mock.sum()
                    )


                # --------------------------------------------
                # Top Prediction
                # --------------------------------------------

                top_idx = int(
                    np.argmax(probs)
                )

                top_class = CLASS_NAMES[top_idx]

                top_conf = (
                    probs[top_idx] * 100
                )


                st.markdown("---")


                # --------------------------------------------
                # Prediction Metrics
                # --------------------------------------------

                m_col1, m_col2 = st.columns(2)


                with m_col1:

                    st.metric(
                        label="Predicted Class",
                        value=top_class,
                    )


                with m_col2:

                    st.metric(
                        label="Confidence Score",
                        value=f"{top_conf:.2f}%",
                    )


                # --------------------------------------------
                # Description
                # --------------------------------------------

                st.info(
                    f"💡 **Description:** "
                    f"{CLASS_DESCRIPTIONS[top_class]}"
                )


                # =================================================
                # Six-Class Probability Distribution
                # =================================================

                st.markdown(
                    "#### Probability Distribution Across 6 Classes"
                )


                df_probs = pd.DataFrame(
                    {
                        "Morphological Class":
                            CLASS_NAMES,

                        "Confidence (%)":
                            probs * 100,
                    }
                ).sort_values(
                    "Confidence (%)",
                    ascending=True
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
                        b=10
                    ),
                    showlegend=False,
                    xaxis_title="Confidence Percentage (%)",
                    yaxis_title="",
                )


                st.plotly_chart(
                    fig_probs,
                    use_container_width=True
                )


    else:

        st.info(
            "👈 Upload an image or select a sample galaxy "
            "to begin analysis."
        )
