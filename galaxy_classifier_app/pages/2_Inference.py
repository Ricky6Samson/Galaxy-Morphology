import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Galaxy Classifier - Real-Time Inference",
    page_icon="🔭",
    layout="wide",
)


# ============================================================
# 2. ROBUST PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = BASE_DIR / "assets"
SAMPLES_DIR = BASE_DIR / "assets" / "samples"

# IMPORTANT:
# Your repository uses "model" (singular), not "models".
MODEL_PATH = BASE_DIR / "model" / "galaxy10_efficientnetb0_phase2_run2.keras"

GALAXY10_IMAGE = ASSETS_DIR / "galaxy10.png"
CONFUSION_MATRIX = ASSETS_DIR / "confusion_matrix.png"
TRAINING_HISTORY = ASSETS_DIR / "training_history.png"


# ============================================================
# 3. CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Distorted Galaxy",
    "Merging Galaxies",
    "Round Elliptical",
    "In-between Elliptical",
    "Cigar Elliptical",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on (No Bulge)",
    "Edge-on (With Bulge)",
]


# ============================================================
# 4. CLASS DESCRIPTIONS
# ============================================================

CLASS_DESCRIPTIONS = {
    "Distorted Galaxy":
        "Shows non-symmetric shape, tidal tails, or irregular structural disruption.",

    "Merging Galaxies":
        "Two or more gravitationally interacting galaxy cores in the process of merging.",

    "Round Elliptical":
        "Smooth, featureless elliptical galaxy with near-spherical symmetry.",

    "In-between Elliptical":
        "Slightly elongated, smooth elliptical galaxy profile.",

    "Cigar Elliptical":
        "Highly elongated, needle- or cigar-shaped smooth elliptical galaxy.",

    "Barred Spiral":
        "Spiral galaxy featuring a prominent central linear bar structure of stars.",

    "Unbarred Tight Spiral":
        "Spiral arms tightly wound around the galactic nucleus without a central bar.",

    "Unbarred Loose Spiral":
        "Open, well-defined loose spiral arm structure.",

    "Edge-on (No Bulge)":
        "Disk-dominated galaxy viewed edge-on with no visible central galactic bulge.",

    "Edge-on (With Bulge)":
        "Disk galaxy viewed edge-on featuring a prominent central nuclear bulge.",
}


# ============================================================
# 5. LOAD MODEL
# ============================================================

@st.cache_resource
def load_classifier_model(model_path):

    import tensorflow as tf

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found at:\n{model_path}"
        )

    return tf.keras.models.load_model(model_path)


try:

    model = load_classifier_model(str(MODEL_PATH))

    MODEL_LOADED = True

except Exception as e:

    model = None
    MODEL_LOADED = False

    MODEL_ERROR = str(e)


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image: Image.Image,
    target_size=(224, 224)
):

    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = image.convert("RGB")

    # Pad to square without distorting galaxy morphology
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

    # Resize
    resized_img = padded_img.resize(target_size)

    # Convert to numpy
    img_array = np.array(
        resized_img,
        dtype=np.float32,
    )

    # EfficientNet preprocessing
    img_array = preprocess_input(img_array)

    # Add batch dimension
    return np.expand_dims(
        img_array,
        axis=0,
    )


# ============================================================
# 7. HEADER
# ============================================================

st.title("🔭 Interactive Galaxy Morphology Classifier")

st.markdown(
    """
    Upload custom optical astronomical images (JPG, JPEG, or PNG) to classify their
    morphological type according to the **Hubble Sequence** using your trained
    EfficientNetB0 deep learning model.
    """
)

st.divider()


# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Model Settings")

IMAGE_SIZE = st.sidebar.selectbox(
    "Target Image Resolution",
    options=[
        (224, 224),
        (256, 256),
        (69, 69),
    ],
    format_func=lambda x:
        f"{x[0]} x {x[1]} "
        f"{'(Default)' if x == (224, 224) else ''}",
    index=0,
    help="Select input resolution expected by the neural network.",
)


# Model status
if MODEL_LOADED:

    st.sidebar.success(
        "✅ Model loaded successfully!"
    )

else:

    st.sidebar.error(
        "❌ Model failed to load."
    )

    with st.sidebar.expander("Show model error"):

        st.code(
            MODEL_ERROR,
            language="text",
        )

        st.caption(
            f"Expected model path:\n{MODEL_PATH}"
        )


# ============================================================
# 9. MODEL NOTES
# ============================================================

with st.expander(
    "📌 Note: Classification Performance & Edge Cases"
):

    st.markdown(
        """
        **Model Performance Observations:**

        - **High Accuracy Subsets:** The EfficientNetB0 backbone reliably isolates
          structural targets with distinct, high-contrast features such as
          *Edge-on* disks, *Barred Spirals*, and *Merging* cores.

        - **Fine-Grained Classification Challenges:** Morphological categories
          lying along a continuous visual spectrum—specifically **Round Elliptical**,
          **In-between Elliptical**, **Cigar Elliptical**, and **Distorted** profiles—
          exhibit higher inter-class ambiguity.

        - **Physiological Basis:** Ellipticals lack distinct sharp edges or dust lanes,
          making axial ratios highly sensitive to image cropping and viewing angles.
          Refer to the horizontal probability distribution chart when reviewing
          these edge cases to evaluate soft confidence scores.
        """
    )


# ============================================================
# 10. SESSION STATE
# ============================================================

if "selected_image" not in st.session_state:

    st.session_state["selected_image"] = None


if "image_source_name" not in st.session_state:

    st.session_state["image_source_name"] = ""


# ============================================================
# 11. MAIN TWO-COLUMN INTERFACE
# ============================================================

col_input, col_display = st.columns(
    [1, 1],
    gap="large",
)


# ============================================================
# 12. INPUT COLUMN
# ============================================================

with col_input:

    st.subheader("1. Select or Upload Image")

    upload_tab, sample_tab = st.tabs(
        [
            "📤 Upload Image",
            "🖼️ Try Sample Image",
        ]
    )


    # --------------------------------------------------------
    # Upload Image
    # --------------------------------------------------------

    with upload_tab:

        uploaded_file = st.file_uploader(
            "Choose a galaxy image file",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
            help="Supports standard optical astronomical images (RGB).",
        )

        if uploaded_file is not None:

            st.session_state["selected_image"] = (
                Image.open(uploaded_file)
                .convert("RGB")
            )

            st.session_state["image_source_name"] = (
                uploaded_file.name
            )


    # --------------------------------------------------------
    # Sample Images
    # --------------------------------------------------------

    with sample_tab:

        st.markdown(
            "Test with sample astronomical images:"
        )

        # Exact filename stems used by the project.
        sample_images = {
            "Barred Spiral":
                "sample_barred",

            "Merging":
                "sample_merging",

            "Edge-on (with Bulge)":
                "sample_edgeon(bulge)",

            "Edge-on (no Bulge)":
                "sample_edgeon(no bulge)",

            "Unbarred Loose Spiral":
                "sample_unbarred (loose spiral)",

            "Unbarred Tight Spiral":
                "sample_unbarred (tight spiral)",
        }


        sample_cols = st.columns(3)


        for idx, (label, base_name) in enumerate(
            sample_images.items()
        ):

            col = sample_cols[idx % 3]

            valid_path = None


            # Search for image regardless of extension
            # and filename capitalization.
            if SAMPLES_DIR.exists():

                for file_path in SAMPLES_DIR.iterdir():

                    if not file_path.is_file():
                        continue

                    if file_path.suffix.lower() not in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                    }:
                        continue

                    if (
                        file_path.stem.lower()
                        == base_name.lower()
                    ):

                        valid_path = file_path
                        break


            # ------------------------------------------------
            # Image found
            # ------------------------------------------------

            if valid_path is not None:

                col.image(
                    str(valid_path),
                    use_container_width=True,
                )

                if col.button(
                    f"Load {label}",
                    key=f"sample_{idx}",
                    use_container_width=True,
                ):

                    st.session_state[
                        "selected_image"
                    ] = (
                        Image.open(valid_path)
                        .convert("RGB")
                    )

                    st.session_state[
                        "image_source_name"
                    ] = f"Sample ({label})"


            # ------------------------------------------------
            # Image missing
            # ------------------------------------------------

            else:

                col.warning(
                    f"Image not found:\n{base_name}"
                )


# ============================================================
# 13. DISPLAY / CLASSIFICATION COLUMN
# ============================================================

with col_display:

    st.subheader("2. Preview & Classification")


    # --------------------------------------------------------
    # No Image Selected
    # --------------------------------------------------------

    if st.session_state["selected_image"] is None:

        st.info(
            "👈 Upload an image or select a sample galaxy "
            "to begin analysis."
        )


    # --------------------------------------------------------
    # Image Selected
    # --------------------------------------------------------

    else:

        st.image(
            st.session_state["selected_image"],
            caption=(
                "Input Image: "
                + st.session_state["image_source_name"]
            ),
            use_container_width=True,
        )


        # ====================================================
        # CLASSIFICATION BUTTON
        # ====================================================

        if st.button(
            "🚀 Classify Galaxy Morphology",
            type="primary",
            use_container_width=True,
        ):

            if not MODEL_LOADED:

                st.error(
                    "The trained model could not be loaded. "
                    "Check the model path and deployment logs."
                )

            else:

                with st.spinner(
                    "Processing image through neural network..."
                ):

                    try:

                        # ------------------------------------
                        # Preprocess
                        # ------------------------------------

                        input_tensor = preprocess_image(
                            st.session_state[
                                "selected_image"
                            ],
                            target_size=IMAGE_SIZE,
                        )


                        # ------------------------------------
                        # Prediction
                        # ------------------------------------

                        raw_preds = model.predict(
                            input_tensor,
                            verbose=0,
                        )[0]


                        # ------------------------------------
                        # Convert predictions to probabilities
                        # ------------------------------------

                        # Normally the model already outputs
                        # softmax probabilities.

                        if (
                            raw_preds.max() > 1.0
                            or raw_preds.min() < 0.0
                        ):

                            exp_preds = np.exp(
                                raw_preds
                                - np.max(raw_preds)
                            )

                            probs = (
                                exp_preds
                                / exp_preds.sum()
                            )

                        else:

                            probs = raw_preds


                        # ------------------------------------
                        # Top prediction
                        # ------------------------------------

                        top_idx = int(
                            np.argmax(probs)
                        )

                        top_class = CLASS_NAMES[
                            top_idx
                        ]

                        top_conf = (
                            probs[top_idx] * 100
                        )


                        st.markdown("---")


                        # ------------------------------------
                        # Metrics
                        # ------------------------------------

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


                        # ------------------------------------
                        # Description
                        # ------------------------------------

                        st.info(
                            "💡 **Description:** "
                            + CLASS_DESCRIPTIONS[
                                top_class
                            ]
                        )


                        # ====================================================
                        # PROBABILITY DISTRIBUTION
                        # ====================================================

                        st.markdown(
                            "#### Probability Distribution Across All Classes"
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
                            xaxis_title=(
                                "Confidence Percentage (%)"
                            ),
                            yaxis_title="",
                        )


                        st.plotly_chart(
                            fig_probs,
                            use_container_width=True,
                        )


                    except Exception as e:

                        st.error(
                            "Prediction failed."
                        )

                        with st.expander(
                            "Show technical error"
                        ):

                            st.exception(e)
