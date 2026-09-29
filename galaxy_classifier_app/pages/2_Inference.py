import os
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Galaxy Classifier - Real-Time Inference",
    page_icon="🔭",
    layout="wide",
)

# 2. Hardcoded Relative Model Path
MODEL_PATH = "models/galaxy10_efficientnetb0_phase2_run2.keras"

# 3. Class Names Mapping & Descriptions (Galaxy10 DECaLS)
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

CLASS_DESCRIPTIONS = {
    "Distorted Galaxy": "Shows non-symmetric shape, tidal tails, or irregular structural disruption.",
    "Merging Galaxies": "Two or more gravitationally interacting galaxy cores in the process of merging.",
    "Round Elliptical": "Smooth, featureless elliptical galaxy with near-spherical symmetry.",
    "In-between Elliptical": "Slightly elongated, smooth elliptical galaxy profile.",
    "Cigar Elliptical": "Highly elongated, needle- or cigar-shaped smooth elliptical galaxy.",
    "Barred Spiral": "Spiral galaxy featuring a prominent central linear bar structure of stars.",
    "Unbarred Tight Spiral": "Spiral arms tightly wound around the galactic nucleus without a central bar.",
    "Unbarred Loose Spiral": "Open, well-defined loose spiral arm structure.",
    "Edge-on (No Bulge)": "Disk-dominated galaxy viewed edge-on with no visible central galactic bulge.",
    "Edge-on (With Bulge)": "Disk galaxy viewed edge-on featuring a prominent central nuclear bulge.",
}


# 4. Cached Model Loader
@st.cache_resource
def load_classifier_model(path: str):
    """Loads and caches model to prevent reloading on user interaction."""
    if not os.path.exists(path):
        return None
    import tensorflow as tf

    return tf.keras.models.load_model(path)


# 5. Image Preprocessing
def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """Pads image to square aspect ratio, resizes, and processes for EfficientNet."""
    img = image.convert("RGB")

    # Pad to square ratio to avoid squashing shapes
    max_dim = max(img.size)
    padded_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    padded_img.paste(
        img, ((max_dim - img.size[0]) // 2, (max_dim - img.size[1]) // 2)
    )

    resized_img = padded_img.resize(target_size)
    
    # EfficientNet expects raw [0, 255] float arrays, OR use tf.keras.applications.efficientnet.preprocess_input
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img_array = np.array(resized_img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    
    return np.expand_dims(img_array, axis=0)

# 6. Header Section
st.title("🔭 Interactive Galaxy Morphology Classifier")
st.markdown("""
Upload custom optical astronomical images (JPG, JPEG, or PNG) to classify their morphological 
type according to the **Hubble Sequence** using your trained deep learning model.
""")

st.divider()

# 7. Sidebar Controls
st.sidebar.header("⚙️ Model Settings")

# Formatted option labels for professional portfolio presentation
IMAGE_SIZE = st.sidebar.selectbox(
    "Target Image Resolution",
    options=[(224, 224), (256, 256), (69, 69)],
    format_func=lambda x: f"{x[0]} x {x[1]} {'(Default)' if x == (224, 224) else ''}",
    index=0,
    help="Select input resolution expected by your neural network",
)

# Load cached model
model = load_classifier_model(MODEL_PATH)

if model is None:
    st.sidebar.warning(
        f"⚠️ Model file not found at `{MODEL_PATH}`. Using mock preview mode."
    )
else:
    st.sidebar.success("✅ Model loaded successfully!")

with st.expander("📌Note: Classification Performance & Edge Cases"):
    st.markdown("""
    **Model Performance Observations:**
    - **High Accuracy Subsets:** The EfficientNetB0 backbone reliably isolates structural targets with distinct, high-contrast features (such as *Edge-on* disks, *Barred Spirals*, and *Merging* cores).
    - **Fine-Grained Classification Challenges:** Morphological categories lying along a continuous visual spectrum—specifically **Round Elliptical**, **In-between Elliptical**, **Cigar Elliptical**, and **Distorted** profiles—exhibit higher inter-class ambiguity. 
    - **Physiological Basis:** Ellipticals lack distinct sharp edges or dust lanes, making axial ratios highly sensitive to image cropping and viewing angles. Refer to the horizontal probability distribution chart when reviewing these edge cases to evaluate soft confidence scores.
    """)

# Initialize session state for image tracking at the top
if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None
if "image_source_name" not in st.session_state:
    st.session_state["image_source_name"] = ""


# 8. Dual Column Interface Layout
col_input, col_display = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("1. Select or Upload Image")

    upload_tab, sample_tab = st.tabs(["📤 Upload Image", "🖼️ Try Sample Image"])

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose a galaxy image file",
            type=["jpg", "jpeg", "png"],
            help="Supports standard optical astronomical images (RGB)",
        )
        if uploaded_file is not None:
            st.session_state["selected_image"] = Image.open(uploaded_file)
            st.session_state["image_source_name"] = uploaded_file.name

    with sample_tab:
        st.markdown("Test with sample astronomical images:")
        sample_cols = st.columns(3)

        sample_images = {
            "Barred Spiral": "assets/samples/sample_barred",
            "Merging": "assets/samples/sample_merging",
            "Edge-on(with Bulge)": "assets/samples/sample_edgeon(bulge)",
            "Edge-on(no Bulge)":"assets/samples/sample_edgeon(no bulge)",
            "Unbarred Loose Spiral":"assets/samples/sample_unbarred (loose spiral)",
            "Unbarred Tight Spiral":"assets/samples/sample_unbarred (tight spiral)"
        }

        for idx, (label, base_path) in enumerate(sample_images.items()):
            col = sample_cols[idx % 3]

            # Dynamic extension lookup
            valid_path = None
            for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]:
                full_path = f"{base_path}{ext}"
                if os.path.exists(full_path):
                    valid_path = full_path
                    break

            if valid_path:
                if col.button(
                    f"Load {label}",
                    key=f"sample_{idx}", 
                    use_container_width=True,
                ):
                    # Persist to session state on click
                    st.session_state["selected_image"] = Image.open(valid_path)
                    st.session_state["image_source_name"] = f"Sample ({label})"
            else:
                col.caption(f"`{label}` (missing)")

with col_display:
    st.subheader("2. Preview & Classification")

    # Read image from persistent session state
    if st.session_state["selected_image"] is not None:
        st.image(
            st.session_state["selected_image"],
            caption=f"Input Image: {st.session_state['image_source_name']}",
            use_container_width=True,
        )

        if st.button("🚀 Classify Galaxy Morphology", type="primary"):
            with st.spinner("Processing image through neural network..."):

                # Preprocess tensor using state image
                input_tensor = preprocess_image(
                    st.session_state["selected_image"], target_size=IMAGE_SIZE
                )

                # Execute Model Prediction
                if model is not None:
                    raw_preds = model.predict(input_tensor)[0]
                    if raw_preds.max() > 1.0 or raw_preds.min() < 0.0:
                        exp_preds = np.exp(raw_preds - np.max(raw_preds))
                        probs = exp_preds / exp_preds.sum()
                    else:
                        probs = raw_preds
                else:
                    np.random.seed(len(st.session_state["image_source_name"]))
                    raw_mock = np.random.dirichlet(np.ones(10))
                    probs = raw_mock / raw_mock.sum()

                top_idx = int(np.argmax(probs))
                top_class = CLASS_NAMES[top_idx]
                top_conf = probs[top_idx] * 100

                st.markdown("---")

                # Clean dual metric display
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

                st.info(
                    f"💡 **Description:** {CLASS_DESCRIPTIONS[top_class]}"
                )

                # Confidence Plot
                st.markdown(
                    "#### Probability Distribution Across All Classes"
                )

                df_probs = pd.DataFrame(
                    {
                        "Morphological Class": CLASS_NAMES,
                        "Confidence (%)": probs * 100,
                    }
                ).sort_values("Confidence (%)", ascending=True)

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
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    xaxis_title="Confidence Percentage (%)",
                    yaxis_title="",
                )

                st.plotly_chart(fig_probs, use_container_width=True)

    else:
        st.info(
            "👈 Upload an image or select a sample galaxy to begin analysis."
        )