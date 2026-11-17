
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# 1. Paths and Page Configuration
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

st.set_page_config(
    page_title="Galaxy Morphology Classifier",
    page_icon="🌌",
    layout="wide"
)

st.markdown(
    """
    <style>
    div[data-testid="stColumn"] img {
        height: 380px !important;
        object-fit: contain !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 2. Main Title and Introduction
# ------------------------------------------------------------

st.title("🌌 Automated Galaxy Morphology Classifier")

st.markdown("""
Welcome to the Galaxy Morphology Analysis Platform.

This application uses transfer learning with EfficientNetB0
to classify astronomical images into six broad morphological
categories. The six-class model is an updated version of the
original ten-class Galaxy10 DECaLS classification experiment.
""")

st.divider()

# ------------------------------------------------------------
# 3. High-Level Model Metrics
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Validation Accuracy",
        value="78.69%"
    )

with col2:
    st.metric(
        label="Validation Loss",
        value="0.6096"
    )

with col3:
    st.metric(
        label="Morphological Classes",
        value="6"
    )

with col4:
    st.metric(
        label="Training Epochs",
        value="47"
    )

st.caption(
    "Metrics shown are from the final six-class Phase 3 model "
    "on the validation set."
)

st.divider()

# ------------------------------------------------------------
# 4. Original Galaxy10 Dataset Reference
# ------------------------------------------------------------

st.subheader("🌌 Original Galaxy10 DECaLS Dataset")

IMAGE_PATH = ASSETS_DIR / "galaxy10.png"

if IMAGE_PATH.exists():
    st.image(
        IMAGE_PATH,
        caption="Original Galaxy10 DECaLS Dataset: 10 Morphological Classes",
        use_container_width=True,
    )
else:
    st.info(
        "Place `galaxy10.png` in your `assets/` folder "
        "to display the original dataset classes."
    )

st.markdown("""
The original Galaxy10 DECaLS dataset contains 17,736 galaxy images
across ten morphological categories. For the final application,
related categories were consolidated into six broader classes.
""")

st.divider()

# ------------------------------------------------------------
# 5. Conversion from 10 Classes to 6 Classes
# ------------------------------------------------------------

st.subheader("🔄 From 10 Galaxy10 Classes to 6 Morphological Classes")

st.markdown("""
The original ten categories were grouped according to their
shared morphological characteristics. This produced a
six-class classification framework while retaining the
major structural distinctions between galaxies.
""")

class_mapping = pd.DataFrame({
    "Original Galaxy10 Classes": [
        "Disturbed",
        "Merging",
        "Round Smooth",
        "In-between Round Smooth",
        "Cigar Smooth",
        "Barred Spiral",
        "Unbarred Tight Spiral",
        "Unbarred Loose Spiral",
        "Edge-on No Bulge",
        "Edge-on With Bulge",
    ],
    "Final Six-Class Category": [
        "Disturbed",
        "Merging",
        "Elliptical",
        "Elliptical",
        "Elliptical",
        "Barred Spiral",
        "Unbarred Spiral",
        "Unbarred Spiral",
        "Edge-on",
        "Edge-on",
    ],
})

st.dataframe(
    class_mapping,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "The original ten categories are consolidated into six "
    "broader morphological classes."
)

st.divider()

# ------------------------------------------------------------
# 6. Six-Class Dataset Distribution
# ------------------------------------------------------------

st.subheader("📊 Six-Class Dataset Distribution")

distribution_data = {
    "Class": [
        "Disturbed",
        "Merging",
        "Elliptical",
        "Barred Spiral",
        "Unbarred Spiral",
        "Edge-on",
    ],
    "Samples": [
        1081,
        1853,
        5006,
        2043,
        4457,
        3296,
    ],
}

df_dist = pd.DataFrame(distribution_data)

fig = px.bar(
    df_dist,
    x="Class",
    y="Samples",
    color="Samples",
    color_continuous_scale="Viridis",
    labels={
        "Samples": "Image Count",
        "Class": "Morphological Class",
    },
    text="Samples",
)

fig.update_traces(
    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    textposition="outside",
)

fig.update_layout(
    xaxis_tickangle=-20,
    showlegend=False,
    height=450,
    margin=dict(l=20, r=20, t=20, b=70),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "The six-class dataset contains 17,736 images. "
    "Elliptical and Unbarred Spiral galaxies form the largest categories."
)

st.divider()

# ------------------------------------------------------------
# 7. Model Architecture
# ------------------------------------------------------------

st.subheader("🧠 Six-Class Model Architecture")

st.markdown("""
The final classifier uses **EfficientNetB0**, pretrained on ImageNet,
as its feature extraction backbone.

| Component | Configuration |
|---|---|
| Backbone | EfficientNetB0 |
| Pretraining | ImageNet |
| Input resolution | 224 × 224 × 3 |
| Feature extraction | Global Average Pooling |
| Fully connected layer | Dense (256 units, ReLU) |
| Regularisation | Dropout (0.4) |
| Output layer | Softmax (6 classes) |
""")

st.divider()

# ------------------------------------------------------------
# 8. Model Diagnostics and Training Progression
# ------------------------------------------------------------

st.subheader("🔬 Model Diagnostics & Progression")

diag_col1, diag_col2 = st.columns(2)

with diag_col1:
    st.markdown("#### Six-Class Confusion Matrix")

    cm_path = ASSETS_DIR / "cm_phase3.png"

    if cm_path.exists():
        st.image(
            cm_path,
            caption="Confusion Matrix: Six-Class Phase 3 Model",
            use_container_width=True,
        )
    else:
        st.info(
            "Add your six-class Phase 3 confusion matrix as "
            "`cm_phase3_6class.png` in the `assets/` folder."
        )

with diag_col2:
    st.markdown("#### Training & Validation History")

    history_path = ASSETS_DIR / "phase3_augmented_performance.png"

    if history_path.exists():
        st.image(
            history_path,
            caption="Six-Class Model: Training and Validation Accuracy and Loss",
            use_container_width=True,
        )
    else:
        st.info(
            "Add your six-class training history as "
            "`phase3_augmented_performance_6class.png` "
            "in the `assets/` folder."
        )

st.markdown("""
**Training observations:**

- **Progressive fine-tuning:** The model was trained through
  successive stages, beginning with a frozen feature extractor
  and gradually fine-tuning the backbone.
- **Data augmentation:** Spatial augmentation was introduced
  during the final training stage to improve generalisation.
- **Final performance:** The Phase 3 model achieved 78.69%
  validation accuracy with a validation loss of 0.6096.
""")

st.divider()

# ------------------------------------------------------------
# 9. Model Iteration Comparison
# ------------------------------------------------------------

st.subheader("📈 Model Iteration Comparison")

runs_data = {
    "Training Stage": [
        "Phase 1",
        "Phase 2 - Run 1",
        "Phase 2 - Run 2",
        "Phase 3 - Augmented",
    ],
    "Training Configuration": [
        "Frozen EfficientNetB0 backbone",
        "Top 40 layers fine-tuned",
        "Continued fine-tuning",
        "Augmented fine-tuning",
    ],
    "Validation Accuracy": [
        "72.89%",
        "77.54%",
        "78.49%",
        "78.69%",
    ],
    "Validation Loss": [
        "0.7588",
        "0.6544",
        "0.6152",
        "0.6096",
    ],
}

df_runs = pd.DataFrame(runs_data)

st.dataframe(
    df_runs,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Validation performance across the successive training stages "
    "of the final six-class model."
)
