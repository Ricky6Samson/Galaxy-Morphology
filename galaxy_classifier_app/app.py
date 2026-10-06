import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# Absolute path to the folder containing this Python file
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

# 1. Page Configuration
st.set_page_config(
    page_title="Galaxy Morphology Classifier", page_icon="🌌", layout="wide"
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

# 2. Main Title and Subtitle
st.title("🌌 Automated Galaxy Morphology Classifier")
st.markdown("""
Welcome to the Galaxy Morphology Analysis Platform. This application leverages deep learning 
to classify astronomical imagery into standard Hubble sequence morphological types.
""")

st.divider()

# 3. High-Level Performance Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Model Accuracy", value="71.98%", delta="+2.39%")
with col2:
    st.metric(label="Precision (Macro)", value="0.70")
with col3:
    st.metric(label="Recall (Macro)", value="0.70")
with col4:
    st.metric(label="Total Test Images", value="3548")

st.divider()

# 4. Galaxy Classes Reference Image
st.subheader("🌌 Galaxy10 DECals Morphological Classes")
IMAGE_PATH = ASSETS_DIR / "galaxy10.png"

if os.path.exists(IMAGE_PATH):
    st.image(
        IMAGE_PATH,
        caption="Galaxy10 DECals Dataset: 10 Morphological Classes (0–9)",
        use_container_width=True,
    )
else:
    st.info("Place `galaxy10.png` in your `assets/` folder to render reference classes.")

st.divider()

# 5. Conversion from 10 Classes to 6 Classes
st.subheader("🔄 From 10 Galaxy10 Classes to 6 Morphological Classes")

st.markdown("""
The original Galaxy10 DECaLS dataset contains 10 morphological classes. 
For the application, related morphological categories are grouped into six broader classes 
to provide a more interpretable classification scheme.
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
    "6-Class Category": [
        "Distorted",
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
    "Related Galaxy10 categories are merged based on their shared morphological characteristics."
)

st.divider()

# 6. Dataset Class Distribution
st.subheader("📊 Six-Class Dataset Distribution")

distribution_data = {
    "Class": [
        "Distorted",
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
    "The six-class grouping reduces the original class imbalance by combining "
    "morphologically related categories, while retaining the major structural distinctions."
)

st.divider()

# 7. Model Diagnostics
st.subheader("🔬 Model Diagnostics & Progression")

diag_col1, diag_col2 = st.columns(2)

with diag_col1:
    st.markdown("#### Confusion Matrix (Phase 2, Run 2)")
    cm_path = ASSETS_DIR / "cm_phase3.png"
    if os.path.exists(cm_path):
        st.image(
            cm_path,
            caption="Confusion Matrix across 10 Morphological Classes",
            use_container_width=True,
        )
    else:
        st.info(
            "Place `cm_phase3.png` in your `assets/` folder to render the heatmap."
        )

with diag_col2:
    st.markdown("#### Training & Validation History")
    history_path = ASSETS_DIR / "phase3_augmented_performance.png"
    if os.path.exists(history_path):
        st.image(
            history_path,
            caption="Accuracy and Loss progression across epochs",
            use_container_width=True,
        )
    else:
        st.info(
            "Place `phase3_augmented_performance.png` in your `assets/` folder to render the curves."
        )

st.markdown("""
**Key Diagnostic Insights:**
* **Highest Performing:** Edge-on galaxies (Classes 8 & 9) achieved the strongest classification accuracy (>88% F1-score) due to distinct linear profile features.
* **Primary Confusion:** Class 0 (Distorted) and Class 1 (Merging) showed significant overlap due to shared irregular, low-density tidal features.
""")

st.divider()

# 8. Model Iteration Table
st.subheader("📈 Model Iteration Comparison")

runs_data = {
    "Run / Phase": [
        "Phase 1",
        "Phase 2 - Run 1",
        "Phase 2 - Run 2 ",
        "Phase 3 - Augmented (Best)",
    ],
    "Architecture / Details": [
        "Baseline EfficientNetB0",
        "Top 40 layers unfrozen",
        "Top 40 layers unfrozen run 2",
        "Augmented Images",
    ],
    "Val Accuracy": ["72.89%", "77.54%", "78.49%", "78.69%"],
    "Macro F1-Score": ["0.6253", "0.6960", "0.7147", "0.7204"],
}

df_runs = pd.DataFrame(runs_data)
st.dataframe(df_runs, hide_index=True, use_container_width=True)
